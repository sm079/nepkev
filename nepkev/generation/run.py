"""Resumable generation and verification runs.

Layout of a run directory (under data/, never committed):

    scenarios.jsonl     sampled scenarios (deterministic from the config)
    raw.jsonl           one line per rendered scenario, appended as requests finish
    usage.jsonl         one line per request: model, tokens, cost
    failures.jsonl      requests that failed (retried on the next invocation)
    batches.json        submitted Message Batches, when --batch is used
    records.jsonl       canonical records built from scenarios + raw, with the verdicts merged in
    verify.jsonl        verifier verdicts, one line per record
    adjudicator.jsonl   adjudicator verdicts on the records the verifier disputed

Re-running a command skips work already on disk, so an interrupted run continues where it stopped, and a run whose
model outputs are all on disk rebuilds without any API call.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path

from ..config import GenConfig, ModelCall, Price
from ..records import Labels, Record, Source, read_jsonl, write_jsonl
from ..scenarios import Scenario, generate_scenarios
from ..taxonomy import Taxonomy
from .prompts import (GENERATION_SCHEMA, VERIFICATION_SCHEMA, generation_user_message, system_prompt,
                      verification_user_message)
from .providers import OutputError, RefusalError, Result, Usage, make_provider


def cost(usage: Usage, price: Price | None, batch: bool = False) -> float | None:
    if price is None:
        return None
    usd = (usage.input_tokens * price.input + usage.output_tokens * price.output
           + usage.cache_read_tokens * price.cache_read + usage.cache_write_tokens * price.cache_write) / 1e6
    return round(usd * (0.5 if batch else 1.0), 6)


def _done_ids(path: Path, key: str) -> set[str]:
    return {r[key] for r in read_jsonl(path)} if path.exists() else set()


def load_or_sample_scenarios(cfg: GenConfig, tax: Taxonomy, out: Path) -> list[Scenario]:
    path = out / "scenarios.jsonl"
    fresh = generate_scenarios(cfg, tax)
    if path.exists():
        existing = [Scenario.model_validate(d) for d in read_jsonl(path)]
        if [s.model_dump() for s in existing] != [s.model_dump() for s in fresh]:
            raise RuntimeError(f"{path} differs from what {cfg.name} samples now; use a new run directory")
        return existing
    write_jsonl(path, fresh)
    return fresh


async def _run_requests(provider, call: ModelCall, jobs: list[tuple[str, str, str, dict]],
                        on_result: Callable[[str, Result], None], on_failure: Callable[[str, Exception], None],
                        label: str) -> None:
    sem = asyncio.Semaphore(call.concurrency)
    done = 0

    async def one(job_id: str, system: str, user: str, schema: dict) -> None:
        nonlocal done
        async with sem:
            for attempt in range(2):          # one extra attempt for truncated/invalid output; the SDK retries HTTP errors
                try:
                    on_result(job_id, await provider.complete_json(system, user, schema))
                    break
                except OutputError as e:
                    if attempt == 1:
                        on_failure(job_id, e)
                except (RefusalError, Exception) as e:   # noqa: BLE001 - recorded and retried on the next invocation
                    on_failure(job_id, e)
                    break
            done += 1
            print(f"{label} {done}/{len(jobs)}", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))


def _log_failure(out: Path) -> Callable[[str, Exception], None]:
    def on_failure(job_id: str, e: Exception) -> None:
        write_jsonl(out / "failures.jsonl", [{"job_id": job_id, "error": f"{type(e).__name__}: {e}", "t": time.time()}],
                    append=True)
    return on_failure


# ---------------------------------------------------------------- generation

def generate(cfg: GenConfig, tax: Taxonomy, out: Path, limit: int | None = None, batch: bool = False,
             scenarios: list[Scenario] | None = None, system: str | None = None,
             user_message: Callable[[list[Scenario], Taxonomy], str] = generation_user_message) -> dict:
    """Render scenarios (the config's own sample unless `scenarios` is given) with the generator call."""
    out.mkdir(parents=True, exist_ok=True)
    if scenarios is None:
        scenarios = load_or_sample_scenarios(cfg, tax, out)
    elif not (out / "scenarios.jsonl").exists():
        write_jsonl(out / "scenarios.jsonl", scenarios)
    raw_path, usage_path = out / "raw.jsonl", out / "usage.jsonl"
    done = _done_ids(raw_path, "scenario_id")
    pending = [s for s in scenarios if s.scenario_id not in done]
    if limit is not None:
        pending = pending[:limit]
    call = cfg.generator
    k = call.scenarios_per_request
    chunks = {f"g{i // k:05d}-{pending[i].scenario_id}": pending[i:i + k] for i in range(0, len(pending), k)}
    print(f"generate: {len(scenarios)} scenarios, {len(done)} done, {len(pending)} pending -> {len(chunks)} requests "
          f"({call.provider}:{call.model}{', batch' if batch else ''})", flush=True)
    if not chunks:
        return summarize(out)
    system = system or system_prompt(call.prompt, tax)
    jobs = [(jid, system, user_message(c, tax), GENERATION_SCHEMA) for jid, c in chunks.items()]
    provider = make_provider(call)

    def on_result(job_id: str, res: Result, batched: bool = False) -> None:
        # a batch collected by a later invocation may name jobs this invocation did not plan; accept any known scenario
        want = {s.scenario_id for s in chunks[job_id]} if job_id in chunks else {s.scenario_id for s in scenarios}
        items = [it for it in res.data.get("items", []) if it.get("scenario_id") in want and it["scenario_id"] not in done]
        done.update(it["scenario_id"] for it in items)
        write_jsonl(raw_path, [{**it, "provider": call.provider, "model": res.model, "requested_model": call.model,
                                "prompt_version": call.prompt, "request_id": res.request_id, "job_id": job_id}
                               for it in items], append=True)
        write_jsonl(usage_path, [{"job_id": job_id, "role": "generate", "model": res.model, "batch": batched,
                                  **res.usage.__dict__, "usd": cost(res.usage, cfg.prices.get(res.model), batched),
                                  "scenarios_returned": len(items), "scenarios_requested": len(want), "t": time.time()}],
                    append=True)

    if batch:
        return _batch_flow(provider, out, jobs, on_result, _log_failure(out))
    asyncio.run(_run_requests(provider, call, jobs, on_result, _log_failure(out), "generate"))
    return summarize(out)


def _batch_flow(provider, out: Path, jobs, on_result, on_failure) -> dict:
    """Submit pending jobs as one Message Batch, or collect a batch that finished. Call again to poll."""
    state_path = out / "batches.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else []
    open_batches = [b for b in state if not b.get("collected")]
    if not open_batches:
        bid = provider.submit_batch(jobs)
        state.append({"id": bid, "jobs": [j[0] for j in jobs], "submitted": time.time()})
        state_path.write_text(json.dumps(state, indent=1))
        print(f"submitted batch {bid} with {len(jobs)} requests; run the same command again to collect", flush=True)
        return {"batch": bid, "status": "submitted"}
    for b in open_batches:
        status, counts = provider.batch_status(b["id"])
        print(f"batch {b['id']}: {status} {counts}", flush=True)
        if status != "ended":
            continue
        for cid, res in provider.batch_results(b["id"]):
            if isinstance(res, Exception):
                on_failure(cid, res)
            else:
                on_result(cid, res, True)
        b["collected"] = time.time()
    state_path.write_text(json.dumps(state, indent=1))
    return {"batches": [{"id": b["id"], "collected": bool(b.get("collected"))} for b in state]}


def summarize(out: Path) -> dict:
    usage = list(read_jsonl(out / "usage.jsonl")) if (out / "usage.jsonl").exists() else []
    fails = list(read_jsonl(out / "failures.jsonl")) if (out / "failures.jsonl").exists() else []
    tok = {k: sum(u[k] for u in usage) for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
    usd = [u["usd"] for u in usage if u.get("usd") is not None]
    return {"scenarios": sum(1 for _ in read_jsonl(out / "scenarios.jsonl")),
            "rendered": len(_done_ids(out / "raw.jsonl", "scenario_id")), "requests": len(usage),
            "failed_requests_logged": len(fails), "tokens": tok, "usd": round(sum(usd), 4) if usd else None}


# ---------------------------------------------------------------- canonical records

def build_records(cfg: GenConfig, out: Path) -> list[Record]:
    """Devanagari records from scenarios + raw renderings (first rendering of a scenario wins)."""
    scen = {s["scenario_id"]: Scenario.model_validate(s) for s in read_jsonl(out / "scenarios.jsonl")}
    recs: list[Record] = []
    seen: set[str] = set()
    for row in read_jsonl(out / "raw.jsonl"):
        s = scen.get(row["scenario_id"])
        if s is None or s.scenario_id in seen:
            continue
        seen.add(s.scenario_id)
        mock = row["provider"] == "dry_run"
        recs.append(Record(
            record_id=f"{s.scenario_id}-deva", scenario_id=s.scenario_id, scenario_group_id=s.scenario_group_id,
            domain=s.domain, taxonomy_version=s.taxonomy_version, scenario_facts=s.facts,
            visible_facts=s.visible_facts, withheld_facts=s.withheld_facts, message=row["message"].strip(), script="deva",
            style_tags=[s.style["length"], s.style["tone"], s.style["formality"],
                        *(["negation"] if s.family == "negation" else []), *(["mock"] if mock else [])],
            difficulty_family=s.family,
            labels=Labels(issue=s.issue, issue_target=s.issue_target, clarification_needed=s.clarification_needed),
            issues_all=s.issues_all, stated_priority=s.stated_priority,
            source=Source(generator="dry-run" if mock else "llm", provider=row["provider"], model=row["model"],
                          prompt_version=row["prompt_version"], seed=cfg.seed),
            note=row.get("evidence", "")))
    return recs


# ---------------------------------------------------------------- verification

def disputed(recs: list[Record]) -> list[Record]:
    return [r for r in recs if (r.validation.checks.get("verifier") or {}).get("agrees") is False]


def verify(cfg: GenConfig, tax: Taxonomy, recs: list[Record], out: Path, role: str = "verifier") -> dict:
    """Labels from a model that sees only the text, merged into validation.checks[role].

    role "verifier": every generated message, by a different model than the generator.
    role "adjudicator": only messages the verifier disputed, by a stronger model; its verdict decides them.
    """
    call = getattr(cfg, role)
    if call is None:
        raise RuntimeError(f"config {cfg.name} has no {role}")
    path = out / ("verify.jsonl" if role == "verifier" else f"{role}.jsonl")
    done = _done_ids(path, "id")
    pending = [r for r in recs if r.record_id not in done]
    k = call.messages_per_request
    chunks = {f"v{i // k:05d}-{pending[i].record_id}": pending[i:i + k] for i in range(0, len(pending), k)}
    print(f"{role}: {len(recs)} records, {len(done)} done -> {len(chunks)} requests ({call.provider}:{call.model})", flush=True)
    if chunks:
        system = system_prompt(call.prompt, tax)
        jobs = [(jid, system, verification_user_message([{"id": r.record_id, "service": r.domain, "message": r.message}
                                                         for r in c]), VERIFICATION_SCHEMA) for jid, c in chunks.items()]

        def on_result(job_id: str, res: Result) -> None:
            want = {r.record_id for r in chunks[job_id]}
            write_jsonl(path, [{**it, "model": res.model} for it in res.data.get("items", []) if it.get("id") in want],
                        append=True)
            write_jsonl(out / "usage.jsonl", [{"job_id": job_id, "role": role, "model": res.model, "batch": False,
                                               **res.usage.__dict__, "usd": cost(res.usage, cfg.prices.get(res.model)),
                                               "t": time.time()}], append=True)

        asyncio.run(_run_requests(make_provider(call), call, jobs, on_result, _log_failure(out), role))
    return merge_verdicts(recs, path, role)


def merge_verdicts(recs: list[Record], path: Path, role: str = "verifier") -> dict:
    verdicts = {v["id"]: v for v in read_jsonl(path)} if path.exists() else {}
    stats = {"verified": 0, "agree": 0, "disagree": 0, "mock": 0}
    for r in recs:
        v = verdicts.get(r.record_id)
        if v is None:
            continue
        if v.get("_mock"):
            stats["mock"] += 1
            r.validation.checks[role] = {"model": v["model"], "mock": True}
            continue
        agrees = (v["issue"] == r.labels.issue and v["clarification_needed"] == r.labels.clarification_needed
                  and v["language_ok"])
        r.validation.checks[role] = {"model": v["model"], "issue": v["issue"], "clarification_needed": v["clarification_needed"],
                                     "issues_mentioned": v["issues_mentioned"], "language_ok": v["language_ok"], "agrees": agrees}
        stats["verified"] += 1
        stats["agree" if agrees else "disagree"] += 1
    return stats


# ---------------------------------------------------------------- budget estimate

def estimate(cfg: GenConfig, tax: Taxonomy) -> dict:
    n_sc = len(generate_scenarios(cfg, tax))
    e, g, v = cfg.estimate, cfg.generator, cfg.verifier
    gen_req = -(-n_sc // g.scenarios_per_request)
    ver_req = -(-n_sc // v.messages_per_request)
    gen = Usage(input_tokens=n_sc * e.user_tokens_per_scenario,
                output_tokens=n_sc * e.output_tokens_per_scenario + gen_req * e.thinking_tokens_per_request,
                cache_read_tokens=max(0, gen_req - 1) * e.system_tokens, cache_write_tokens=e.system_tokens)
    ver = Usage(input_tokens=n_sc * e.verify_user_tokens_per_message,
                output_tokens=n_sc * e.verify_output_tokens_per_message + ver_req * e.thinking_tokens_per_request,
                cache_read_tokens=max(0, ver_req - 1) * e.system_tokens, cache_write_tokens=e.system_tokens)
    gp, vp = cfg.prices.get(g.model), cfg.prices.get(v.model)
    return {"scenarios": n_sc, "romanized_twins_before_checks": round(n_sc * cfg.roman_share),
            "generation": {"model": g.model, "requests": gen_req, "tokens": gen.__dict__,
                           "usd_standard": cost(gen, gp), "usd_batch": cost(gen, gp, batch=True)},
            "verification": {"model": v.model, "requests": ver_req, "tokens": ver.__dict__,
                             "usd_standard": cost(ver, vp), "usd_batch": cost(ver, vp, batch=True)},
            "note": "From the config's estimate block; thinking tokens vary. Cache writes assume one per run."}
