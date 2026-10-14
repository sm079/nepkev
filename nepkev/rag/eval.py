"""Evaluate the RAG layer on eval/rag_set.jsonl and write eval/report.md (+ report.json).

    uv run python -m nepkev.rag.eval --select-threshold --retrieval-only   # choose min_similarity on dev
    uv run python -m nepkev.rag.eval --yes                                  # + answers and the LLM judge (API calls)
    uv run python -m nepkev.rag.eval --retrieval-only --check eval/baseline.json   # CI gate, no API calls

Needs a migrated database with the knowledge base ingested (NEPKEV_DATABASE_URL; python -m nepkev.rag.ingest).

Retrieval (all test questions): recall@k and MRR of the answering article within the question's domain, and the
refusal gate: questions whose best chunk is less similar than min_similarity are refused without calling the model.
The threshold is chosen on the dev split only (nepkev.rag.evalset.best_threshold: keep every answerable dev question,
refuse as many must-refuse ones as possible) and written to configs/rag.yaml.

Answers (test questions, with --yes): the service's own pipeline (nepkev.rag.answer.Answerer.respond) on the same
retrieved chunks. Reported: refusal precision/recall end to end, how many citations quote their chunk verbatim (as the
model returned them, before the service drops bad ones), how many answers cite an answering article, and whether an
LLM judge finds each answer supported by what it cites. A hand-checked sample of judge verdicts (eval/judge_check.jsonl)
measures the judge. Model outputs are cached in eval/answers.jsonl and eval/judge.jsonl, so the report rebuilds
without new calls; changing the prompt, model or retrieved chunks invalidates the cache for that question.

95% intervals: bootstrap over topics (nepkev.rag.evalset.bootstrap).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from sqlalchemy import select

from .. import ROOT
from ..cli import load_dotenv
from ..service.db import make_engine, make_sessionmaker
from ..service.settings import Settings
from ..taxonomy import load_taxonomy
from .answer import Answerer, AnswerResult
from .config import load_rag_config, write_min_similarity
from .embed import HFEmbedder
from .evalset import (SCRIPTS, EvalRow, best_threshold, bootstrap, first_gold_rank, load_eval_set, mrr, percentile,
                      recall_at)
from .kb import kb_chunks, normalize
from .llm import AnthropicAnswerModel, Draft, load_prompt
from .store import Hit, KbChunk, search

EVAL_DIR = ROOT / "eval"
MAX_RANK = 10
JUDGE_MODEL = "claude-sonnet-5-5"
JUDGE_PROMPT = "judge_v1"
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["supported", "partially_supported", "unsupported"]},
                   "unsupported_statements": {"type": "array", "items": {"type": "string"}}},
    "required": ["verdict", "unsupported_statements"], "additionalProperties": False,
}


# ---------------------------------------------------------------- retrieval

def retrieve_all(db, rows: list[EvalRow], embedder) -> dict[str, dict]:
    out = {}
    for i, r in enumerate(rows, 1):
        t = time.perf_counter()
        hits = search(db, embedder.embed_queries([r.question])[0], r.domain, MAX_RANK)
        out[r.id] = {"hits": hits, "ms": (time.perf_counter() - t) * 1000,
                     "rank": None if r.must_refuse else first_gold_rank([h.doc_id for h in hits], r.gold_doc_ids)}
        print(f"retrieve {i}/{len(rows)}", end="\r", flush=True)
    print()
    return out


def gate_metrics(rows: list[EvalRow], top: dict[str, float], threshold: float) -> dict:
    """Counts over the list as given (a bootstrap sample repeats rows), never over a set of ids."""
    refused = [r for r in rows if top[r.id] < threshold]
    pos = [r for r in rows if r.must_refuse]
    answerable = len(rows) - len(pos)
    return {"refusal_precision": sum(r.must_refuse for r in refused) / len(refused) if refused else float("nan"),
            "refusal_recall": sum(top[r.id] < threshold for r in pos) / len(pos) if pos else float("nan"),
            "answerable_kept": sum(top[r.id] >= threshold for r in rows if not r.must_refuse) / answerable
            if answerable else float("nan")}


def retrieval_metrics(rows: list[EvalRow], ret: dict) -> dict:
    ranks = [ret[r.id]["rank"] for r in rows if not r.must_refuse]
    return {"n": len(ranks), "recall_at_1": recall_at(ranks, 1), "recall_at_3": recall_at(ranks, 3),
            "recall_at_5": recall_at(ranks, 5), "mrr_at_10": mrr(ranks)}


# ---------------------------------------------------------------- answers (cached)

class CachedModel:
    """Wraps an answer model; reuses a stored draft when model, prompt, question and chunks are unchanged."""

    def __init__(self, inner, path: Path, allow_calls: bool):
        self.inner, self.path, self.allow_calls = inner, path, allow_calls
        self.name, self.prompt_version = inner.name, inner.prompt_version
        self.cache = {d["key"]: d for d in _read_jsonl(path)}
        self.calls = 0

    def key(self, question: str, hits: list[Hit]) -> str:
        blob = json.dumps([self.name, load_prompt(self.prompt_version), question, [h.chunk_id for h in hits]],
                          ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def draft(self, question: str, hits: list[Hit], service: str) -> Draft:
        k = self.key(question, hits)
        if k not in self.cache:
            if not self.allow_calls:
                raise SystemExit("answers are not cached for this configuration; rerun with --yes to call the model")
            t = time.perf_counter()
            d = self.inner.draft(question, hits, service)
            self.cache[k] = {"key": k, "question": question, "draft": asdict(d),
                             "generate_ms": round((time.perf_counter() - t) * 1000, 1)}
            self.calls += 1
        return Draft(**self.cache[k]["draft"])

    def generate_ms(self, question: str, hits: list[Hit]) -> float:
        return self.cache[self.key(question, hits)]["generate_ms"]

    def save(self, used: set[str]) -> None:
        _write_jsonl(self.path, [v for k, v in sorted(self.cache.items(), key=lambda kv: kv[1]["question"]) if k in used])


def judge(client, question: str, result: AnswerResult) -> dict:
    cited = {c.chunk_id for c in result.citations}
    excerpts = "\n".join(f'<excerpt id="{h.chunk_id}">\n{h.title}\n\n{h.text}\n</excerpt>'
                         for h in result.hits if h.chunk_id in cited)
    resp = client.messages.create(
        model=JUDGE_MODEL, max_tokens=4000, system=load_prompt(JUDGE_PROMPT),
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": f"<question>\n{question}\n</question>\n\n<answer>\n{result.answer}\n"
                                              f"</answer>\n\n<cited_excerpts>\n{excerpts}\n</cited_excerpts>"}])
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"judge stopped with {resp.stop_reason}")
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


def answer_metrics(rows: list[EvalRow], results: dict[str, AnswerResult], verdicts: dict[str, dict]) -> dict:
    pos = [r for r in rows if r.must_refuse]
    refused = [r for r in rows if results[r.id].refused]
    answered = [r for r in rows if not results[r.id].refused]
    raw = [(r, cid, q) for r in rows if (d := results[r.id].draft) and not d.refused for cid, q in d.citations]

    def verbatim(r, cid, q) -> bool:
        hit = next((h for h in results[r.id].hits if h.chunk_id == cid), None)
        return hit is not None and bool(normalize(q)) and normalize(q) in hit.text

    judged = [r for r in answered if r.id in verdicts]
    return {
        "refusal_precision": sum(r.must_refuse for r in refused) / len(refused) if refused else float("nan"),
        "refusal_recall": sum(results[r.id].refused for r in pos) / len(pos) if pos else float("nan"),
        "answerable_answered": sum(not results[r.id].refused for r in rows if not r.must_refuse)
        / max(1, len(rows) - len(pos)),
        "answered": len(answered),
        "citations_returned": len(raw),
        "citation_verbatim": sum(verbatim(*x) for x in raw) / len(raw) if raw else float("nan"),
        "answers_citing_gold": sum(any(c.doc_id in r.gold_doc_ids for c in results[r.id].citations)
                                   for r in answered if not r.must_refuse) / max(1, sum(not r.must_refuse for r in answered)),
        "judge_supported": sum(verdicts[r.id]["verdict"] == "supported" for r in judged) / len(judged) if judged else float("nan"),
        "judge_partial": sum(verdicts[r.id]["verdict"] == "partially_supported" for r in judged) / len(judged) if judged else float("nan"),
        "judge_unsupported": sum(verdicts[r.id]["verdict"] == "unsupported" for r in judged) / len(judged) if judged else float("nan"),
        "refusal_reasons": {k: sum(results[r.id].refusal_reason == k for r in rows)
                            for k in ("low_similarity", "model_declined", "no_valid_citation")},
    }


# ---------------------------------------------------------------- report

def fmt(x: float, ci: list[float] | None = None) -> str:
    if x != x:   # nan
        return "n/a"
    return f"{x:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--retrieval-only", action="store_true", help="no answer model or judge calls")
    ap.add_argument("--select-threshold", action="store_true", help="choose min_similarity on dev and save it")
    ap.add_argument("--yes", action="store_true", help="allow API calls for answers and judge verdicts not cached yet")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "report.md")
    ap.add_argument("--write-baseline", type=Path, help="save the CI gate's metrics (recall@5, gate refusal recall)")
    ap.add_argument("--check", type=Path, help="exit 1 if recall@5 or the gate's refusal recall is below this baseline")
    args = ap.parse_args(argv)

    load_dotenv()   # ANTHROPIC_API_KEY for the answer model and the judge
    settings, cfg = Settings(), load_rag_config()
    rows = load_eval_set()
    dev, test = [r for r in rows if r.split == "dev"], [r for r in rows if r.split == "test"]
    embedder = HFEmbedder(cfg.embedding_model).load()
    engine = make_engine(settings.database_url)
    with make_sessionmaker(engine)() as db:
        stored = dict(db.execute(select(KbChunk.id, KbChunk.embedding_model)).all())
        if stored != {c.id: cfg.embedding_model for c in kb_chunks()}:
            raise SystemExit("kb_chunks does not match kb/ and the configured embedder: run python -m nepkev.rag.ingest")
        ret = retrieve_all(db, rows, embedder)
    engine.dispose()
    top = {r.id: ret[r.id]["hits"][0].score if ret[r.id]["hits"] else -1.0 for r in rows}

    chosen = best_threshold([top[r.id] for r in dev if not r.must_refuse], [top[r.id] for r in dev if r.must_refuse])
    if args.select_threshold:
        write_min_similarity(chosen)
        threshold = round(chosen, 6)
    elif cfg.min_similarity is None:
        raise SystemExit("configs/rag.yaml has no min_similarity yet: run with --select-threshold")
    else:
        threshold = cfg.min_similarity

    unit = lambda r: r.unit  # noqa: E731
    rm = retrieval_metrics(test, ret)
    rm_ci = bootstrap([r for r in test if not r.must_refuse], unit, {
        "recall_at_5": lambda s: retrieval_metrics(s, ret)["recall_at_5"], "mrr_at_10": lambda s: retrieval_metrics(s, ret)["mrr_at_10"]})
    gm = gate_metrics(test, top, threshold)
    gm_ci = bootstrap(test, unit, {k: (lambda s, k=k: gate_metrics(s, top, threshold)[k]) for k in gm})
    report = {"threshold": threshold, "threshold_chosen_on_dev": round(chosen, 6), "embedding_model": cfg.embedding_model,
              "top_k": cfg.top_k, "n": {"dev": len(dev), "test": len(test),
                                        "test_answerable": sum(not r.must_refuse for r in test),
                                        "test_must_refuse": sum(r.must_refuse for r in test)},
              "retrieval": rm, "retrieval_ci95": rm_ci, "retrieval_by_script": {
                  sc: retrieval_metrics([r for r in test if r.script == sc], ret) for sc in SCRIPTS},
              "gate": gm, "gate_ci95": gm_ci, "gate_dev": gate_metrics(dev, top, threshold),
              "retrieve_ms": {"p50": percentile([ret[r.id]["ms"] for r in test], 0.5),
                              "p95": percentile([ret[r.id]["ms"] for r in test], 0.95)}}

    baseline = {"recall_at_5": rm["recall_at_5"], "gate_refusal_recall": gm["refusal_recall"]}
    if args.check:
        want = json.loads(args.check.read_text(encoding="utf-8"))
        bad = [f"{k} {baseline[k]:.4f} < baseline {want[k]:.4f}" for k in want if baseline[k] < want[k] - 1e-9]
        print(json.dumps(baseline, indent=2))
        if bad:
            sys.exit("RAG quality gate failed: " + "; ".join(bad))
        print("RAG quality gate passed")
        return
    if args.write_baseline:
        args.write_baseline.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8", newline="\n")

    results, verdicts, latency = {}, {}, {}
    if not args.retrieval_only:
        import anthropic

        tax = load_taxonomy()
        model = CachedModel(AnthropicAnswerModel(settings.llm_model, effort=settings.llm_effort, prompt_version=cfg.prompt),
                            EVAL_DIR / "answers.jsonl", args.yes)
        answerer = Answerer(embedder, model, cfg.top_k, threshold)
        used = set()
        for i, r in enumerate(test, 1):
            hits = ret[r.id]["hits"][: cfg.top_k]
            res = answerer.respond(r.question, hits, tax.domain(r.domain).name)
            results[r.id] = res
            gen = 0.0
            if res.draft is not None:
                used.add(model.key(r.question, hits))
                gen = model.generate_ms(r.question, hits)
            latency[r.id] = {"retrieve": ret[r.id]["ms"], "generate": gen if res.draft is not None else None}
            print(f"answer {i}/{len(test)}", end="\r", flush=True)
        print()
        model.save(used)

        judge_path, client = EVAL_DIR / "judge.jsonl", anthropic.Anthropic(max_retries=2)
        cache = {d["id"]: d for d in _read_jsonl(judge_path)}
        answered = [r for r in test if not results[r.id].refused]
        for i, r in enumerate(answered, 1):
            res = results[r.id]
            k = hashlib.sha256(json.dumps([JUDGE_MODEL, load_prompt(JUDGE_PROMPT), r.question, res.answer,
                                           [c.chunk_id for c in res.citations]], ensure_ascii=False).encode()).hexdigest()[:16]
            if cache.get(r.id, {}).get("key") != k:
                if not args.yes:
                    raise SystemExit("judge verdicts are not cached for these answers; rerun with --yes")
                cache[r.id] = {"id": r.id, "key": k, "question": r.question, "answer": res.answer, **judge(client, r.question, res)}
            verdicts[r.id] = cache[r.id]
            print(f"judge {i}/{len(answered)}", end="\r", flush=True)
        print()
        _write_jsonl(judge_path, [cache[r.id] for r in answered])

        am = answer_metrics(test, results, verdicts)
        am_ci = bootstrap(test, unit, {k: (lambda s, k=k: answer_metrics(s, results, verdicts)[k]) for k in (
            "refusal_precision", "refusal_recall", "answerable_answered", "citation_verbatim", "answers_citing_gold",
            "judge_supported")})
        checks = {d["id"]: d for d in _read_jsonl(EVAL_DIR / "judge_check.jsonl")}
        agree = [checks[i]["verdict"] == verdicts[i]["verdict"] for i in checks if i in verdicts]
        gen = [v["generate"] for v in latency.values() if v["generate"] is not None]
        tot = [v["retrieve"] + v["generate"] for v in latency.values() if v["generate"] is not None]
        report |= {"answer_model": model.name, "prompt": model.prompt_version, "judge_model": JUDGE_MODEL,
                   "answers": am, "answers_ci95": am_ci, "judge_check": {"n": len(agree), "agree": sum(agree)},
                   "latency_ms": {"generate": {"p50": percentile(gen, 0.5), "p95": percentile(gen, 0.95)},
                                  "retrieve_plus_generate": {"p50": percentile(tot, 0.5), "p95": percentile(tot, 0.95)}},
                   "new_model_calls": model.calls}

    (args.out.with_suffix(".json")).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8", newline="\n")
    args.out.write_text(render(report, test, results), encoding="utf-8", newline="\n")
    print(args.out.read_text(encoding="utf-8"))


def render(rep: dict, test: list[EvalRow], results: dict[str, AnswerResult]) -> str:
    n, r, ci = rep["n"], rep["retrieval"], rep["retrieval_ci95"]
    g, gci = rep["gate"], rep["gate_ci95"]
    lines = [
        "# RAG evaluation", "",
        "Synthetic knowledge base (`kb/`, demo policies) and a hand-written question set (`eval/rag_set.jsonl`). "
        f"Test split: {n['test']} questions ({n['test_answerable']} answerable, {n['test_must_refuse']} must refuse); "
        f"dev split: {n['dev']} questions, used only to choose the refusal threshold. 95% intervals: bootstrap over "
        "topics. Produced by `python -m nepkev.rag.eval`.", "",
        f"Embedding model `{rep['embedding_model']}`, top {rep['top_k']} chunks of the question's domain, refusal "
        f"threshold {rep['threshold']:.4f} (chosen on dev: keep every answerable question, refuse as many others as "
        "possible).", "",
        "## Retrieval (test)", "",
        "| recall@1 | recall@3 | recall@5 | MRR@10 |", "|---|---|---|---|",
        f"| {fmt(r['recall_at_1'])} | {fmt(r['recall_at_3'])} | {fmt(r['recall_at_5'], ci['recall_at_5'])} | "
        f"{fmt(r['mrr_at_10'], ci['mrr_at_10'])} |", "",
        "| script | answerable | recall@1 | recall@5 | MRR@10 |", "|---|---|---|---|---|",
        *(f"| {sc} | {m['n']} | {fmt(m['recall_at_1'])} | {fmt(m['recall_at_5'])} | {fmt(m['mrr_at_10'])} |"
          for sc, m in rep["retrieval_by_script"].items()), "",
        "## Refusal gate (similarity only, no model call)", "",
        "| split | refusal precision | refusal recall | answerable questions kept |", "|---|---|---|---|",
        f"| test | {fmt(g['refusal_precision'], gci['refusal_precision'])} | {fmt(g['refusal_recall'], gci['refusal_recall'])} "
        f"| {fmt(g['answerable_kept'], gci['answerable_kept'])} |",
        f"| dev | {fmt(rep['gate_dev']['refusal_precision'])} | {fmt(rep['gate_dev']['refusal_recall'])} | "
        f"{fmt(rep['gate_dev']['answerable_kept'])} |", "",
        f"Retrieval latency (embed the question + pgvector search), p50 / p95: {rep['retrieve_ms']['p50']:,.0f} / "
        f"{rep['retrieve_ms']['p95']:,.0f} ms.", ""]
    if "answers" in rep:
        a, aci = rep["answers"], rep["answers_ci95"]
        rr = a["refusal_reasons"]
        jc = rep["judge_check"]
        lines += [
            f"## Answers (test, `{rep['answer_model']}`, prompt `{rep['prompt']}`)", "",
            "| metric | value |", "|---|---|",
            f"| refusal precision (gate + model + citation check) | {fmt(a['refusal_precision'], aci['refusal_precision'])} |",
            f"| refusal recall | {fmt(a['refusal_recall'], aci['refusal_recall'])} |",
            f"| answerable questions answered | {fmt(a['answerable_answered'], aci['answerable_answered'])} |",
            f"| citations quoting their chunk verbatim (as returned, {a['citations_returned']} citations) | "
            f"{fmt(a['citation_verbatim'], aci['citation_verbatim'])} |",
            f"| answers citing an answering article | {fmt(a['answers_citing_gold'], aci['answers_citing_gold'])} |",
            f"| judged supported by the cited chunks ({a['answered']} answers) | {fmt(a['judge_supported'], aci['judge_supported'])} |",
            f"| judged partially supported / unsupported | {fmt(a['judge_partial'])} / {fmt(a['judge_unsupported'])} |", "",
            f"Refusals by stage: {rr['low_similarity']} by the similarity gate, {rr['model_declined']} by the model, "
            f"{rr['no_valid_citation']} for lack of a verbatim citation.", "",
            f"Judge: `{rep['judge_model']}`. Hand check: {jc['agree']} of {jc['n']} judge verdicts agree with a "
            "reading of the answer against its cited chunks (20 judged answers taken at a fixed stride, plus every one the judge did "
            "not call supported; `eval/judge_check.jsonl`).", "",
            f"Latency p50 / p95 for questions that reach the model: answer generation "
            f"{ms_pair(rep['latency_ms']['generate'])} ms, retrieval plus generation "
            f"{ms_pair(rep['latency_ms']['retrieve_plus_generate'])} ms. Refusals at the gate take only the retrieval "
            "time. Routing is separate (see the API section of the README).", ""]
        misses = [x for x in test if not x.must_refuse and results[x.id].refused] + \
                 [x for x in test if x.must_refuse and not results[x.id].refused]
        if misses:
            lines += ["## Wrong refusal decisions (test)", "", "| id | script | question | expected | got |", "|---|---|---|---|---|"]
            for x in misses:
                res = results[x.id]
                got = f"refused ({res.refusal_reason})" if res.refused else "answered"
                lines.append(f"| {x.id} | {x.script} | {x.question} | {'refuse' if x.must_refuse else 'answer'} | {got} |")
            lines.append("")
    return "\n".join(lines)


def ms_pair(d: dict) -> str:
    return f"{d['p50']:,.0f} / {d['p95']:,.0f}" if d["p50"] == d["p50"] else "n/a"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
