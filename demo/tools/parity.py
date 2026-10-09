"""Score an exported ONNX backbone the way the browser demo does and compare it with Kev's own predictions.

Each question runs as one causal row (Kev's row form for the hybrid Qwen3.5 backbone):
  <|fim_prefix|> state <|fim_middle|> instruction (<|box_start|> option <|box_end|>)... <|fim_suffix|>
read out at <|fim_suffix|> (<decide>) and at every <|box_end|> (</opt>), then through the pointer head.

  python demo/tools/parity.py --model build/webgpu/onnx-int4-webgpu --head build/webgpu/merged \
      --kev data/mix-v3-add/kev/test.jsonl --rows runs/mix-v3-add/runs/selected-test/rows.json --n 200

Needs onnxruntime, numpy and transformers (for the tokenizer); not part of the project environment.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

STATE, Q, OPT, END_OPT, DECIDE = "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
DTYPES = {1: np.float32, 10: np.float16, 7: np.int64}


def render(v) -> str:
    """kev.api.render for the shapes this project sends (strings; None as empty)."""
    return "" if v is None else str(v)


def option_text(name: str, desc) -> str:
    return name if desc in (None, "") else f"{name}: {render(desc)}"


def questions(req: dict) -> list[dict]:
    """kev.api.to_record: the option strings and the keys their probabilities are reported under."""
    out = []
    for qid, q in req["questions"].items():
        if q["type"] == "noul":
            c = q.get("criteria") or {}
            opts, keys = [option_text("no", c.get("false")), option_text("yes", c.get("true"))], ["false", "true"]
        else:
            opts, keys = [option_text(k, v) for k, v in q["criteria"].items()], list(q["criteria"])
        out.append({"id": qid, "instr": render(q.get("instructions")), "options": opts, "keys": keys})
    return out


class Runner:
    def __init__(self, model_dir: Path, head_dir: Path, provider: str):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), providers=[provider])
        self.special = {t: self.tok.convert_tokens_to_ids(t) for t in (STATE, Q, OPT, END_OPT, DECIDE)}
        meta = json.loads((head_dir / "head.json").read_text(encoding="utf-8"))
        d, dp = meta["hidden_size"], meta["pointer_dim"]
        flat = np.frombuffer((head_dir / "head.bin").read_bytes(), dtype="<f4")
        sizes = [dp * d, dp, dp * d, dp]
        qw, qb, kw, kb = np.split(flat, np.cumsum(sizes)[:-1])
        self.qw, self.qb, self.kw, self.kb = qw.reshape(dp, d), qb, kw.reshape(dp, d), kb
        self.scale, self.temperature = 1 / np.sqrt(dp), meta["temperature"]

    def user_tokens(self, text: str) -> list[int]:
        return self.tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids

    def row(self, state: str, q: dict) -> tuple[list[int], list[int]]:
        s = self.special
        ids = [s[STATE], *self.user_tokens(state), s[Q], *self.user_tokens(q["instr"])]
        ends = []
        for o in q["options"]:
            ids += [s[OPT], *self.user_tokens(o), s[END_OPT]]
            ends.append(len(ids) - 1)
        ids.append(s[DECIDE])
        return ids, ends

    def feeds(self, ids: list[int]) -> dict:
        n = len(ids)
        f = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, n), dtype=np.int64),
             "position_ids": np.broadcast_to(np.arange(n, dtype=np.int64), (3, 1, n)).copy()}
        for i in self.sess.get_inputs():
            if i.name in f:
                continue
            shape = [1 if d == "batch_size" else 0 if d == "past_sequence_length" else 256 if d == "kv_cache_dim" else d
                     for d in i.shape]
            f[i.name] = np.zeros(shape, dtype=DTYPES[{"tensor(float)": 1, "tensor(float16)": 10}[i.type]])
        return f

    def probs(self, state: str, q: dict) -> np.ndarray:
        ids, ends = self.row(state, q)
        h = self.sess.run(["hidden_states"], self.feeds(ids))[0][0].astype(np.float32)
        qv = self.qw @ h[-1] + self.qb
        kv = h[ends] @ self.kw.T + self.kb
        z = (kv @ qv) * self.scale / self.temperature
        z = np.exp(z - z.max())
        return z / z.sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--ref-model", type=Path, help="another export to compare against (e.g. the fp32 one)")
    ap.add_argument("--ref-cache", type=Path, help="JSON of the reference model's probabilities; written on first use")
    ap.add_argument("--head", required=True, type=Path)
    ap.add_argument("--kev", required=True, type=Path, help="Kev evaluation export (one request per line)")
    ap.add_argument("--rows", type=Path, help="kev.benchmark rows.json scored on the same export")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    args = ap.parse_args()

    run = Runner(args.model, args.head, args.provider)
    cache = json.loads(args.ref_cache.read_text(encoding="utf-8")) if args.ref_cache and args.ref_cache.exists() else {}
    fresh = not cache
    same = args.ref_model is not None and args.ref_model.resolve() == args.model.resolve()
    other = (run if same else Runner(args.ref_model, args.head, args.provider)) if args.ref_model and fresh else None
    have_ref = bool(cache) or other is not None
    ref = {(r["id"], r["question"]): np.array(r["p"]) for r in json.loads(args.rows.read_text(encoding="utf-8"))} if args.rows else {}
    reqs = [json.loads(l) for l in args.kev.read_text(encoding="utf-8").splitlines() if l.strip()]
    step = max(1, len(reqs) // args.n)
    picked = list(range(0, len(reqs), step))[: args.n]

    stats = {name: [0.0, 0] for name in ("ref-model", "rows")}   # max |dp|, argmax flips
    n_q, correct, ref_correct, scored, t = 0, 0, 0, 0, 0.0
    for done, line in enumerate(picked, 1):
        req = reqs[line]
        for q in questions(req):
            t0 = time.perf_counter(); p = run.probs(req["state"], q); t += time.perf_counter() - t0
            n_q += 1
            key = f"{line}/{q['id']}"
            if other is not None:
                cache[key] = (p if same else other.probs(req["state"], q)).tolist()
            refs = {"ref-model": np.array(cache[key]) if key in cache else None, "rows": ref.get((f"custom/{line}", q["id"]))}
            for name, rp in refs.items():
                if rp is not None:
                    stats[name][0] = max(stats[name][0], float(np.abs(p - rp).max()))
                    stats[name][1] += int(p.argmax() != rp.argmax())
            src = req["questions"][q["id"]]
            if q["id"] == "q1" and src.get("target") is None:
                scored += 1; correct += int(q["keys"][int(p.argmax())] == src["label"])
                if refs["ref-model"] is not None:
                    ref_correct += int(q["keys"][int(refs["ref-model"].argmax())] == src["label"])
        if done % 20 == 0 or done == len(picked):
            parts = [f"vs {name}: max|dp| {v[0]:.4f}, flips {v[1]}/{n_q}" for name, v in stats.items() if refs[name] is not None]
            acc = f"issue acc {correct / max(scored, 1):.3f}" + (f" (ref-model {ref_correct / max(scored, 1):.3f})" if have_ref else "")
            print(f"{done}/{len(picked)} records  {'  '.join(parts)}  {acc} on {scored}  {1000 * t / n_q:.0f} ms/question", flush=True)

    if args.ref_cache and fresh and cache:
        args.ref_cache.write_text(json.dumps(cache), encoding="utf-8")


if __name__ == "__main__":
    main()
