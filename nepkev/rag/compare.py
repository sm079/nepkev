"""Compare embedding models on the eval set's retrieval task, in memory (no database), and write a markdown table.

    uv run python -m nepkev.rag.compare --out eval/embedders.md

For every model: recall@1/@5 and MRR of the answering article among the question's domain (dev and test, and test by
script), how well the top similarity separates answerable from must-refuse questions (AUC), the vector size and the
time to embed one query on CPU. The model choice is made on dev; test is shown for the record.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .embed import MODELS, HFEmbedder
from .evalset import SCRIPTS, auc, first_gold_rank, load_eval_set, mrr, percentile, recall_at
from .kb import kb_chunks


def evaluate_model(name: str, rows, chunks) -> dict:
    emb = HFEmbedder(name).load()
    passages = emb.embed_passages([c.embed_text for c in chunks])
    timings, out = [], []
    emb.embed_queries([rows[0].question])   # warm-up
    for r in rows:
        t = time.perf_counter()
        q = emb.embed_queries([r.question])[0]
        timings.append((time.perf_counter() - t) * 1000)
        idx = [i for i, c in enumerate(chunks) if c.domain == r.domain]
        scores = passages[idx] @ q
        order = np.argsort(-scores)
        out.append({"row": r, "top": float(scores[order[0]]),
                    "rank": None if r.must_refuse else first_gold_rank([chunks[idx[i]].doc_id for i in order],
                                                                       r.gold_doc_ids)})
    return {"name": name, "dim": emb.dim, "query_ms_p50": percentile(timings, 0.5), "results": out}


def summarize(res: dict, split: str, script: str | None = None) -> dict:
    rs = [x for x in res["results"] if x["row"].split == split and (script is None or x["row"].script == script)]
    ans = [x for x in rs if not x["row"].must_refuse]
    ranks = [x["rank"] for x in ans]
    return {"n": len(ans), "r1": recall_at(ranks, 1), "r5": recall_at(ranks, 5), "mrr": mrr(ranks),
            "auc": auc([x["top"] for x in ans], [x["top"] for x in rs if x["row"].must_refuse]) if script is None else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", type=Path, default=Path("eval/embedders.md"))
    args = ap.parse_args()

    rows, chunks = load_eval_set(), kb_chunks()
    results = []
    for i, name in enumerate(args.models, 1):
        print(f"{i}/{len(args.models)} {name}", flush=True)
        results.append(evaluate_model(name, rows, chunks))

    lines = ["# Embedding model comparison", "",
             f"Retrieval of the answering article among the question's domain ({len(chunks)} chunks in all), "
             "in memory with cosine similarity. AUC: how well the top similarity separates answerable from "
             "must-refuse questions. Query time: one query embedded on CPU, median. Produced by "
             "`python -m nepkev.rag.compare`.", ""]
    for split in ("dev", "test"):
        lines += [f"### {split}", "", "| model | dim | recall@1 | recall@5 | MRR | refusal AUC | query ms |",
                  "|---|---|---|---|---|---|---|"]
        for res in results:
            s = summarize(res, split)
            lines.append(f"| `{res['name']}` | {res['dim']} | {s['r1']:.3f} | {s['r5']:.3f} | {s['mrr']:.3f} | "
                         f"{s['auc']:.3f} | {res['query_ms_p50']:.0f} |")
        lines.append("")
    lines += ["### test recall@5 by script", "", "| model | " + " | ".join(SCRIPTS) + " |", "|---" * (len(SCRIPTS) + 1) + "|"]
    for res in results:
        cells = [f"{summarize(res, 'test', sc)['r5']:.3f} (n={summarize(res, 'test', sc)['n']})" for sc in SCRIPTS]
        lines.append(f"| `{res['name']}` | " + " | ".join(cells) + " |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
