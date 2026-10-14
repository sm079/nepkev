# RAG evaluation

Synthetic knowledge base (`kb/`, demo policies) and a hand-written question set (`eval/rag_set.jsonl`). Test split: 95 questions (78 answerable, 17 must refuse); dev split: 31 questions, used only to choose the refusal threshold. 95% intervals: bootstrap over topics. Produced by `python -m nepkev.rag.eval`.

Embedding model `intfloat/multilingual-e5-small`, top 5 chunks of the question's domain, refusal threshold 0.8030 (chosen on dev: keep every answerable question, refuse as many others as possible).

## Retrieval (test)

| recall@1 | recall@3 | recall@5 | MRR@10 |
|---|---|---|---|
| 0.795 | 0.910 | 0.936 [0.867, 0.987] | 0.862 [0.784, 0.925] |

| script | answerable | recall@1 | recall@5 | MRR@10 |
|---|---|---|---|---|
| en | 28 | 0.929 | 1.000 | 0.964 |
| deva | 24 | 0.750 | 0.917 | 0.823 |
| roman | 26 | 0.692 | 0.885 | 0.787 |

## Refusal gate (similarity only, no model call)

| split | refusal precision | refusal recall | answerable questions kept |
|---|---|---|---|
| test | 0.667 [0.300, 1.000] | 0.353 [0.133, 0.571] | 0.962 [0.918, 1.000] |
| dev | 1.000 | 0.800 | 1.000 |

Retrieval latency (embed the question + pgvector search), p50 / p95: 29 / 43 ms.

## Answers (test, `claude-opus-5-5`, prompt `answer_v1`)

| metric | value |
|---|---|
| refusal precision (gate + model + citation check) | 0.762 [0.562, 0.941] |
| refusal recall | 0.941 [0.789, 1.000] |
| answerable questions answered | 0.936 [0.880, 0.984] |
| citations quoting their chunk verbatim (as returned, 222 citations) | 1.000 [1.000, 1.000] |
| answers citing an answering article | 0.959 [0.913, 1.000] |
| judged supported by the cited chunks (74 answers) | 0.973 [0.938, 1.000] |
| judged partially supported / unsupported | 0.027 / 0.000 |

Refusals by stage: 9 by the similarity gate, 12 by the model, 0 for lack of a verbatim citation.

Judge: `claude-sonnet-5-5`. Hand check: 22 of 22 judge verdicts agree with a reading of the answer against its cited chunks (20 judged answers taken at a fixed stride, plus every one the judge did not call supported; `eval/judge_check.jsonl`).

Latency p50 / p95 for questions that reach the model: answer generation 5,303 / 8,988 ms, retrieval plus generation 5,332 / 9,015 ms. Refusals at the gate take only the retrieval time. Routing is separate (see the API section of the README).

## Wrong refusal decisions (test)

| id | script | question | expected | got |
|---|---|---|---|---|
| q049 | roman | rahadani le kyc garna milcha? | answer | refused (low_similarity) |
| q052 | roman | ek choti ma 60 hajar pathauna milcha? | answer | refused (model_declined) |
| q079 | roman | atm le paisa diyena, 3 din bhayo firta aayena, complain kasari garne? | answer | refused (low_similarity) |
| q086 | roman | afnai arko account ma pathauda limit lagcha? | answer | refused (model_declined) |
| q104 | deva | डेलिभर्ड देखाउँछ तर मैले पार्सल पाइनँ। | answer | refused (low_similarity) |
| q121 | roman | kathmandu ma aaja pani parcha? | refuse | answered |
