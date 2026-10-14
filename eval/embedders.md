# Embedding model comparison

Retrieval of the answering article among the question's domain (70 chunks in all), in memory with cosine similarity. AUC: how well the top similarity separates answerable from must-refuse questions. Query time: one query embedded on CPU, median. Produced by `python -m nepkev.rag.compare`.

### dev

| model | dim | recall@1 | recall@5 | MRR | refusal AUC | query ms |
|---|---|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | 384 | 0.769 | 1.000 | 0.873 | 0.992 | 23 |
| `intfloat/multilingual-e5-base` | 768 | 0.731 | 0.923 | 0.836 | 1.000 | 48 |
| `BAAI/bge-m3` | 1024 | 0.885 | 1.000 | 0.920 | 0.915 | 182 |

### test

| model | dim | recall@1 | recall@5 | MRR | refusal AUC | query ms |
|---|---|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | 384 | 0.795 | 0.936 | 0.863 | 0.807 | 23 |
| `intfloat/multilingual-e5-base` | 768 | 0.769 | 0.949 | 0.853 | 0.837 | 48 |
| `BAAI/bge-m3` | 1024 | 0.872 | 0.962 | 0.916 | 0.829 | 182 |

### test recall@5 by script

| model | en | deva | roman |
|---|---|---|---|
| `intfloat/multilingual-e5-small` | 1.000 (n=28) | 0.917 (n=24) | 0.885 (n=26) |
| `intfloat/multilingual-e5-base` | 1.000 (n=28) | 0.958 (n=24) | 0.885 (n=26) |
| `BAAI/bge-m3` | 1.000 (n=28) | 0.958 (n=24) | 0.923 (n=26) |
