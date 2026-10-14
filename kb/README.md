# Knowledge base (synthetic)

Short help articles for the three **fictional** services the router knows: an online shop, a digital wallet and a bank.
Every policy, limit, fee and time in here was written for this demo. None of it describes a real company or a real
bank's rules, and none of it comes from customer data.

Each topic exists twice, in English (`-en`) and Nepali (`-ne`), with the same facts. Front matter:

| field | meaning |
|---|---|
| `id` | stable document id; chunk ids are derived from it |
| `domain` | `shop`, `wallet` or `bank`; retrieval is filtered by the router's channel |
| `lang` | `en` or `ne` |
| `title` | article title (embedded with the text) |
| `updated_at` | date the article was last changed |

`uv run python -m nepkev.rag.ingest` chunks and embeds these files into `kb_chunks`; it re-embeds only what changed.
