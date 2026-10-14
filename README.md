# nepkev: routing Nepali support messages with a 0.8B model

`nepkev` fine-tunes [Kev](https://github.com/jaredpalmer/kev)-0.8B, a small decision model, to route Nepali
customer-support messages for three demo services: an online shop, a digital wallet and a bank. Messages come in
Devanagari or Romanized Nepali, often mixed with English. For each one the model answers two questions: which of the
service's categories the problem belongs to (or `other`), and whether an agent should ask a follow-up question first.

| message (service) | answer |
|---|---|
| `OTP aaudaina login garna sakdina` (wallet) | `login_otp`, confidence 0.9999 |
| `मेरो डेबिट कार्ड हरायो, ब्लक गरिदिनुस्` (bank) | `card_issue`, 0.996 |
| `mero paisa katyo tara recharge bhayena` (wallet) | unsure (0.38), so it asks which problem it is |

The training data is generated: scenarios are sampled with their labels fixed first, a large language model writes
each one as a message, and a second model checks every label from the text alone.

The router runs behind an HTTP API (FastAPI, Postgres, Docker). The same API answers customers' questions from a small
knowledge base, citing only text it retrieved (RAG).

**Checkpoint** [sm079/nepkev](https://huggingface.co/sm079/nepkev) · **Browser demo** [`demo/`](demo/README.md)
(runs on the client, WebGPU or CPU) · **API and setup** [docs/usage.md](docs/usage.md) ·
**RAG** [docs/rag.md](docs/rag.md)

## Results

Issue accuracy of the best run (mix-v3-add), with 95% bootstrap intervals. Neither test set was trained on.

| predictor | generated test (889 messages) | real app reviews (215) |
|---|---|---|
| keyword rules | 0.619 | 0.335 |
| Kev-0.8B, untouched | 0.425 | 0.237 |
| **Kev-0.8B, fine-tuned** | **0.966** [0.953, 0.980] | **0.600** [0.535, 0.660] |

The real reviews are Nepali reviews of wallet, bank and shopping apps from a public app store. Their labels come from
a large language model and were not checked by a person.

## Findings

1. **Fine-tuning is what makes Nepali work.** Untouched, Kev sends most Nepali messages to `other` (5x its true
   share) and asks to clarify 99.8% of the time. Two epochs on 7,400 messages (all generated except 379 real
   reviews) take it to 0.966, and it keeps its accuracy on Kev's own English transfer suite (0.646 vs 0.648).
2. **The drop on real reviews is mostly a data problem.** The model learns the task as the data presents it, but the
   generated messages are easier than real ones: the untouched model scores 0.557 on generated app-review-style
   messages and 0.237 on real reviews. Generated messages describe one incident and state the fact that decides the
   category, because the prompt requires it. Real reviews complain about something that keeps happening, bury the
   problem in a rant, quote the app's error text, name real companies and often leave the deciding fact out.
3. **Copying the voice was not enough.** Rewriting the same scenarios in app-review style (typos, swearing, mostly
   Romanized) gained only +0.051 [+0.005, +0.094] on real reviews. The generator saw the same 30 example reviews in
   every request and slid back to a politer voice of its own: 22% of its messages greet and 20% say please, against
   0% and 6% of real reviews.
4. **The errors are `other` and money mix-ups.** On real reviews, 44% of the errors send a real problem to `other`
   (predicted 2.4x its true share) and a third swap one money category for another. Reviews rarely say whether the
   money left the account, which is what separates those categories.
5. **Confidence is still useful.** On real reviews the model is wrong on 6 of the 63 it answers with 90%+ confidence,
   so a threshold can route the clear cases and hand the rest to a person.
6. **Training choices that held up:** a Romanized twin of every message (Romanized accuracy 0.874 → 0.952 across
   runs), every category balanced within its domain, and Kev's gold-removal augmentation off, because `other` already
   plays "none of the above" (`other` precision 0.72-0.82 → 0.96).

## Cited answers (RAG)

`POST /answer` retrieves from 58 synthetic help articles (English and Nepali) with multilingual-e5-small and pgvector,
limited to the service the message came from. A similarity gate or the answer model can refuse, and the service drops
any citation that does not quote a retrieved chunk word for word. On 95 held-out hand-written questions in English,
Devanagari and Romanized Nepali:

| measure | test |
|---|---|
| retrieval recall@5 | 0.936 [0.867, 0.987] |
| out-of-scope questions refused | 0.941 [0.789, 1.000] |
| answerable questions answered | 0.936 [0.880, 0.984] |
| answers supported by their citations (LLM judge, spot-checked by hand) | 0.973 [0.938, 1.000] |

## Limitations

- **Real labels are model-made.** Nobody checked them, and in the money categories the labeller has to guess too.
- **The real test is small.** 215 scored reviews give intervals of about ±6 points; bank (29) and shop (22) are rough.
- **Clarification does not transfer.** F1 is 0.874 on generated test and 0.267 on real reviews.
- **Calibration is fitted on generated data.** ECE rises from 0.021 to 0.094 on real reviews, and of the real reviews
  that pass the confidence threshold chosen on generated data, 26.5% are wrong (3.3% on generated test).
- **Demo taxonomy.** The categories are not any real institution's, and the router only routes: no transactions, no
  authentication.
- **Rule-based romanization.** Some words come out in spellings typists would not use.
- **Small, in-house RAG test.** The 126 questions were written by the author of the articles, so they share its
  wording. Romanized questions retrieve worst (recall@5 0.885, against 1.000 for English).

## Next

The fix is in the data:

- Generate each message from one real review of the same category, so it inherits how people tell the problem
  (recurring, buried, half-said), not only how they spell.
- Add a real dev set to pick the checkpoint and the routing threshold.
- Get human-checked labels for the real test, starting with the money categories.

## More

- [docs/results.md](docs/results.md): every run, both test sets, the real-vs-generated comparison, data quality,
  design decisions
- [docs/usage.md](docs/usage.md): setup, the HTTP API, the data and training pipeline, repository layout
- [docs/rag.md](docs/rag.md): the RAG layer, embedding-model comparison, full results and failure cases
- [docs/taxonomy-notes.md](docs/taxonomy-notes.md): the categories and the pairs that are easy to confuse
- [demo/README.md](demo/README.md): the browser demo
