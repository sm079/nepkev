# Results on real app reviews

The numbers in [results.md](results.md) are all on generated test data. This page covers a second test set:
Nepali reviews that people wrote about wallet, banking and shopping apps in a public app store. The best run,
mix-v3-add, gets **0.600** issue accuracy here, against 0.966 on generated test data.

For scale: Kev's own README reports the untouched Kev-0.8B at **0.648 / 0.697** (development / test) on English
datasets and rule types it never trained on ("new sources"), and 0.827 / 0.838 on held-out examples of the datasets
it trained on. Our run reproduces the 0.648 on the development set (transfer-v4). So the fine-tuned model on real
Nepali reviews sits a little below what the base model does on unfamiliar English tasks. The tasks and labels differ,
so this is a rough comparison, not a controlled one.

## Test set

- 273 reviews (wallet 207, bank 37, shop 29), sampled from the same pool as the 379 `external` training records.
  None was trained on, and near-duplicates of test reviews were removed from training.
- Sampled per domain, at most half of any category and at most the domain's median count of non-`other` categories,
  so the mix is flatter than the raw reviews (where about 62% of problems are `other`).
- Labels come from a large language model and were not checked by a person. Accuracy here is agreement with those
  labels on real text, not with human judgement.
- 215 reviews name one concrete problem and are scored for issue accuracy (wallet 164, bank 29, shop 22). 22 are
  vague and 15 name several problems; these get soft targets and count only toward clarification. 21 are not support
  requests at all (praise, feature requests, rants) and are reported separately.
- Bank and shop have fewer than 30 scored reviews each, so their numbers are rough.

## Results

Issue accuracy on the 215 scored reviews (95% CIs from a bootstrap):

| run | fine-tuned [95% CI] | wallet | bank | shop | `other` predicted / true | `other` precision |
|---|---|---|---|---|---|---|
| mix-v2 | 0.549 [0.484, 0.610] | 0.543 | 0.586 | 0.545 | 2.59x | 0.30 |
| **mix-v3-add** | **0.600** [0.535, 0.660] | 0.598 | 0.621 | 0.591 | 2.41x | 0.28 |
| mix-v3-replace | 0.549 [0.486, 0.609] | 0.537 | 0.517 | 0.682 | 2.68x | 0.24 |
| keyword rules | 0.335 [0.276, 0.399] | 0.335 | 0.310 | 0.364 | 0.00x | n/a |
| Kev-0.8B untouched | 0.237 [0.182, 0.297] | 0.244 | 0.138 | 0.318 | 8.23x | 0.12 |

mix-v3-add minus mix-v2, paired on the same reviews: +0.051 [+0.005, +0.094]. mix-v3-add minus the untouched model:
+0.363 [+0.289, +0.430]. mix-v1 was scored on an earlier, larger real test (600 reviews, different sampling): 0.543
[0.500, 0.587] against 0.315 untouched and 0.289 rules, so it is not in the table.

More for mix-v3-add on real text:

| measure | real reviews | generated test |
|---|---|---|
| issue accuracy | 0.600 | 0.966 |
| macro-F1 | 0.607 | 0.965 |
| issue NLL | 1.389 | 0.122 |
| ECE | 0.094 | 0.021 |
| routed at the calib confidence threshold | 63.3%, with 26.5% of those wrong | 99.6%, 3.3% wrong |
| clarification F1 | 0.267 [0.107, 0.406] | 0.874 |
| clarification rate (predicted / true) | 9.1% / 14.7% | 8.8% / 10.0% |
| wrong at 90%+ confidence | 6 of 63 | |

The bias checks fail on real text for every predictor. The fine-tuned model over-predicts `other` and has recall
under 0.5 for `payment_failed_debited` (0.19), `payment_failed_no_debit` (0.33) and `unexpected_transaction` (0.38) in
the wallet domain.

## Where the errors are (mix-v3-add)

86 of the 215 scored reviews are wrong:

| error | count | share of errors |
|---|---|---|
| a real category sent to `other` | 38 | 44% |
| one money problem mistaken for another (debited vs not debited, refund, transfer not credited, unexpected transaction) | 28 | 33% |
| `other` sent to a category | 7 | 8% |
| anything else | 13 | 15% |

Most frequent: `payment_failed_no_debit` -> `other` (6), `payment_failed_debited` -> `other` (5), and 4 each of
`transfer_not_credited`, `login_otp`, `withdrawal_issue` and `bank_linking` -> `other`.

On the 21 reviews that are not support requests, the fine-tuned model answers `other` for 14 and asks to clarify for
3; the keyword router asks to clarify for 19 and routes most of the rest to `login_otp`; the untouched model answers
`other` for all 21 and asks to clarify every time.

## Findings

1. **Fine-tuning helps on real text too, but much less.** 0.600 against 0.237 untouched and 0.335 for the rules,
   while generated test is at 0.966. The gap between the two test sets (about 0.37) is the main open problem.
2. **`other` is the main failure.** It is predicted 2.4x as often as it is true, and only 28% of `other` predictions
   are right. Real reviews often describe a problem loosely or bury it in a rant, and the model falls back to `other`
   when nothing matches the clear wording it was trained on. On generated
   test the same model predicts `other` at 1.03x with 0.96 precision.
3. **Money categories blur.** A third of the errors swap one money category for another. Reviews rarely say whether
   the money left the account, which is exactly what separates `payment_failed_debited` from
   `payment_failed_no_debit`; the reference labels have to guess as well, so part of this error is in the labels.
4. **App-review-style data helped a little.** Adding it (mix-v3-add) gained +0.051 over mix-v2 on the same reviews,
   with an interval that just excludes 0. Using it instead of the clean data
   (mix-v3-replace) scored the same as mix-v2 (0.549). Re-telling clean scenarios in a messy voice moves the model
   only part of the way: style was not the whole gap.
5. **The untouched model is much worse on real text than on app-review-style text** (0.237 vs 0.557). Generated
   reviews state their problem clearly; real ones often do not. That difference in content, not spelling or
   profanity, is what the generated data misses.
6. **Clarification does not transfer.** F1 falls from 0.874 to 0.267, and the model asks to clarify less often than
   the labels say it should (9.1% vs 14.7%). The vague and multi-problem messages it learned from are generated
   ones, and real reviews are vague in other ways.
7. **Confidence is still useful.** At 90%+ confidence the fine-tuned model is wrong on 6 of 63 reviews, and ECE is
   0.094. A confidence threshold can route the clear cases and hand the rest to a person, though the calib threshold
   chosen on generated data lets through far more errors here (26.5%) than on generated test (3.3%).

## What would close the gap

- Generated messages whose *content* is like real reviews: half-described problems, several complaints in one
  rant, background without the key fact, not only a messier voice for clear scenarios.
- A real dev set used for model selection and for the routing threshold, so that real-text behaviour, not
  generated-text NLL, picks the checkpoint.
- Human-checked labels for at least the real test, especially in the money categories.
- More real bank and shop reviews, so those domains can be measured.
