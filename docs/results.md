# Results

All runs fine-tune Kev-0.8B (`jaredpalmer/kev-0.8b@9a45d25`) on one rented RTX 4090. There are two test sets:
generated messages (never trained on) and real app reviews. External records are never scored.

## Generated test data

Test sets: the pilot has its own split (467 records). mix-v1, mix-v2 and the mix-v3 runs share the same 598 clean
test records (same scenarios, seed and split); the mix-v3 test adds 291 app-review-style records (254 Romanized, 37
Devanagari), 889 in all. 95% confidence intervals come from a bootstrap that resamples whole scenario groups.

### Runs

| run | training data (records) | gold removal | epochs (selected) | test acc [95% CI] | untouched | rules | clarify F1 |
|---|---|---|---|---|---|---|---|
| pilot | 2,378 Devanagari + 1,164 Romanized twins (3,542) | 8% | 2 (e2) | 0.921 [0.891, 0.950] | 0.350 | 0.657 | 0.810 |
| mix-v1 | 2,378 + 2,369 twins + 1,690 external, skewed to a few categories (6,437) | 8% | 4 (e2) | 0.952 [0.930, 0.970] | 0.361 | 0.658 | 0.857 |
| mix-v2 | 2,378 + 2,369 twins + 379 external, balanced per category (5,126) | off | 2 (e2) | 0.961 [0.943, 0.978] | 0.361 | 0.658 | 0.845 |
| **mix-v3-add** | mix-v2 + 2,274 app-review style (7,400) | off | 2 (e2) | **0.966** [0.953, 0.980] | 0.425 | 0.619 | 0.874 |
| mix-v3-replace | 2,274 app-review style + 379 external (2,653) | off | 2 (e2) | 0.792 [0.763, 0.826] | 0.425 | 0.619 | 0.667 |

"untouched" is the released Kev-0.8B with no fine-tuning; "rules" is the keyword router (`configs/rules/demo-1.yaml`).
The mix-v3 tests include the app-review-style records, which are harder for the rules and easier for the untouched
model, so their baseline columns differ from mix-v1/v2.

Accuracy by part of the test set (fine-tuned model):

| run | clean synthetic | translated (Romanized) | app-review style | bank | shop | wallet |
|---|---|---|---|---|---|---|
| pilot | 0.948 | 0.874 | | 0.899 | 0.944 | 0.921 |
| mix-v1 | 0.963 | 0.941 | | 0.956 | 0.947 | 0.953 |
| mix-v2 | 0.978 | 0.944 | | 0.972 | 0.963 | 0.947 |
| mix-v3-add | 0.985 | 0.952 | 0.962 | 0.981 | 0.957 | 0.961 |
| mix-v3-replace | 0.673 | 0.777 | 0.931 | 0.752 | 0.763 | 0.867 |

Calibration and retention:

| run | dev NLL by epoch | temperature | test ECE | coverage at 5% error | transfer-v4 acc (untouched 0.648) | transfer-v4 ECE (untouched 0.049) | confident errors on transfer-v4 (untouched 0.2%) |
|---|---|---|---|---|---|---|---|
| pilot | 0.452, 0.358 | 1.48 | 0.027 | 92.4% | 0.646 | 0.121 | 3.5% |
| mix-v1 | 0.447, 0.381, 0.391, 0.430 | 1.29 | 0.044 | 100% | 0.643 | 0.092 | 2.6% |
| mix-v2 | 0.411, 0.302 | 1.35 | 0.028 | 100% | 0.645 | 0.111 | 3.7% |
| mix-v3-add | 0.323, 0.262 | 1.55 | 0.021 | 100% | 0.646 | 0.091 | 2.1% |
| mix-v3-replace | 0.549, 0.447 | 1.29 | 0.052 | 56.8% | 0.646 | 0.123 | 4.0% |

Bias checks on test (limits fixed before any result: per-category predicted share at most 1.5x its true share and
recall at least 0.5 for categories with 10+ items; `other` predicted at most 1.5x its share with precision at least
0.7; predicted clarification rate within 5 points of the true rate):

| run | `other` predicted / true | `other` precision | clarification rate (pred / true) | verdict |
|---|---|---|---|---|
| pilot | 1.36x | 0.72 | 7.9% / 10.1% | pass |
| mix-v1 | 1.17x | 0.82 | 8.7% / 10.0% | pass |
| mix-v2 | 0.96x | 0.96 | 9.4% / 10.0% | pass |
| mix-v3-add | 1.03x | 0.96 | 8.8% / 10.0% | pass |
| mix-v3-replace | 1.39x | 0.64 | 5.8% / 10.0% | fail: 6 categories over-predicted, 2 under 0.5 recall, `other` precision |
| untouched Kev (any run) | 5.0-5.5x | 0.16-0.18 | 99.8% / 10% | fail |

### Paired comparisons on the shared 598 clean test records

| comparison | accuracy difference [95% CI] |
|---|---|
| mix-v2 minus mix-v1 | +0.009 [-0.007, +0.027] |
| mix-v3-add minus mix-v2 | +0.007 [-0.009, +0.024] |
| mix-v3-add minus mix-v1 | +0.017 [+0.000, +0.036] |
| mix-v3-replace minus mix-v2 | -0.236 [-0.276, -0.192] |

### Findings

1. **Fine-tuning is what makes Nepali work.** The untouched model gets 0.35-0.43 and sends most Nepali messages to
   `other` (predicted 5x its true share) while asking to clarify almost always (99.8% vs 10%). After 2 epochs it gets
   0.92-0.97. The keyword router stays at 0.62-0.66.
2. **Romanized was the weak script.** In the pilot, where only half the records had a Romanized twin, Romanized test
   accuracy was 0.874 against 0.948 for Devanagari. Every later run gives every record a twin (`roman_share: 1.0`), and
   Romanized accuracy rose to 0.94-0.95. Other things changed at the same time, so the twin share is likely but not
   proven to be the cause.
3. **Two epochs.** mix-v1 trained four: dev NLL was lowest after epoch 2 (0.381) and rose after (0.391, 0.430). Every
   later run trains two, as Kev's own fine-tuning example does, and keeps the epoch with the lowest dev NLL (always 2).
4. **Gold removal is off.** Kev's trainer can remove the correct category from the candidates so that "none of the
   above" becomes right (`p_none`). Kev needs that because its datasets have no out-of-scope class. Here `other` is
   that option and is always offered, and out-of-scope scenarios teach it. With gold removal at 8% (pilot, mix-v1),
   `other` was over-predicted (1.17-1.36x, precision 0.72-0.82); with it off (mix-v2 onwards), 0.96-1.03x with
   precision 0.96. mix-v1 also had skewed external data, so this comparison is not clean.
5. **Balance every category.** Scenarios are sampled least-used-category first, and the training mix is checked so
   that each category has about the same share of its domain's labels (6.7-10.3% in mix-v3-add, `other` 7.0-8.0%).
   The skewed external data in mix-v1 was replaced by a balanced subset in mix-v2.
6. **App-review-style data adds without hurting the clean data.** Adding it (mix-v3-add) left the clean test where it was
   (+0.007, interval includes 0) and scored 0.962 on app-review-style test. Using it instead of the clean data
   (mix-v3-replace) lost 0.236 on the clean test, failed the bias checks, and still scored lower on app-review style
   itself (0.931 vs 0.962). The clean data helps on the messy data too.

   How this data was made limits what it can show. The style came from 30 example reviews, a small fraction of those
   available, shown unchanged in every request. The messages re-tell the clean data's scenarios, so they keep that
   data's content (one clearly stated problem, the same sampled facts, the same family mix) and change only the
   voice. The model sees each scenario three times (Devanagari, Romanized twin, app-review style): more surface
   variety, no new situations. A 0.962 on this test therefore says the model reads messy spelling and swearing, not
   that it handles what messy messages are about.
7. **The untouched model finds app-review style easier than clean Nepali** (0.557 vs 0.301 clean and 0.420
   Romanized). A likely reason is that these messages carry more English (app, OTP, refund, balance), and the base model
   is English-trained. The keyword router finds them harder (0.538 vs 0.66), since its Nepali cues are spelled many ways.
8. **Hardest families:** a stated priority between two problems (0.71-0.78 in every run) and irrelevant background
   before the problem (0.81-0.93). Contrast pairs, negations and corrections are at 0.91-1.00 from mix-v1 to
   mix-v3-add.
9. **Calibration holds in distribution, not outside it.** Temperatures fitted on calib (1.29-1.55) give test ECE of
   0.02-0.05, but calib comes from the same data as training. On Kev's English transfer-v4 suite, accuracy is retained
   (0.643-0.646 vs 0.648) while ECE rises from 0.049 to 0.09-0.12 and errors at 90%+ confidence from 0.2% to 2-4%.

## Real app reviews

Nepali reviews that people wrote about wallet, banking and shopping apps in a public app store. mix-v3-add gets
**0.600** issue accuracy here, against 0.966 on generated test data.

For scale: Kev's own README reports the untouched Kev-0.8B at **0.648 / 0.697** (development / test) on English
datasets and rule types it never trained on ("new sources"), and 0.827 / 0.838 on held-out examples of the datasets
it trained on. Our run reproduces the 0.648 on the development set (transfer-v4). So the fine-tuned model on real
Nepali reviews sits a little below what the base model does on unfamiliar English tasks. The tasks and labels differ,
so this is a rough comparison, not a controlled one.

### Test set

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

### Results

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

### Where the errors are (mix-v3-add)

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

### Findings

1. **Fine-tuning helps on real text too, but much less.** 0.600 against 0.237 untouched and 0.335 for the rules,
   while generated test is at 0.966. The gap between the two test sets (about 0.37) is the main open problem, and
   most of it comes from the data ([Why real reviews are harder](#why-real-reviews-are-harder)).
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

## Why real reviews are harder

The drop from 0.966 to 0.600 is mostly a data problem. The model learns the task as the training data presents it
(0.966 on held-out generated messages, 0.962 on the app-review-style ones). Real reviews differ from that data in what
they say, not only in how they spell it, and their labels are less certain.

**The generated messages are easier.** The same untouched model, trained on neither, scores 0.557 on generated
app-review-style test messages and 0.237 on real reviews. The single-problem messages of both side by side (2,547
accepted app-review-style messages, the 215 scored real reviews):

| | real reviews | generated, app-review style |
|---|---|---|
| words, median (10th-90th percentile) | 23 (9-50) | 26 (15-50) |
| contains a number | 33% | 65% |
| names a real company or app | 33% | 0% (the prompt forbids it) |
| greets ("namaste", "namaskar") | 0% | 22% |
| says "but" ("tara", "तर") | 11% | 51% |
| says "please" / "plz" | 6% | 20% |
| contains "!!" | 2% | 19% |
| contains "worst" | 10% | 1% |
| distinct word pairs in 200-message samples | 0.82 | 0.63 |

- **The content is cleaner.** Each generated message re-tells a sampled incident with its amount and with the fact
  that decides the category, because the prompt says messy writing must not hide the problem. That is the "money was
  taken, but the recharge failed" shape behind the 51% "but". Real reviews more often complain about something that
  keeps happening ("session expire hunxa"), mention the problem in passing inside a rant, quote the app's error text,
  name the company or a competitor, and leave the deciding fact out. These are the reviews that end up in `other`.
- **The style examples did not take.** The generator saw the same 30 real reviews in every request, at the top of a
  long prompt, and slid back to a politer voice of its own: greetings, "please", double exclamation marks, and less
  varied wording than real reviews. Rewriting the scenarios in this voice gained only +0.051 on real reviews.
- **Part of the gap is in the labels.** The real labels come from a large language model reading the same unclear
  text. In the money categories it has to guess whether money left the account, and some short complaints it gave a
  category ("my money vanished, what is this") are arguably too vague for any.

Only Kev-0.8B has been trained, so model size is untested. But a fine-tuned model at 0.966 in distribution, and an
untouched model that loses more than half its accuracy on real text, both point at the data first.

## What would close the gap

- **One real review per generated message.** Pair each scenario with a real review of the same category (never one
  from the real test or dev set) and let the generator take how that person told their problem: recurring, buried in
  a rant, half-said, in the app's own words. The scenario still fixes the label, the "must not hide the problem" rule
  becomes "a careful reader can tell", and real names become invented ones. Most labelled reviews are not used yet,
  so there are references to draw on.
- **Check new data before training on it.** On a few hundred messages, the untouched model's accuracy (0.237 on real
  reviews) and how easily a simple classifier tells them from real reviews show whether the data moved toward real
  text.
- A real dev set used for model selection and for the routing threshold, so that real-text behaviour, not
  generated-text NLL, picks the checkpoint.
- A larger real test. 215 scored reviews give intervals of about ±6 points, too wide to see a gain of a few points.
- Human-checked labels for at least the real test, especially in the money categories.
- More real bank and shop reviews, so those domains can be measured.

## Data quality

| step | clean synthetic | app-review style |
|---|---|---|
| written | 3,000 | 2,964 |
| verifier model agrees with the scenario label | 2,860 (95.3%) | 2,895 (97.7%) |
| adjudicator model agrees on the disputes | 134 of 140 | 66 of 69 |
| rejected: label disputed | 4 | 3 |
| rejected by the deterministic checks | 32 (22 Latin letters in clean Devanagari, 4 missing amount or negation, 6 duplicates) | 130 (122 script glitches, 7 not Nepali, 1 duplicate) |
| accepted | 2,964 | 2,831 |
| Romanized twins accepted | 2,954 | |

The app-review-style script glitches were Romanized words with stray Devanagari vowel signs (`katेko`) and messages
meant to be Devanagari that came out in Latin letters. Both are rejected, not repaired.

## Decisions

| decision | why |
|---|---|
| Instructions and category descriptions in English, only the message in Nepali | Kev was trained on English instructions; the goal is Nepali input |
| Labels decided from scenario facts before text exists | a label can never rest on something the text lacks |
| Vague and multi-issue messages train on soft targets, never scored for accuracy | the model learns to spread probability where the text cannot decide |
| Generator never sees category ids | it copied them into messages when it did |
| Clean Devanagari has no Latin letters; app-review-style Devanagari may keep English words | clean data models one keyboard per message; app-review style follows how such reviews are typed |
| Every record gets a Romanized twin | Romanized was the weak script |
| Gold removal off | `other` is always a candidate and is taught by out-of-scope examples (finding 4) |
| 2 epochs, select by dev NLL, temperature on calib, test read once | finding 3; test never tunes anything |
| Same scenario group, same split; near-duplicate groups merged | no paraphrase of a test item is trained on |
