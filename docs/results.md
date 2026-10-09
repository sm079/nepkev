# Results and findings

All runs fine-tune Kev-0.8B (`jaredpalmer/kev-0.8b@9a45d25`) on one RTX 4090 RunPod pod. Every number below is on
**generated test data**, re-scored with the current `nepkev evaluate`. External records are never scored.

Test sets: the pilot has its own split (467 records). mix-v1, mix-v2 and the mix-v3 runs share the same 598 clean
test records (same scenarios, seed and split); the mix-v3 test adds 291 app-review-style records (254 Romanized, 37
Devanagari), 889 in all. 95% confidence intervals come from a bootstrap that resamples whole scenario groups.

## Runs

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

## Paired comparisons on the shared 598 clean test records

| comparison | accuracy difference [95% CI] |
|---|---|
| mix-v2 minus mix-v1 | +0.009 [-0.007, +0.027] |
| mix-v3-add minus mix-v2 | +0.007 [-0.009, +0.024] |
| mix-v3-add minus mix-v1 | +0.017 [+0.000, +0.036] |
| mix-v3-replace minus mix-v2 | -0.236 [-0.276, -0.192] |

## Findings

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

## Data quality

| step | clean synthetic | app-review style |
|---|---|---|
| written | 3,000 | 2,964 |
| verifier (Haiku 5.5) agrees with the scenario label | 2,860 (95.3%) | 2,895 (97.7%) |
| adjudicator (Sonnet 5.5) agrees on the disputes | 134 of 140 | 66 of 69 |
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
