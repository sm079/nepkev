# Taxonomy notes (demo-1)

The three taxonomies are demo routing schemes for a fictional online shop, digital wallet and bank. They do not describe any real institution. The machine-readable version is `configs/taxonomy/demo-1.yaml`. These notes explain the boundaries that are easy to get wrong.

## Labels per message

Each message gets two independent labels:

1. **issue**: one category from the selected domain, or no hard label when the text cannot identify one.
2. **clarification_needed**: whether the agent must ask a follow-up question before routing.

| Message | issue | clarification_needed |
|---|---|---|
| One identifiable problem | that category | false |
| Several problems, customer says which comes first | the stated priority | false |
| Several problems, no stated priority | soft target split evenly over the named problems | true |
| Too vague to identify a problem ("problem vayo help garnus") | soft target spread evenly over all categories | true |
| Clear and specific, but outside the taxonomy | `other` | false |

`other` means "outside the list", never "unclear". A vague message is a clarification case, not `other`.

Unresolvable issue questions are trained on soft targets instead of being dropped. This teaches the model to spread probability when the text cannot decide, which is what the confidence threshold for deferring relies on. They are never counted in issue accuracy.

## Confusable pairs

| Pair | Deciding question |
|---|---|
| `digital_banking_login_otp` vs `account_access` (bank) | Is the app or OTP failing, or is the account itself frozen, dormant or blocked so that even a branch cannot operate it? |
| `payment_failed_debited` vs `payment_failed_no_debit` | Was money deducted? |
| `payment_failed_debited` vs `duplicate_debit` | One failed payment that still took money, or one purchase charged two or more times? |
| `transfer_not_credited` vs `wrong_recipient` | Did the money reach nobody, or did it reach the wrong person or account because of the customer's mistake? |
| `transfer_not_credited` vs `withdrawal_issue` (wallet) | Sent to another person, or moved to the customer's own bank account or an agent? |
| `withdrawal_atm_issue` vs `card_issue` (bank) | Did the ATM fail to give cash, or is the card itself blocked, lost or not working? |
| `withdrawal_atm_issue` vs `payment_failed_debited` (bank) | ATM cash withdrawal, or a payment to a merchant or bill? |
| `wrong_damaged_item` vs `return_exchange` (shop) | Is the item wrong or damaged, or correct but unwanted or the wrong fit? |
| `order_status` vs `delivery_delay_missing` (shop) | A status question with nothing late yet, or late past the promise, lost, or marked delivered but not received? |
| `cancellation` vs `return_exchange` (shop) | Before or after delivery? |
| `duplicate_debit` vs `unexpected_transaction` | A known payment repeated, or a transaction the customer does not recognise? |
| `product_inquiry` / `account_service_inquiry` vs complaints | A question before buying or opening, or a problem with something already in use? |

Each pair is generated as counterfactual siblings that share a scenario group, so both always land in the same split.

## What the model sees

The question text and category descriptions are English (`description` fields). Kev was trained on English instructions, and the project's goal is Nepali *input*, not Nepali instructions. The `ne` fields are for people: the UI and reviewers.
