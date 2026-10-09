You check labels in a dataset of Nepali customer-support messages (Devanagari, or Romanized with English words mixed in). Many are written the way real users write app reviews: typos, chat spellings, ranting, insults and swearing. That is expected and is not a reason to reject a message. You see only the message and the service it was sent to. Decide from the text alone, never from what the customer probably meant.

For each message return:

- `issue`: the id of the single category that fits the customer's problem, or null when the message is too vague to tell what the problem is, or when it raises several problems without saying which one to handle first. If the customer states which problem to handle first, choose that one. Use `other` only for a clear, specific request that fits none of the listed categories; `other` never means "unclear".
- `clarification_needed`: true exactly when `issue` is null.
- `issues_mentioned`: the ids of every category the message describes (empty for a vague message).
- `language_ok`: false if the message is not Nepali (for example Hindi or plain English), or is so garbled a Nepali reader could not follow it. Messy spelling, slang, swearing and mixed-in English are fine.

The categories for each service (id: description), with what each excludes:

{taxonomy}
