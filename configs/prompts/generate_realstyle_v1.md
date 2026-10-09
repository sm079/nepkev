You write Nepali customer complaints the way real people write them in app-store reviews and support chats, for a synthetic training dataset. The messages train a model that routes support tickets for three fictional services: an online shop, a digital wallet and a bank. Every message is fictional. Never use real names of people, companies, banks or wallets ("yo app", "tapaiko bank", "wallet" are fine), and never include credentials, real account numbers or phone numbers.

The model already sees plenty of clean, careful Nepali. What it lacks is how people really type: fast, on a phone, often annoyed. Below are real reviews from Nepali users. Write like them.

Real reviews (style only; their problems and apps are not your scenarios):

{examples}

Each scenario names its script:

- `roman`: Romanized Nepali, the way most users type: chat spellings (xa / cha / chha, vako / bhako, k, ni, ta, ho ra, garnu paryo, pathaye), English words mixed in freely (app, OTP, refund, balance, server, update, worst, please, sir), and inconsistent spelling across messages. No Devanagari letters.
- `deva`: Devanagari, typed casually: spoken forms, English words in Devanagari or left in Latin letters, the same carelessness.

How to write:

- Match the examples' register: typos, missing or doubled punctuation, run-on sentences, random capitals or ALL CAPS, emoji, star-rating talk ("1 star", "worst app ever"), complaints about the company, threats to uninstall or switch, swearing or insults where the tone calls for it, as the examples do.
- Length, tone and formality come with each scenario. Short is one line, medium is 2 to 3 sentences, long is a rant of 4 to 6 sentences. Show the tone through wording, never by naming the emotion. `formal` means a polite, more careful user, still writing on a phone.
- Vary openings and wording. Do not copy sentences from the examples.

Hard rules (the labels depend on these):

1. The message must carry enough evidence for the described situation. A support agent who reads only the message must be able to tell which kind of problem it is, or, when the situation says the customer is vague, must not be able to tell. Messy writing and ranting must not hide the problem.
2. Include every visible fact (amounts with the same value; any number format). Never reveal a withheld fact, even indirectly.
3. Describe what happened in the customer's own words. Never use support-desk or category vocabulary (for example "payment failed debited", "duplicate debit case", "transfer not credited issue").
4. When the situation asks for a negation or a correction, the message must contain it explicitly.
5. Do not add a second problem the situation does not describe. General complaints about the app or service ("bekar app", "customer care le phone uthaudaina") are fine as long as they do not describe another problem from the list below.

Also return `evidence`: one short English sentence saying which words in the message decide the situation (or why the problem cannot be identified).

The kinds of problems each service handles, so you can make each message clearly one kind and not a neighbouring one:

{taxonomy_plain}
