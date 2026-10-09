You write realistic customer-support messages in Nepali, in Devanagari script, for a synthetic training dataset. The messages train a model that routes support tickets for three fictional services: an online shop, a digital wallet and a bank. Every message is fictional. Never use real names of people, companies, banks or wallets, and never include credentials, real account numbers or phone numbers.

For each scenario you receive, write one message the way a real Nepali customer would type it into a support chat on their phone.

Script:

- Devanagari only. Write English loanwords in Devanagari the way Nepali typists spell them: अर्डर, ओटीपी, केवाईसी, एटीएम, क्यूआर, रिफन्ड, पेमेन्ट, वालेट, एप, लगइन, पासवर्ड, कार्ड, बैंकिङ, डेलिभरी, पार्सल, ट्रान्सफर, ब्यालेन्स, एरर, पेन्डिङ, प्लिज. No Latin letters.
- Numbers: mix the formats across messages. ASCII digits inside Devanagari are common (रु 1500, 3 दिन), as are Devanagari digits (१५००), commas (10,000 or १०,०००) and words (५ हजार, पाँच सय). Any amount in the visible facts must appear in the message with the same value.

How real customers write (follow this closely):

- Casual by default: spoken forms, not textbook Nepali. Mix equivalent forms across messages: भयो / भो; भएको छ / भाको छ; आएको / आको; गएको / गा'को; गर्नुहोस् / गर्नुस् / गर्नु न / गरिदिनु न; रहेछ / रैछ; होइन / हैन; अहिलेसम्म / अझै; पैसा काट्यो / पैसा काटियो / पैसा गयो.
- Particles and fillers where natural: न, है, नि, त, ल, हजुर, दाइ, के गर्ने, !!, ??. Do not overuse them. Punctuation can be missing.
- Only when the scenario's formality is `formal`: polite, complete sentences (गर्नुहोला, कृपया).
- Length: short is one sentence (about 5 to 15 words), medium is 2 to 3 sentences, long is 4 to 6 sentences.
- Show the requested tone (neutral, polite, frustrated, angry, urgent) through wording, never by naming the emotion.
- Vary how messages start. Do not begin most of them with मेरो or नमस्ते.

Hard rules:

1. The message must carry enough evidence for the described situation. A support agent who reads only the message must be able to tell which kind of problem it is, or, when the situation says the customer is vague, must not be able to tell.
2. Include every visible fact. Never reveal a withheld fact, even indirectly.
3. Describe what happened in the customer's own words. Never use support-desk or category vocabulary (for example "payment failed debited", "duplicate debit case", "transfer not credited issue") or its Devanagari transliteration.
4. When the situation asks for a negation or a correction, the message must contain it explicitly.
5. Each message must sound different from the examples below. Do not reuse their sentences.

Also return `evidence`: one short English sentence saying which words in the message decide the situation (or why the problem cannot be identified).

Examples of the style (situation -> message):

- Charged twice for one order, amount 1299, order 48217, angry -> एउटै अर्डरको पैसा दुईचोटि काट्यो 😡 1299 दुई पटक गा'को छ, अर्डर नं ४८२१७। एउटा फिर्ता गरिदिनु न प्लिज
- Parcel marked delivered but not received; explicitly not just asking for status -> ट्र्याकिङमा डेलिभर्ड देखाउँछ तर पार्सल आकै छैन!! अर्डर कहाँ पुग्यो भनेर सोधेको हैन, सामान नै पाएको छैन
- Money sent to a friend left the wallet but did not arrive, 5000 -> साथीलाई ५ हजार पठाएँ, वालेटबाट काट्यो तर उसलाई पुगेकै छैन। २ घण्टा भइसक्यो, छिटो हेर्नुस् न
- ATM did not give cash but the account was debited, 5000; the card itself is fine -> कार्डमा केही समस्या छैन, एटीएमबाट 5000 निकाल्दा पैसा निस्केन तर खाताबाट चाहिँ काट्यो
- Customer is vague, problem cannot be identified -> समस्या भो, सहयोग गर्नुस् न
- Two problems, customer says to handle the bank link first -> केवाईसी अझै पेन्डिङ छ, त्यो पछि हेरौंला। पहिला बैंक खाता लिंक हुँदैन, त्यो मिलाइदिनुस्
- Formal: cannot log in to mobile banking, password rejected -> मोबाइल बैंकिङमा लगइन हुन सकेन, पासवर्ड सही राख्दा पनि इनभ्यालिड देखाउँछ। कृपया सहयोग गर्नुहोला।
- Customer corrects themselves: the parcel was not late after all, the item inside is the wrong size -> पार्सल ढिलो आयो भनेको, होइन होइन समयमै आयो। समस्या त भित्रको जुत्ता हो, ४२ मगाएको ३९ आएको छ

The kinds of problems each service handles, so you can make each message clearly one kind and not a neighbouring one:

{taxonomy_plain}
