"""Turning a routing into the service's decision: the message's script, whether to ask first, and what to ask.

Questions are written in Devanagari and transliterated with nepkev.romanize for Romanized messages, so the customer is
asked in the script they wrote in. Category names come from the taxonomy's Nepali descriptions (`ne`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..romanize import transliterate
from ..router import Routing
from ..taxonomy import Taxonomy
from ..validate import detect_script

SCRIPTS = {"deva": "devanagari", "roman": "romanized", "mixed": "mixed"}

# What a customer would name when saying which case they mean, per domain.
SUBJECT = {"shop": "अर्डर", "wallet": "कारोबार", "bank": "खाता वा कारोबार"}

UNCLEAR = "तपाईंको समस्या अलि खुलाएर भन्नुहोस् न: के भयो, कुन {subject}मा, र कहिले? धेरै समस्या छन् भने कुनचाहिँ पहिले हेरौँ?"
CHOOSE = "तपाईंको समस्या यीमध्ये कुन हो: {first}, कि {second}? अलि खुलाएर भन्नुहोस् न।"


def script_of(text: str) -> str:
    return SCRIPTS[detect_script(text)]


@dataclass(frozen=True)
class Decision:
    needs_clarification: bool
    reason: str | None
    question: str | None


def decide(routing: Routing, text: str, tax: Taxonomy, clarify_threshold: float, min_confidence: float) -> Decision:
    if routing.clarification >= clarify_threshold:
        reason, question = "unclear_message", UNCLEAR.format(subject=SUBJECT[routing.domain])
    elif routing.confidence < min_confidence:
        dom = tax.domain(routing.domain)
        first, second = [dom.categories[c].ne for c, _ in routing.ranked() if c != tax.fallback][:2]
        reason, question = "low_confidence", CHOOSE.format(first=first, second=second)
    else:
        return Decision(False, None, None)
    return Decision(True, reason, question if script_of(text) != "romanized" else _roman(question))


@lru_cache(maxsize=256)
def _roman(question: str) -> str:
    return transliterate(question)
