"""Structured scenarios: the facts and gold labels a message is rendered from.

Labels are decided here, from scenario facts and taxonomy rules, before any text exists. The
renderer must put every visible fact into the message and must not reveal withheld ones, so a
label is never justified by information the text lacks. All identifiers and amounts are
fictional and sampled from a seeded RNG.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from pydantic import BaseModel

from .config import GenConfig
from .taxonomy import Taxonomy


class Scenario(BaseModel):
    scenario_id: str
    scenario_group_id: str
    domain: str
    taxonomy_version: str
    family: str                         # difficulty family, e.g. "contrast:transfer_not_credited/wrong_recipient"
    issue: str | None
    issue_target: dict[str, float] | None = None
    clarification_needed: bool
    issues_all: list[str]
    stated_priority: str | None = None
    facts: dict[str, Any]
    visible_facts: list[str]
    withheld_facts: list[str] = []
    style: dict[str, str]
    instructions: str                   # what the renderer must convey (English, for the generator only)


# ---------------------------------------------------------------- facts

PRODUCTS = ["bluetooth speaker", "running shoes", "rice cooker", "phone cover", "kurta", "school bag", "earbuds",
            "wall clock", "water bottle", "smart watch", "jacket", "light bulb pack", "gas stove", "notebook set"]
BILLERS = ["electricity bill", "water bill", "internet bill", "mobile top-up", "cable television subscription", "school fee", "insurance premium"]
METHODS = ["wallet", "debit card", "mobile banking", "QR"]
BANK_PRODUCTS = ["fixed deposit", "savings account", "current account", "remittance service", "debit card",
                 "mobile banking", "locker service", "education loan"]
PLACES = ["New Road", "Lakeside Pokhara", "Butwal", "Chabahil", "Biratnagar", "Koteshwor", "Narayangadh"]


def _amount(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.45:
        return rng.choice([100, 200, 500]) * rng.randint(1, 20)          # round amounts: 500, 1500, 10000
    if r < 0.75:
        return rng.randrange(150, 9999)                                   # odd amounts: 1299, 840
    return rng.randint(10, 60) * 1000                                     # large transfers


FACT_SAMPLERS = {
    "order_id": lambda r: str(r.randint(10000, 99999)),
    "amount_npr": _amount,
    "days_since_order": lambda r: r.randint(1, 6),
    "days_late": lambda r: r.randint(2, 15),
    "days_waiting": lambda r: r.randint(2, 21),
    "hours_since": lambda r: r.choice([1, 2, 3, 5, 12, 24, 48]),
    "times_charged": lambda r: r.choice([2, 2, 2, 3]),
    "ordered_item": lambda r: r.choice(PRODUCTS),
    "received_item": lambda r: "a different size/colour or damaged version of it",
    "item": lambda r: r.choice(PRODUCTS),
    "product": lambda r: r.choice(PRODUCTS),
    "bank_product": lambda r: r.choice(BANK_PRODUCTS),
    "payment_method": lambda r: r.choice(METHODS),
    "biller": lambda r: r.choice(BILLERS),
    "device": lambda r: r.choice(["new phone", "same phone after update", "friend's phone"]),
    "kyc_status": lambda r: r.choice(["pending", "rejected"]),
    "account_status": lambda r: r.choice(["dormant", "frozen", "blocked pending KYC update"]),
    "channel": lambda r: r.choice(["mobile banking", "internet banking"]),
    "card_type": lambda r: r.choice(["debit card", "credit card"]),
    "time": lambda r: r.choice(["2 am", "late last night", "this morning", "yesterday evening"]),
    "atm_location_fictional": lambda r: r.choice(PLACES),
    "bank_name_fictional": lambda r: "their linked bank",
}


def sample_facts(slots: list[str], rng: random.Random) -> dict[str, Any]:
    return {k: FACT_SAMPLERS[k](rng) for k in slots if k in FACT_SAMPLERS}


def _pick(dist: dict[str, float], rng: random.Random) -> str:
    return rng.choices(list(dist), weights=list(dist.values()))[0]


# ---------------------------------------------------------------- sampling

class _Balancer:
    """Hands out categories least-used-first, so every category gets close to equal coverage."""

    def __init__(self, items: list[str], rng: random.Random):
        self.items, self.rng, self.used = list(items), rng, Counter()

    def next(self, exclude: set[str] = frozenset()) -> str:
        pool = [i for i in self.items if i not in exclude]
        low = min(self.used[i] for i in pool)
        choice = self.rng.choice([i for i in pool if self.used[i] == low])
        self.used[choice] += 1
        return choice


def _counts(n: int, mix: dict[str, float]) -> dict[str, int]:
    raw = {k: n * v for k, v in mix.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def generate_scenarios(cfg: GenConfig, tax: Taxonomy) -> list[Scenario]:
    out: list[Scenario] = []
    for domain in tax.domains:
        out.extend(_domain_scenarios(cfg, tax, domain))
    return out


def _domain_scenarios(cfg: GenConfig, tax: Taxonomy, domain: str) -> list[Scenario]:
    rng = random.Random(f"{cfg.seed}:{domain}")
    dom = tax.domain(domain)
    fb = tax.fallback
    regular = [c for c in dom.categories if c != fb]
    bal = _Balancer(regular, rng)
    counts = _counts(cfg.scenarios_per_domain, cfg.family_mix)
    out: list[Scenario] = []
    seq = 0

    def make(family: str, issue: str | None, issues_all: list[str], group: str | None = None, **kw) -> Scenario:
        nonlocal seq
        seq += 1
        sid = f"{cfg.name}-{domain}-{seq:05d}"
        slots = sorted({s for c in issues_all if c in dom.categories for s in dom.categories[c].facts})
        facts = sample_facts(slots, rng)
        style = {k: _pick(getattr(cfg.style, k), rng) for k in ("length", "tone", "formality")}
        unresolvable = family in ("vague", "multi_issue:no_priority")
        target = None
        if unresolvable:
            keys = issues_all if family == "multi_issue:no_priority" else list(dom.categories)
            target = {k: round(1 / len(keys), 6) for k in keys}
        visible = list(facts)
        withheld = kw.pop("withheld", [])
        for w in withheld:
            if w in visible:
                visible.remove(w)
        return Scenario(scenario_id=sid, scenario_group_id=group or sid, domain=domain, taxonomy_version=tax.version,
                        family=family, issue=None if unresolvable else issue, issue_target=target,
                        clarification_needed=unresolvable, issues_all=issues_all, facts=facts, visible_facts=visible,
                        withheld_facts=withheld, style=style, **kw)

    def d(cid: str) -> str:
        """Plain-language situation for a category: its description, lower-cased. The renderer never sees ids."""
        text = dom.categories[cid].description
        return text if text[:2].isupper() else text[0].lower() + text[1:]     # keep acronyms: "KYC or ..."

    # contrast pairs first: their categories count toward the balancer, so later families fill the gaps
    for family, n in sorted(counts.items(), key=lambda kv: kv[0] != "contrast"):
        if family == "contrast":
            pairs = dom.contrasts
            for i in range(n // 2):
                a, b, axis = pairs[i % len(pairs)]
                gid = f"{cfg.name}-{domain}-cg{i:04d}"
                for this, other in ((a, b), (b, a)):
                    bal.used[this] += 1
                    out.append(make(f"contrast:{a}/{b}", this, [this], group=gid,
                                    instructions=f"Situation: {d(this)}. It must clearly not read as: {d(other)}. The deciding "
                                                 f"difference is {axis}; the message states it explicitly."))
            if n % 2:
                c = bal.next()
                out.append(make("single", c, [c], instructions=f"Situation: {d(c)}."))
        elif family == "out_of_scope":
            for _ in range(n):
                topic = rng.choice(dom.categories[fb].include)
                out.append(make("out_of_scope", fb, [], instructions=f"A clear, specific request about: {topic}. It is not one "
                                "of the problems the service lists, and it is not vague."))
        elif family == "vague":
            for _ in range(n):
                hidden = bal.next()
                out.append(make("vague", None, [], withheld=["underlying_issue"],
                                instructions="The customer only says something is wrong and asks for help, without saying what. "
                                             "Nothing in the message may hint at what the problem is."))
                out[-1].facts["underlying_issue"] = hidden
                out[-1].withheld_facts = ["underlying_issue"]
        elif family.startswith("multi_issue"):
            for _ in range(n):
                a = bal.next()
                b = bal.next(exclude={a, *dom.categories[a].neighbors})
                if family == "multi_issue:stated_priority":
                    out.append(make(family, b, [a, b], stated_priority=b,
                                    instructions=f"Two problems: (1) {d(a)}; (2) {d(b)}. The customer explicitly asks to sort "
                                                 f"out (2) first and leave (1) for later."))
                else:
                    out.append(make(family, None, [a, b],
                                    instructions=f"Two problems with equal weight: (1) {d(a)}; (2) {d(b)}. The customer does not "
                                                 f"say which to handle first."))
        else:
            for _ in range(n):
                c = bal.next()
                nb = dom.categories[c].neighbors
                extra = {
                    "single": "",
                    "negation": f" The message explicitly rules out a neighbouring problem ({d(nb[0]) if nb else 'another problem'}) "
                                "with a negation, e.g. 'that part is fine, but ...'.",
                    "corrected_statement": " The customer first says something misleading, then corrects themselves "
                                           "('not that, actually ...'). The corrected statement is the real situation.",
                    "irrelevant_background": " Add one or two sentences of irrelevant personal background before the problem.",
                    "missing_details": " Leave out secondary details such as amounts, dates or IDs; the problem itself stays clear.",
                }[family]
                withheld = [k for k in ("amount_npr", "order_id", "days_waiting", "days_late") if family == "missing_details"]
                out.append(make(family, c, [c], withheld=withheld, instructions=f"Situation: {d(c)}.{extra}"))
    return out
