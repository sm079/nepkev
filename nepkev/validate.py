"""Deterministic record validation: schema, script/language, evidence, labels, leakage, duplicates.

Every check is local and reproducible. The LLM verifier's verdict (nepkev.generation.run.verify) sits under
`validation.checks["verifier"]`, and the adjudicator's under `["adjudicator"]`: a disputed record is accepted only if
the adjudicator agrees with its gold label, and stays `queued` (unused) while no adjudicator has seen it.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .records import DEVA_TO_ASCII, REALSTYLE, Record, normalize
from .taxonomy import Taxonomy

DEVA_LETTER = re.compile(r"[ऀ-ॣॱ-ॿ]")

# Function words that separate Nepali from Hindi. Deliberately small and high-precision: words both
# languages use (के, को, ko, paisa) are left out.
NE_DEVA = {"छ", "छैन", "मा", "भयो", "भो", "मेरो", "मैले", "लाई", "बाट", "गर्ने", "हो", "होइन", "पनि", "अनि", "तर", "भएको",
           "गरें", "गरेँ", "आयो", "आएन", "हुँदैन", "कसरी", "कति", "अझै", "गर्नुस्", "गरिदिनु", "चाहिँ", "त", "न", "रैछ", "छु"}
HI_DEVA = {"है", "हैं", "नहीं", "मेरा", "मेरे", "मेरी", "का", "की", "लिए", "गया", "गई", "हुआ", "रहा", "रही", "और", "क्या", "आप",
           "जल्दी", "लेकिन", "अभी", "बहुत", "क्योंकि", "चाहिए", "मुझे", "हमें", "करना", "करें", "कीजिए", "वाला", "थी"}
NE_ROMAN = {"cha", "chha", "xa", "chaina", "chhaina", "xaina", "ma", "bhayo", "vayo", "bho", "mero", "maile", "lai", "bata", "garne",
            "garna", "tara", "ani", "pani", "ni", "ho", "haina", "aayo", "ayo", "katyo", "katiyo", "hudaina", "kasari", "kati", "ajhai", "ajai",
            "garnus", "garnu", "vako", "bhako", "vaxa", "raixa", "khata", "milxa", "milcha", "hunxa", "k", "aako", "gareko", "pugena"}
HI_ROMAN = {"nahi", "nahin", "mera", "mere", "meri", "kya", "gaya", "gayi", "hua", "raha", "rahi", "hoga", "karo", "kijiye", "aur", "hai", "hain",
            "pehle", "pehli", "pehla", "bhi", "lekin", "mujhe", "abhi", "kyunki", "kyun", "karna", "karein", "karo", "aap", "apna", "mein"}

NEGATION = {"छैन", "होइन", "हैन", "नाइँ", "नगरेको", "हुँदैन", "आएन", "पुगेन", "निस्केन", "दिएनन्", "नभएको", "आउँदैन", "खुल्दैन", "मिल्दैन", "चाहिँदैन",
            "chaina", "chhaina", "xaina", "haina", "hoina", "hudaina", "aayena", "ayena", "aaena", "pugena", "pugyena", "bhaena", "niskena", "dienan", "vayena", "bhayena",
            "audaina", "aaudaina", "khuldaina", "mildaina", "nagareko", "navako", "not", "no", "never", "didn't", "didnt", "dont", "don't"}
# Nepali negation is mostly morphological: -एन/-ैन/-दैन endings (गएन, छैन, चल्दैन), the न- prefix on participles
# (नकाटिएको, नभएको, नहुने) and हैन/होइन. Romanized text has the same shapes (gaena, chaldaina, nagareko).
NEG_DEVA = re.compile(r"^(?:.+(?:एन|ेन|ैन|दैन|ैनन्|ेनन्|एनन्|िनँ)|न[ऀ-ॿ]+(?:को|ने|ई|एर|ेर|ी|का))$")
NEG_ROMAN = re.compile(r"^(?:.+(?:aina|ena|daina|ainan|enan)|na[a-z]+(?:ko|ne|i|era|kai))$")

THOUSAND = re.compile(r"(\d+(?:\.\d+)?)\s*(?:k\b|hajar\b|hazar\b|हजार)", re.I)
LAKH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:lakh\b|lac\b|लाख)", re.I)
PLAIN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def amounts_in(text: str) -> set[float]:
    """Every amount a message states, in rupees, whatever the format (1,500 / १५०० / 1.5k / ५ हजार / 2 lakh)."""
    t = text.translate(DEVA_TO_ASCII)
    out: set[float] = set()
    for m in THOUSAND.finditer(t):
        out.add(round(float(m.group(1)) * 1_000, 2))
    for m in LAKH.finditer(t):
        out.add(round(float(m.group(1)) * 100_000, 2))
    for m in PLAIN.finditer(t):
        try:
            out.add(round(float(m.group().replace(",", "")), 2))
        except ValueError:
            pass
    return out


def words(text: str) -> list[str]:
    # Devanagari block minus the danda/double danda (U+0964/0965), which are sentence punctuation
    return re.findall(r"[\wऀ-ॣ०-ॿ'‌‍]+", text.lower())


# Nepali attaches postpositions and verb endings to the word (फोनमा, खाताबाट, गयो, हुन्छ); Hindi writes
# postpositions as separate words (फोन में), which HI_DEVA catches.
NE_DEVA_SUFFIX = re.compile(r"[ऀ-ॣ०-ॿ]{2,}(मा|लाई|बाट|सँग|ले|को|दैन|्छ|्छु|्छन्|छन्|एँ|ेँ|ें|यो|ियो|्यो|एको|ेको|ाको|इसक्यो|ेछ|एन|ेन|ैन|नुस्|ुहोस्|िदिनु)$")
NE_ROMAN_SUFFIX = re.compile(r"\w{2,}(xa|chha|xaina|chaina|chhaina|eko|yeko|ena|yena|daina|nus|nuhos|raixa|sakyo)$")


def detect_script(text: str) -> str:
    """'deva' (Devanagari letters only), 'roman' (Latin letters only) or 'mixed' (both: never valid).
    Digits, punctuation and emoji do not count."""
    has_dev, has_lat = bool(DEVA_LETTER.search(text)), bool(re.search(r"[A-Za-z]", text))
    return "mixed" if has_dev and has_lat else "deva" if has_dev else "roman"


def nepali_hindi_scores(text: str, script: str) -> tuple[int, int]:
    ws = words(text)
    ne, hi, suf = (NE_DEVA, HI_DEVA, NE_DEVA_SUFFIX) if script == "deva" else (NE_ROMAN, HI_ROMAN, NE_ROMAN_SUFFIX)
    n_ne = sum(w in ne or bool(suf.match(w)) for w in ws)
    return n_ne, sum(w in hi for w in ws)


def has_negation(text: str) -> bool:
    ws = set(words(text))
    return bool(ws & NEGATION) or any(NEG_DEVA.match(w) or NEG_ROMAN.match(w) for w in ws)


@dataclass
class Finding:
    reason: str
    detail: str = ""


def check_record(rec: Record, tax: Taxonomy) -> list[Finding]:
    """Per-record checks. Empty list = passes."""
    f: list[Finding] = []
    if rec.taxonomy_version != tax.version:
        return [Finding("taxonomy_version", f"{rec.taxonomy_version} != {tax.version}")]
    if rec.domain not in tax.domains:
        return [Finding("unknown_domain", rec.domain)]
    cats = set(tax.categories(rec.domain))
    msg = rec.message

    # length
    if len(msg) < 8:
        f.append(Finding("too_short", str(len(msg))))
    if len(msg) > 700:
        f.append(Finding("too_long", str(len(msg))))

    # script: a clean message is pure Devanagari or pure Latin script; app-review-style Devanagari may keep English
    # words in Latin letters, as Devanagari reviews do. Romanized text never contains Devanagari.
    got = detect_script(msg)
    if got != rec.script and not (REALSTYLE in rec.style_tags and rec.script == "deva" and got == "mixed"):
        f.append(Finding("script_mismatch", f"declared {rec.script}, content is {got}"))

    # Nepali, not Hindi or plain English
    n_ne, n_hi = nepali_hindi_scores(msg, rec.script)
    if n_ne == 0:
        f.append(Finding("not_nepali", "no Nepali function words"))
    elif n_hi > n_ne:
        f.append(Finding("hindi_not_nepali", f"{n_hi} Hindi vs {n_ne} Nepali markers"))

    # labels against taxonomy and family
    lab = rec.labels
    if lab.issue is not None and lab.issue not in cats:
        f.append(Finding("unknown_label", lab.issue))
    if lab.issue_target and (bad := set(lab.issue_target) - cats):
        f.append(Finding("unknown_label", f"target keys {sorted(bad)}"))
    if bad := set(rec.issues_all) - cats - {""}:
        # issues outside the taxonomy are allowed only for `other` records (e.g. staff_complaint)
        if lab.issue != tax.fallback:
            f.append(Finding("unknown_label", f"issues_all {sorted(bad)}"))
    fam = rec.difficulty_family
    unresolvable = fam == "vague" or fam == "multi_issue:no_priority"
    if unresolvable != lab.clarification_needed:
        f.append(Finding("label_family_mismatch", f"{fam} with clarification_needed={lab.clarification_needed}"))
    if unresolvable and lab.issue is not None:
        f.append(Finding("label_family_mismatch", f"{fam} must not carry a hard issue label"))
    if not unresolvable and lab.issue is None:
        f.append(Finding("label_family_mismatch", f"{fam} needs an issue label"))
    if fam == "multi_issue:stated_priority" and (rec.stated_priority != lab.issue or len(rec.issues_all) < 2):
        f.append(Finding("label_family_mismatch", "stated priority must equal the issue label and name 2+ issues"))

    # evidence: visible numeric facts must be in the text
    have = amounts_in(msg)
    for k in rec.visible_facts:
        v = rec.scenario_facts.get(k)
        if k.endswith("_npr") and isinstance(v, (int, float)) and round(float(v), 2) not in have:
            f.append(Finding("missing_evidence", f"{k}={v} not stated"))
    if ("negation" in rec.style_tags or fam.startswith("negation")) and not has_negation(msg):
        f.append(Finding("missing_negation", "tagged negation but no negation marker"))

    # leakage of label machinery into the text
    low = msg.lower()
    spaced = normalize(msg)
    for cid in cats:
        if "_" in cid and cid in low:
            f.append(Finding("label_leakage", cid))
        elif cid.count("_") >= 2 and cid.replace("_", " ") in spaced:
            # "payment failed debited": an id written as words. Two-word ids ("order status") are ordinary phrases.
            f.append(Finding("label_leakage", f"{cid} written as words"))
    for cid, desc in tax.criteria(rec.domain).items():
        if normalize(desc) in normalize(msg):
            f.append(Finding("label_leakage", f"description of {cid}"))
    return f


# ---------------------------------------------------------------- near-duplicates (MinHash LSH)

def shingles(text: str, k: int = 5) -> set[str]:
    t = normalize(text)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def _minhash(sh: set[str], n: int = 64) -> list[int]:
    hs = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") for s in sh]
    out = []
    for i in range(n):
        a, b = 0x9E3779B97F4A7C15 * (2 * i + 1), 0xBF58476D1CE4E5B9 * (i + 7)
        out.append(min(((h * a + b) & 0xFFFFFFFFFFFFFFFF) for h in hs))
    return out


def near_duplicate_pairs(texts: dict[str, str], groups: dict[str, str], threshold: float = 0.85,
                         bands: int = 16, rows: int = 4) -> list[tuple[str, str, float]]:
    """Pairs of ids from *different* groups whose 5-gram Jaccard is >= threshold."""
    sh = {i: shingles(t) for i, t in texts.items()}
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for i, s in sh.items():
        mh = _minhash(s, bands * rows)
        for b in range(bands):
            buckets[(b, *mh[b * rows:(b + 1) * rows])].append(i)
    seen, out = set(), []
    for ids in buckets.values():
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = sorted((ids[x], ids[y]))
                if (a, b) in seen or groups[a] == groups[b]:
                    continue
                seen.add((a, b))
                if (j := jaccard(sh[a], sh[b])) >= threshold:
                    out.append((a, b, round(j, 3)))
    return out


# ---------------------------------------------------------------- dataset pass

@dataclass
class Report:
    n: int = 0
    status: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    by_domain_issue: dict = field(default_factory=lambda: defaultdict(Counter))
    by_script: Counter = field(default_factory=Counter)
    by_family: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"n": self.n, "status": dict(self.status), "rejection_reasons": dict(self.reasons),
                "accepted_by_domain_issue": {d: dict(c) for d, c in self.by_domain_issue.items()},
                "accepted_by_script": dict(self.by_script), "accepted_by_family": dict(self.by_family),
                "warnings": self.warnings}


def validate_records(recs: list[Record], tax: Taxonomy, near_dup_threshold: float = 0.85) -> Report:
    """Validate in place (sets rec.validation) and return the report. First occurrence wins for duplicates."""
    rep = Report(n=len(recs))
    for r in recs:
        found = check_record(r, tax)
        # keep earlier checks (the verifier's verdict) and replace only this pass's findings
        r.validation.checks = {**r.validation.checks, "findings": [f"{x.reason}: {x.detail}" for x in found]}
        if found:
            r.validation.status, r.validation.rejection_reason = "rejected", found[0].reason
        else:
            ver = r.validation.checks.get("verifier") or {}
            adj = r.validation.checks.get("adjudicator") or {}
            if ver.get("agrees") is False and "agrees" in adj:
                # the adjudicator settles a dispute: its agreement with the gold label accepts, otherwise reject
                r.validation.status = "accepted" if adj["agrees"] else "rejected"
                r.validation.rejection_reason = None if adj["agrees"] else "label_disputed"
            else:
                r.validation.status = "queued" if ver.get("agrees") is False else "accepted"
                r.validation.rejection_reason = None

    live = [r for r in recs if r.validation.status != "rejected"]
    first_by_hash: dict[str, Record] = {}
    for r in live:
        prev = first_by_hash.setdefault(r.content_hash, r)
        if prev is not r and prev.scenario_group_id != r.scenario_group_id:
            r.validation.status, r.validation.rejection_reason = "rejected", "exact_duplicate"
            r.validation.checks["duplicate_of"] = prev.record_id
    live = [r for r in live if r.validation.status != "rejected"]
    by_id = {r.record_id: r for r in live}
    order = {r.record_id: i for i, r in enumerate(recs)}
    for a, b, j in near_duplicate_pairs({r.record_id: r.message for r in live},
                                        {r.record_id: r.scenario_group_id for r in live}, near_dup_threshold):
        later = by_id[max(a, b, key=order.__getitem__)]
        if later.validation.status != "rejected":
            later.validation.status, later.validation.rejection_reason = "rejected", "near_duplicate"
            later.validation.checks["near_duplicate_of"] = {"id": min(a, b, key=order.__getitem__), "jaccard": j}

    for r in recs:
        rep.status[r.validation.status] += 1
        if r.validation.status == "rejected":
            rep.reasons[r.validation.rejection_reason] += 1
        elif r.validation.status == "accepted":
            rep.by_domain_issue[r.domain][r.labels.issue or "(clarify)"] += 1
            rep.by_script[r.script] += 1
            rep.by_family[r.difficulty_family.split(":")[0]] += 1
    for d, c in rep.by_domain_issue.items():
        total = sum(c.values())
        for cid in tax.categories(d):
            if total >= 100 and c[cid] < 0.05 * total:
                rep.warnings.append(f"{d}/{cid}: {c[cid]} of {total} accepted ({c[cid] / total:.1%} < 5%)")
    return rep
