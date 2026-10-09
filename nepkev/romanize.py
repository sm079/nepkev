"""Devanagari -> Romanized Nepali, the way people type Nepali on a phone keyboard.

Every message has exactly one script. The generator writes pure Devanagari; this module derives the
Romanized version of a share of those messages:

1. casual contractions on the Devanagari side (भएको -> भाको), so the rest stays rule-based;
2. one typist profile per message (chh/ch/x, bh/v, ph/f, ...): spellings vary between messages but
   stay consistent within one;
3. English mixed in at a per-message level (none / light / heavy): loanwords always go back to English
   (अर्डरको -> "order ko"), and listed Nepali words switch with the level's probability (तर -> "but");
4. chat habits: ASCII digits, 5k shorthand, plz/acc, lowercase, dropped or doubled punctuation.

Romanization is rule-based (no lexicon of native words). Final inherent-vowel deletion follows a few
Nepali rules (घर -> ghar, but छ -> chha, छैन -> chhaina, पुगेन -> pugena, सकिन -> sakina) plus an exception
list, so a rare word can come out as a spelling typists might not use. That mild noise is
acceptable; derived messages are still kept only if they pass the record checks (amounts, negation,
script, Nepali markers).
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import yaml

from . import CONFIGS
from .records import Derivation, Record, Source
from .taxonomy import Taxonomy
from .validate import check_record

CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "b", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
}
VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
          "औ": "au", "ऑ": "o", "ॲ": "a"}
MATRAS = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e"}
HALANT, NUKTA, ANUSVARA, CANDRABINDU, VISARGA = "्", "़", "ं", "ँ", "ः"
LABIALS = set("पफबभम")
DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# Whole words whose romanization the rules get wrong.
EXCEPTIONS = {"तर": "tara", "अब": "aba", "जब": "jaba", "तब": "taba", "किन": "kina", "कृपया": "kripaya", "अनि": "ani",
              "हजुर": "hajur", "दाइ": "dai", "मसँग": "ma sanga", "मसंग": "ma sanga", "ज्ञान": "gyan", "नमस्ते": "namaste", "र": "ra"}
NATIVE_SPLIT = ("बाट", "लाई", "मा", "सँग", "संग", "देखि", "सम्म")   # case markers safe to write as separate words
ATTACHED_AFTER_CONSONANT = ("को", "का", "की", "ले")
DEVA_WORD = re.compile(r"[ऀ-ॣॱ-ॿ‌‍]+")
AMOUNT = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d{4,})(?![\d,])")


@dataclass(frozen=True)
class RomanizeConfig:
    version: str
    loanwords: dict[str, str]
    postpositions: dict[str, str]
    contractions: dict[str, str]
    english: dict[str, str]
    abbreviations: dict[str, list[str]]
    profile_options: dict[str, dict[str, float]]
    p: dict

    def sample_profile(self, rng: random.Random) -> dict[str, str]:
        """One typist's spelling habits, used for every word of one message."""
        return {k: rng.choices(list(v), weights=list(v.values()))[0] for k, v in self.profile_options.items()}

    def sample_english_level(self, rng: random.Random) -> tuple[str, float]:
        levels = self.p["english_levels"]
        name = rng.choices(list(levels), weights=[v[0] for v in levels.values()])[0]
        return name, levels[name][1]


@lru_cache
def load_romanize(name: str = "romanize") -> RomanizeConfig:
    with (CONFIGS / f"{name}.yaml").open(encoding="utf-8") as f:
        d = yaml.safe_load(f)
    cfg = RomanizeConfig(d["version"], d["loanwords"], d["postpositions"], d["contractions"], d["english"],
                         d["abbreviations"], d["typist_profile"], d["probabilities"])
    for k, v in {**cfg.loanwords, **cfg.english}.items():
        if not isinstance(v, str):
            raise ValueError(f"{name}.yaml: {k!r} maps to {v!r}; quote YAML words like \"no\"")
    return cfg


# ---------------------------------------------------------------- phonetic core

def _aksharas(word: str) -> list[str]:
    """Orthographic syllables (consonant clusters with their signs), enough for the deletion rules."""
    out, cur = [], ""
    for ch in word:
        if (ch in CONSONANTS or ch in VOWELS) and cur and not cur.endswith(HALANT):
            out.append(cur)
            cur = ""
        cur += ch
    if cur:
        out.append(cur)
    return out


def _keep_final_schwa(word: str) -> bool:
    ak = _aksharas(word)
    if len(ak) <= 1:
        return True                                   # छ -> chha, म -> ma
    last, prev = ak[-1], ak[-2]
    if last[-1] not in CONSONANTS or HALANT in last:
        return True                                   # ends in a vowel sign or a conjunct
    if last[0] in ("य", "छ"):
        return True                                   # समय -> samaya; देखाउँछ -> dekhauchha
    if last[0] not in ("न", "र"):
        return False                                  # सेट -> set, बजार -> bajar
    # verb endings -एन / -ैन / -इन (negative) and -एर (conjunctive) keep their vowel
    prev_v = prev.rstrip(ANUSVARA + CANDRABINDU)[-1]
    if prev_v in VOWELS and (len(ak) > 2 or word[0] not in VOWELS):
        return True                                   # पाइन -> paina, होइन -> hoina, आएन -> aaena, भएर -> bhaera (but एक -> ek)
    if prev_v in ("े", "ै"):
        return True                                   # पुगेन -> pugena, छैन -> chhaina, गरेर -> garera
    if prev_v == "ि" and len(ak) >= 3 and last[0] == "न":
        return True                                   # सकिन -> sakina (but दिन -> din)
    return False


def roman_word(word: str, profile: dict[str, str] | None = None) -> str:
    """One native Devanagari word -> Romanized, applying the typist profile."""
    profile = profile or {}
    word = word.replace("‌", "").replace("‍", "").replace(NUKTA, "")
    if word in EXCEPTIONS:
        return EXCEPTIONS[word]
    out: list[str] = []
    n = len(word)
    for i, ch in enumerate(word):
        nxt = word[i + 1] if i + 1 < n else ""
        if ch in CONSONANTS:
            if ch == "व" and i and word[i - 1] == HALANT:
                out.append("v" if profile.get(ch) == "v" else "w")     # स्व, श्व, द्व: koteshwor, swasthya, dwar
            else:
                out.append(profile.get(ch, CONSONANTS[ch]))
            if nxt in MATRAS or nxt == HALANT or (i + 1 == n and not _keep_final_schwa(word)):
                continue
            out.append("a")
        elif ch in VOWELS:
            out.append(profile.get(ch, VOWELS[ch]))
        elif ch in MATRAS:
            out.append(profile.get(ch, MATRAS[ch]))
        elif ch == ANUSVARA:
            if i + 1 < n:
                out.append("m" if nxt in LABIALS else "n")
        elif ch == VISARGA:
            out.append("h")
        elif ch not in (CANDRABINDU, HALANT):
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------- words and messages

def _english_of(word: str, table: dict[str, str], cfg: RomanizeConfig) -> str | None:
    """English for a word or a word + postposition: अर्डरको -> "order ko"."""
    if word in table:
        return table[word]
    for pp in sorted(cfg.postpositions, key=len, reverse=True):
        if word.endswith(pp) and word[: -len(pp)] in table:
            return f"{table[word[: -len(pp)]]} {cfg.postpositions[pp]}"
    return None


def _word(word: str, cfg: RomanizeConfig, profile: dict[str, str] | None, rng: random.Random | None = None,
          english_p: float = 0.0, ops: list[str] | None = None) -> str:
    if (en := _english_of(word, cfg.loanwords, cfg)) is not None:
        return en
    if english_p and rng is not None and (en := _english_of(word, cfg.english, cfg)) is not None and rng.random() < english_p:
        if ops is not None:
            ops.append(f"english:{en.split()[0]}")
        return en
    if word in cfg.postpositions:
        return cfg.postpositions[word]
    for pp in NATIVE_SPLIT:
        stem = word[: -len(pp)]
        # a stem ending in halant is a conjunct, not a word + postposition: जम्मा is "jamma", not "jam ma"
        if word.endswith(pp) and len(_aksharas(stem)) >= 2 and not stem.endswith(HALANT):
            return f"{roman_word(stem, profile)} {cfg.postpositions[pp]}"
    for pp in ATTACHED_AFTER_CONSONANT:
        stem = word[: -len(pp)]
        if word.endswith(pp) and len(_aksharas(stem)) >= 2 and stem[-1] in CONSONANTS:
            # noun ending in a bare consonant + attached postposition: रङको -> "rang ko", not "rangako"
            # (participles like गरेको end their stem in a vowel sign and are left alone)
            return f"{roman_word(stem, profile)} {cfg.postpositions[pp]}"
    return roman_word(word, profile)


def transliterate(text: str, cfg: RomanizeConfig | None = None, profile: dict[str, str] | None = None) -> str:
    """Deterministic core: Devanagari -> Romanized (loanwords to English, ASCII digits, '।' to '.')."""
    cfg = cfg or load_romanize()
    return _tidy(DEVA_WORD.sub(lambda m: _word(m.group(), cfg, profile), _prep(text)))


def _prep(text: str) -> str:
    return text.translate(DEVA_DIGITS).replace("।", ".").replace("॥", ".")


def _tidy(text: str) -> str:
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+([.,!?])", r"\1", text)).strip()


def romanize(text: str, rng: random.Random, cfg: RomanizeConfig | None = None) -> tuple[str, list[str]]:
    """One Romanized rendering of a Devanagari message with a sampled typist, English level and chat habits."""
    cfg = cfg or load_romanize()
    p = cfg.p
    ops: list[str] = []

    def contract(m: re.Match) -> str:
        w = m.group()
        if w in cfg.contractions and rng.random() < p["contraction_p"]:
            ops.append(f"contract:{w}")
            return cfg.contractions[w]
        return w
    text = DEVA_WORD.sub(contract, text)

    profile = cfg.sample_profile(rng)
    defaults = {k: next(iter(v)) for k, v in cfg.profile_options.items()}
    if habits := [f"{k}={v}" for k, v in sorted(profile.items()) if v != defaults[k]]:
        ops.append("typist:" + ",".join(habits))
    level, english_p = cfg.sample_english_level(rng)
    ops.append(f"english_level:{level}")
    out = _tidy(DEVA_WORD.sub(lambda m: _word(m.group(), cfg, profile, rng, english_p, ops), _prep(text)))

    if rng.random() < p["shorthand_p"]:
        cands = [m for m in AMOUNT.finditer(out) if int(m.group().replace(",", "")) % 100 == 0]
        if cands:
            m = rng.choice(cands)
            out = f"{out[:m.start()]}{int(m.group().replace(',', '')) / 1000:g}k{out[m.end():]}"
            ops.append("shorthand")

    def abbrev(m: re.Match) -> str:
        opts = cfg.abbreviations.get(m.group().lower())
        if opts and rng.random() < p["abbrev_p"]:
            ops.append(f"abbrev:{m.group().lower()}")
            return rng.choice(opts)
        return m.group()
    out = re.sub(r"[A-Za-z]+", abbrev, out)
    if rng.random() < p["lowercase_p"] and out != out.lower():
        out = out.lower()
        ops.append("lowercase")
    if rng.random() < p["drop_punct_p"]:
        new = _tidy(re.sub(r"(?<!\d)[,.;:!?](?!\d)|(?<=\d)[.!?](?!\d)", "", out))
        if new != out:
            out = new
            ops.append("drop_punct")
    elif rng.random() < p["double_punct_p"] and re.search(r"[?!]", out):
        out = re.sub(r"([?!])(?![?!])", r"\1\1", out, count=1)
        ops.append("double_punct")
    return out, ops


# ---------------------------------------------------------------- dataset step

def derive_roman_variants(recs: list[Record], tax: Taxonomy, share: float, seed: int = 0,
                          cfg: RomanizeConfig | None = None) -> tuple[list[Record], Counter]:
    """Romanized variants for a seeded `share` of Devanagari records. Children keep labels, group and lineage, and are kept
    only if they pass the per-record checks. Returns (children, rejection reasons)."""
    cfg = cfg or load_romanize()
    out: list[Record] = []
    dropped: Counter = Counter()
    for r in recs:
        if r.script != "deva":
            continue
        rng = random.Random(f"{seed}:roman:{r.record_id}")
        if rng.random() >= share:
            continue
        text, ops = romanize(r.message, rng, cfg)
        mock = "mock" in r.style_tags
        child = Record.model_validate({
            **r.model_dump(), "record_id": f"{r.scenario_id}-roman", "message": text, "script": "roman",
            "content_hash": "", "style_tags": [t for t in r.style_tags if t != "mock"] + ["derived", *(["mock"] if mock else [])],
            "source": Source(generator="romanize", prompt_version=cfg.version, seed=seed).model_dump(),
            "derivation": Derivation(parent_record_id=r.record_id, ops=ops, seed=seed).model_dump(),
            "validation": {"status": "pending", "rejection_reason": None, "checks": {}},
        })
        if findings := check_record(child, tax):
            dropped[findings[0].reason] += 1
            continue
        child.validation.status, child.validation.checks = "accepted", {"findings": [], "derived_from": r.record_id}
        out.append(child)
    return out, dropped
