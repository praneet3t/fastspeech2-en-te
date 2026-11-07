"""Deterministic front ends; original input strings are never modified in place.

Telugu units are orthography-derived phone hypotheses, not verified phonetic
transcriptions. Inherent vowels, vowel signs and virama are explicit. The same
word pronunciations are used in the MFA dictionary and acoustic-model inference.
"""

from dataclasses import dataclass
import re
import unicodedata
from functools import lru_cache
import regex
from num2words import num2words
from threadpoolctl import threadpool_limits

VOWELS = dict(
    zip(
        "అఆఇఈఉఊఋౠఌౡఎఏఐఒఓఔ",
        [
            "a",
            "aa",
            "i",
            "ii",
            "u",
            "uu",
            "r_v",
            "rr_v",
            "l_v",
            "ll_v",
            "e",
            "ee",
            "ai",
            "o",
            "oo",
            "au",
        ],
    )
)
SIGNS = dict(
    zip(
        "ాిీుూృౄెేైొోౌౢౣ",
        [
            "aa",
            "i",
            "ii",
            "u",
            "uu",
            "r_v",
            "rr_v",
            "e",
            "ee",
            "ai",
            "o",
            "oo",
            "au",
            "l_v",
            "ll_v",
        ],
    )
)
CONSONANTS = dict(
    zip(
        "కఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహ",
        [
            "k",
            "kh",
            "g",
            "gh",
            "ng",
            "c",
            "ch",
            "j",
            "jh",
            "ny",
            "tt",
            "tth",
            "dd",
            "ddh",
            "nn",
            "t",
            "th",
            "d",
            "dh",
            "n",
            "p",
            "ph",
            "b",
            "bh",
            "m",
            "y",
            "r",
            "rr",
            "l",
            "ll",
            "v",
            "sh",
            "ss",
            "s",
            "h",
        ],
    )
)
MARKS = {"ం": "anusvara", "ఁ": "candrabindu", "ః": "visarga"}
PUNCTUATION = set('.,!?;:—–-()[]{}"“”‘’/।॥…')


@dataclass
class TextInput:
    original: str
    normalized: str
    words: list[str]
    pronunciations: list[list[str]]
    transformations: list[str]

    @property
    def units(self) -> list[str]:
        return [phone for phones in self.pronunciations for phone in phones]


def telugu_word(word: str) -> list[str]:
    """Map a complete Telugu word without separating combining sequences."""
    phones = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in VOWELS:
            phones.append(VOWELS[ch])
        elif ch in CONSONANTS:
            phones.append(CONSONANTS[ch])
            following = word[i + 1] if i + 1 < len(word) else ""
            if following == "్":
                i += 1
            elif following in SIGNS:
                phones.append(SIGNS[following])
                i += 1
            else:
                phones.append("a")
        elif ch in SIGNS:
            # Supplied test text contains orphan/repeated vowel signs. Preserve
            # the string and expose this literal fallback in the normalization log.
            phones.append(SIGNS[ch])
        elif ch in MARKS:
            phones.append(MARKS[ch])
        elif ch in ("\u200c", "\u200d"):
            pass  # Join controls carry no independent sound.
        else:
            raise ValueError(
                f"Unsupported Telugu symbol {ch!r} U+{ord(ch):04X} in {word!r}"
            )
        i += 1
    if not phones:
        raise ValueError(f"Empty pronunciation: {word!r}")
    return ["te_" + p for p in phones]


@lru_cache(maxsize=1)
def english_resources():
    import cmudict
    from g2p_en import G2p

    return cmudict.dict(), G2p()


@lru_cache(maxsize=50000)
def english_word(word: str) -> tuple[str, ...]:
    dictionary, fallback = english_resources()
    # Explicit context choice for the assignment heteronym; callers pass a tag.
    if word == "conduct_noun":
        return ("K", "AA1", "N", "D", "AH0", "K", "T")
    if word == "conduct_verb":
        return ("K", "AH0", "N", "D", "AH1", "K", "T")
    if word in dictionary:
        return tuple(dictionary[word][0])
    # G2P uses small matrix/vector products: large BLAS thread pools slow it down.
    with threadpool_limits(limits=1, user_api="blas"):
        phones = tuple(p for p in fallback(word) if p != " ")
    if not phones or any(not re.fullmatch(r"[A-Z]+[012]?", p) for p in phones):
        raise ValueError(f"English G2P failed for {word!r}: {phones}")
    return phones


def telugu_cardinal(number: int) -> str:
    """Indian grouping; use the package's small-number lexicon, not its buggy
    thousand/ordinal merger (2026 was rendered as 'రెండు వేయిల ...').
    No corpus transcripts are rewritten to disguise unusual assignment spacing.
    """
    if number < 0 or number >= 10**9:
        raise ValueError("Telugu cardinal range is 0 <= n < 1 billion")
    if number < 100:
        return unicodedata.normalize("NFC", num2words(number, lang="te")).replace(
            "-", " "
        )
    for scale, singular, plural, linking in [
        (10**7, "కోటి", "కోట్లు", "కోట్ల"),
        (10**5, "లక్ష", "లక్షలు", "లక్షల"),
        (1000, "వెయ్యి", "వేలు", "వేల"),
        (100, "వంద", "వందలు", "వందల"),
    ]:
        if number >= scale:
            quotient, remainder = divmod(number, scale)
            if quotient == 1:
                prefix = singular
            else:
                prefix = (
                    telugu_cardinal(quotient) + " " + (linking if remainder else plural)
                )
            return prefix + (" " + telugu_cardinal(remainder) if remainder else "")
    raise AssertionError("Unreachable cardinal branch")


def telugu_ordinal(number: int) -> str:
    if number == 1:
        return "మొదటి"
    word = telugu_cardinal(number)
    if word.endswith(("ు", "ి")):
        word = word[:-1]
    return word + "వ"


def normalize(text: str, language: str) -> tuple[str, list[str]]:
    output = unicodedata.normalize("NFC", text)
    changes = []
    if output != text:
        changes.append("Unicode NFC")
    if language == "english":

        def time(match):
            hour, minute, meridiem = match.groups()
            if not (0 <= int(hour) < 24 and 0 <= int(minute) < 60):
                raise ValueError("Invalid English time")
            # CMUdict's first pronunciation of 'a' is the article AH0.
            # Spoken letter names make the intended meridiem explicit.
            suffix = {"AM": "ay em", "PM": "pee em"}.get((meridiem or "").upper(), "")
            return f"{num2words(int(hour))} {num2words(int(minute))} {suffix}"

        output = re.sub(r"\b(\d{1,2}):(\d{2})\s*(AM|PM)?\b", time, output, flags=re.I)
        output = re.sub(r"\d+", lambda m: num2words(int(m[0])), output)
        output = output.replace("’", "'").lower()
    elif language == "telugu":
        output = "".join(
            str(unicodedata.digit(c)) if c.isdecimal() else c for c in output
        )

        def time(match):
            hour, minute = map(int, match.group(1, 2))
            if not (0 <= hour < 24 and 0 <= minute < 60):
                raise ValueError("Invalid Telugu time")
            ending = "నిమిషాలకు" if match.group(3) else "నిమిషాలు"
            return (
                telugu_cardinal(hour)
                + " గంటల "
                + telugu_cardinal(minute)
                + " "
                + ending
            )

        output = re.sub(r"(\d{1,2}):(\d{2})(\s*గంటలకు)?", time, output)
        output = re.sub(r"(\d+)వ", lambda m: telugu_ordinal(int(m[1])), output)
        output = re.sub(r"\d+", lambda m: telugu_cardinal(int(m[0])), output)
    else:
        raise ValueError(f"Unknown language: {language}")
    if output != unicodedata.normalize("NFC", text):
        changes.append(
            "Telugu cardinal, ordinal and time expansion; date word order retained"
            if language == "telugu"
            else "English case and number/time expansion"
        )
    output = regex.sub(r"\s+", " ", output).strip()
    if "\n" in text or "\r" in text:
        changes.append("Line breaks collapsed internally; raw text preserved")
    if regex.search(r"\s{2,}", text):
        changes.append("Repeated whitespace collapsed internally; raw text preserved")
    return output, changes


def frontend(text: str, language: str) -> TextInput:
    """Return preserved raw text, normalized words and matching pronunciations."""
    normalized, changes = normalize(text, language)
    if language == "english":
        unsupported = regex.sub(r"[a-z\s.,!?;:\-—–()\[\]{}'\"/…]", "", normalized)
        if unsupported:
            raise ValueError(f"Unsupported English input symbols: {unsupported!r}")
        pieces = re.findall(r"[a-z]+(?:'[a-z]+)*|[.,!?;:]", normalized)
    else:
        pieces = regex.findall(
            r"[\p{Script=Telugu}\p{M}\u200c\u200d]+|[^\s]", normalized
        )
    words, pronunciations = [], []
    dictionary = english_resources()[0] if language == "english" else None
    for piece in pieces:
        if piece in PUNCTUATION:
            # Explicit pause token included as a lexical item for MFA as well.
            if piece in ".,!?;:।॥…" and (not words or words[-1] != "pause"):
                words.append("pause")
                pronunciations.append(["sp"])
            continue
        if language == "english":
            word = piece
            if piece == "conduct":
                word = (
                    "conduct_verb"
                    if re.search(r"\b(will|to|can|may|would)\s+conduct\b", normalized)
                    else "conduct_noun"
                )
                changes.append(f"Explicit conduct context heuristic: {word}")
            elif piece not in dictionary:
                changes.append(f"English lexicon fallback: {piece}")
            phones = list(english_word(word))
        else:
            word = piece
            if word[0] in SIGNS or any(
                word[i] in SIGNS and word[i - 1] in SIGNS for i in range(1, len(word))
            ):
                changes.append(
                    f"Orphan/repeated Telugu vowel sign in {word!r}: literal vowel hypothesis, pronunciation unverified"
                )
            phones = telugu_word(word)
        words.append(word)
        pronunciations.append(phones)
    if not words or all(p == ["sp"] for p in pronunciations):
        raise ValueError("Text contains no linguistic units")
    return TextInput(text, normalized, words, pronunciations, changes)
