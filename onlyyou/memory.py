"""Wearer-only transcript -> moments -> journal where every line cites its moments.

If ANTHROPIC_API_KEY or OPENAI_API_KEY is set, an LLM writes the journal; otherwise a
simple extractive journal is produced. Either way `check_grounding` verifies that every
line cites real moments and that its content words actually appear in them.
"""

import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass

STOP = set("""the a an and or but if then so of to in on at for with from by is are was were be been
it this that these those i you he she we they me my your our their as about just really very not no
yes do did does have has had will would can could should there here what when where who how all
le la les un une des et ou mais de du au aux en dans sur pour par avec est sont était qui que quoi ce
cette ces il elle ils elles nous vous je tu on ne pas plus se sa son ses leur leurs""".split())


@dataclass
class Moment:
    id: str
    start: float
    end: float
    text: str
    words_per_s: float


def build_moments(segments, merge_gap=2.0):
    merged = []
    for s, e, t in segments:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e, merged[-1][2] + " " + t)
        else:
            merged.append((s, e, t))
    return [Moment(f"m{i + 1}", round(s, 1), round(e, 1), t, round(len(t.split()) / max(e - s, 0.5), 2))
            for i, (s, e, t) in enumerate(merged)]


def content_words(text):
    return {w for w in re.findall(r"[\w']+", text.lower()) if len(w) > 2 and w not in STOP}


PROMPT = """You are writing a short private journal for the person wearing an AI pendant.
Below are moments: things the wearer said today, with ids and timestamps.
Write at most 6 journal lines in second person ("You ..."). Every line must end with the ids
of the moments it is based on, like [m2] or [m1][m3]. Do not state anything that is not
supported by the cited moments. No headings, no preamble.

Moments:
{moments}"""


def _llm(prompt):
    if os.environ.get("ANTHROPIC_API_KEY"):
        body = {"model": os.environ.get("ONLYYOU_LLM_MODEL", "claude-sonnet-4-5"), "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", json.dumps(body).encode(), {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
            "content-type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=60))["content"][0]["text"]
    if os.environ.get("OPENAI_API_KEY"):
        body = {"model": os.environ.get("ONLYYOU_LLM_MODEL", "gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", json.dumps(body).encode(), {
            "authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "content-type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]
    return None


def journal(moments):
    if not moments:
        return ""
    listing = "\n".join(f"[{m.id}] {m.start:.0f}s-{m.end:.0f}s: {m.text}" for m in moments)
    try:
        text = _llm(PROMPT.format(moments=listing))
    except Exception as e:  # network / auth problems fall back to extractive
        print(f"LLM journal failed ({e}); using extractive journal")
        text = None
    if text:
        return text.strip()
    lines = []
    for m in moments[:6]:
        snippet = m.text if len(m.text) < 140 else m.text[:137] + "..."
        pace = " You were talking fast." if m.words_per_s > 3.2 else ""
        lines.append(f'- At {m.start:.0f}s you said: "{snippet}".{pace} [{m.id}]')
    return "\n".join(lines)


def check_grounding(text, moments, min_overlap=0.3):
    """Each line must cite existing moments, and >= min_overlap of its content words
    must appear in the cited moments."""
    by_id = {m.id: m for m in moments}
    results = []
    for line in [l for l in text.splitlines() if l.strip()]:
        cites = re.findall(r"\[(m\d+)\]", line)
        valid = [c for c in cites if c in by_id]
        words = content_words(re.sub(r"\[m\d+\]", "", line)) - {"talking", "fast", "said", "you"}
        support = set().union(*(content_words(by_id[c].text) for c in valid)) if valid else set()
        overlap = len(words & support) / len(words) if words else 1.0
        results.append({"line": line, "cites": cites, "valid_cites": len(valid) == len(cites) and bool(cites),
                        "overlap": round(overlap, 2),
                        "grounded": bool(valid) and len(valid) == len(cites) and overlap >= min_overlap})
    n = len(results)
    return {"lines": results, "grounded_share": sum(r["grounded"] for r in results) / n if n else 1.0}


def to_json(moments):
    return [asdict(m) for m in moments]
