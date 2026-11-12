"""The pinned datasets: the source of each file, its SHA-256, and how it becomes test instances.

The repository holds no dataset text. fetch() downloads each file from its source at a fixed revision and
refuses a file whose SHA-256 is not the pinned one, so every run reads the same bytes.
"""

import dataclasses
import datetime
import hashlib
import json
import os
import re
import urllib.request

from memorybench.system import Session, Turn

SOURCES = {
    "locomo": {
        "file": "locomo10.json",
        "url": "https://raw.githubusercontent.com/snap-research/locomo/"
               "cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc/data/locomo10.json",
        "sha256": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
        "bytes": 2805274,
        "license": "CC-BY-NC-4.0",
        "home": "https://github.com/snap-research/locomo",
    },
    "longmemeval-s": {
        "file": "longmemeval_s_cleaned.json",
        "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
               "98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json",
        "sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        "bytes": 277383467,
        "license": "MIT",
        "home": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
    },
}

LME_DATE = "%Y/%m/%d (%a) %H:%M"


@dataclasses.dataclass(frozen=True)
class Question:
    id: str
    text: str
    date: str
    answer: object  # the gold answer text; None when the answer track skips the question
    gold: tuple  # the ids of the turns that hold the evidence; empty when the evidence track skips it
    category: str


@dataclasses.dataclass(frozen=True)
class Instance:
    id: str
    sessions: tuple
    questions: tuple
    text_of: dict  # turn id -> turn text, for scoring; the system never gets it


def cache_dir():
    if os.environ.get("MEMORYBENCH_DATA"):
        return os.environ["MEMORYBENCH_DATA"]
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "memory-bench")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check(path, src):
    got = sha256_file(path)
    if got != src["sha256"]:
        raise ValueError("%s: SHA-256 %s is not the pinned %s (%s)" % (path, got, src["sha256"], src["file"]))


def fetch(name, path=None):
    """The path of the dataset file, checked against its pinned SHA-256. Downloads the file when absent."""
    src = SOURCES[name]
    if path:
        _check(path, src)
        return path
    path = os.path.join(cache_dir(), src["file"])
    if os.path.exists(path):
        _check(path, src)
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(src["url"], timeout=60) as r, open(part, "wb") as f:
            for block in iter(lambda: r.read(1 << 20), b""):
                h.update(block)
                f.write(block)
        if h.hexdigest() != src["sha256"]:
            raise ValueError("%s: SHA-256 %s is not the pinned %s" % (src["url"], h.hexdigest(), src["sha256"]))
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)
    return path


def instances(name, path):
    """The test instances of a dataset, in dataset order."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {"locomo": _locomo, "longmemeval-s": _longmemeval}[name](raw)


def _locomo(raw):
    """One instance for each conversation, its sessions in number order (the time order of the dataset).

    Evidence: the "evidence" ids of a question that name a turn. The answer track skips category 5
    (adversarial questions, most of them without an answer field), as most published LoCoMo results do.
    """
    out = []
    for s in raw:
        c = s["conversation"]
        keys = sorted((k for k in c if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
        sessions, text_of = [], {}
        for k in keys:
            turns = []
            for t in c[k]:
                extra = {"image_caption": t["blip_caption"]} if t.get("blip_caption") else {}
                turns.append(Turn(t["dia_id"], t["speaker"], t["text"], extra))
                text_of[t["dia_id"]] = t["text"]
            sessions.append(Session(k, c.get(k + "_date_time", ""), tuple(turns)))
        questions = []
        for i, q in enumerate(s["qa"]):
            ids = re.split(r"[;,\s]+", " ".join(str(e) for e in q.get("evidence") or []))
            gold = tuple(dict.fromkeys(p for p in ids if p in text_of))
            cat = str(q.get("category", ""))
            answer = None if cat == "5" or "answer" not in q else str(q["answer"])
            questions.append(Question("%s/%d" % (s["sample_id"], i), q["question"], "", answer, gold, cat))
        out.append(Instance(s["sample_id"], tuple(sessions), tuple(questions), text_of))
    return out


def _longmemeval(raw):
    """One instance for each question: its haystack of sessions, sorted by session date.

    The dataset order is not always the date order, and 13 haystacks repeat a session id (with other
    content): the second session with an id gets the suffix "~2". Evidence: the turns marked has_answer.
    Abstention questions ("_abs") have no evidence; the answer track keeps them.
    """
    out = []
    for x in raw:
        rows = list(zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]))
        try:
            keys = [datetime.datetime.strptime(date, LME_DATE) for _sid, date, _turns in rows]
            rows = [row for _key, row in sorted(zip(keys, rows), key=lambda kr: kr[0])]
        except ValueError:
            pass  # a date in another form: keep the dataset order
        sessions, text_of, gold, seen = [], {}, [], {}
        for sid, date, turns in rows:
            seen[sid] = seen.get(sid, 0) + 1
            if seen[sid] > 1:
                sid = "%s~%d" % (sid, seen[sid])
            out_turns = []
            for i, t in enumerate(turns):
                tid = "%s#%d" % (sid, i + 1)
                out_turns.append(Turn(tid, t["role"], t["content"]))
                text_of[tid] = t["content"]
                if t.get("has_answer"):
                    gold.append(tid)
            sessions.append(Session(sid, date, tuple(out_turns)))
        qid = x["question_id"]
        if "_abs" in qid:
            gold = []
        q = Question(qid, x["question"], x.get("question_date", ""), str(x["answer"]), tuple(gold),
                     x["question_type"])
        out.append(Instance(qid, tuple(sessions), (q,), text_of))
    return out
