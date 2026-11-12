"""The board: one table for each group of comparable runs, and a head-to-head matrix for each group.

Two runs share a group only when the protocol, the dataset bytes, the track, the budgets and the question
set are the same, and on the answer track also the reader, the judge and the prompts. A run of a licensed
system shows numbers only when its manifest holds a publication grant, or under --private.
"""

import glob
import json
import math
import os
import statistics


def load(paths):
    runs = []
    for p in paths:
        files = sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True)) if os.path.isdir(p) else [p]
        for name in files:
            with open(name, encoding="utf-8") as f:
                r = json.load(f)
            if str(r.get("protocol", "")).startswith("memory-bench/") and r.get("questions"):
                runs.append(r)
    return runs


def group_key(r):
    key = (r["protocol"], r["dataset"]["name"], r["dataset"]["sha256"], r["track"], tuple(r["budgets"]),
           r["question_set"])
    if r["track"] == "answer":
        a = r["answer_models"]
        key += (a["reader"], a["judge"], a["prompts_sha256"])
    return key


def withheld(r, private):
    s = r["system"]
    return s["access"] == "licensed" and not s.get("publication") and not private


def per_question(r):
    """question id -> the value of the question averaged over the budgets (recall, or 1 for a right answer)."""
    if r["track"] == "evidence":
        return {q["id"]: statistics.fmean(h / q["gold"] for h in q["hit"]) for q in r["questions"]}
    return {q["id"]: statistics.fmean(q["ok"]) for q in r["questions"]}


def paired(a, b):
    """(mean of a - b, low, high of its 95% interval, a wins, ties, b wins) over the questions of both."""
    d = [a[k] - b[k] for k in sorted(a) if k in b]
    m = statistics.fmean(d)
    half = 1.96 * statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else math.inf
    return m, m - half, m + half, sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)


def _f(x, digits=3):
    return "" if x is None else "%.*f" % (digits, x)


def _runner(r):
    u = r["runner"]
    return "%s (%s), %s %s" % (u["name"], u["role"], u["platform"].split("-")[0], u["machine"])


def _latest(rs):
    out = {}
    for r in rs:
        k = (r["system"]["name"], r["system"]["version"])
        if k not in out or r["runner"]["date"] > out[k]["runner"]["date"]:
            out[k] = r
    return list(out.values())


def _table(rs, track, budgets):
    metric = "R" if track == "evidence" else "Acc"
    head = ["#", "System", "Version", "Access", "Class"] + ["%s@%d" % (metric, b) for b in budgets] + ["Mean"]
    head += (["MRR"] if track == "evidence" else []) + ["Ingest s", "Recall p50 ms", "Disk MB", "Runner"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, r in enumerate(rs, 1):
        s, m = r["system"], r["summary"]
        values = m["recall"] if track == "evidence" else m["accuracy"]
        cells = [str(i), s["name"], s["version"], s["access"], s["class"]] + [_f(v) for v in values] + [_f(m["mean"])]
        cells += ([_f(m["mrr"])] if track == "evidence" else [])
        cells += [_f(m["ingest_s"], 2), _f(m["recall_ms_p50"], 2), _f(m["disk_bytes"] / 1e6, 1), _runner(r)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _matrix(rs):
    names = [r["system"]["name"] for r in rs]
    vals = [per_question(r) for r in rs]
    lines = ["| row − column | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for i, a in enumerate(vals):
        cells = []
        for j, b in enumerate(vals):
            if i == j:
                cells.append("")
                continue
            m, lo, hi, wins, ties, losses = paired(a, b)
            mark = "▲" if lo > 0 else "▼" if hi < 0 else "="
            cells.append("%+.3f %s (%d/%d/%d)" % (m, mark, wins, ties, losses))
        lines.append("| %s | %s |" % (names[i], " | ".join(cells)))
    return lines


def render(runs, private=False):
    groups = {}
    for r in runs:
        groups.setdefault(group_key(r), []).append(r)
    out = []
    for key in sorted(groups, key=lambda k: (k[1], k[3], k[4])):
        rs = _latest(groups[key])
        shown = sorted((r for r in rs if not withheld(r, private)), key=lambda r: -r["summary"]["mean"])
        hidden = sorted(r["system"]["name"] for r in rs if withheld(r, private))
        r0 = rs[0]
        track, budgets = r0["track"], r0["budgets"]
        out += ["## %s, %s track" % (r0["dataset"]["name"], track), ""]
        info = "Questions: %d. Budgets (tokens): %s. Dataset SHA-256: %s…. Question set: %s…." % (
            len(r0["questions"]), ", ".join(map(str, budgets)), r0["dataset"]["sha256"][:12], r0["question_set"][:12])
        if track == "answer":
            a = r0["answer_models"]
            info += " Reader: %s. Judge: %s. Prompts: %s…." % (a["reader"], a["judge"], a["prompts_sha256"][:12])
        out += [info, ""] + _table(shown, track, budgets) + [""]
        if len(shown) > 1:
            out += ["Head to head: the mean of (row − column) per question, over all budgets. ▲ the row is ahead and "
                    "▼ behind (the 95% interval excludes 0); = no clear difference. In brackets: questions won, "
                    "tied and lost by the row.", ""] + _matrix(shown) + [""]
        if len({(r["runner"]["platform"], r["runner"]["machine"]) for r in shown}) > 1:
            out += ["Time and disk come from different machines in this group: compare them only between rows "
                    "with the same runner machine.", ""]
        if hidden:
            out += ["Withheld (licensed, no publication grant): %s." % ", ".join(hidden), ""]
    return "\n".join(out)
