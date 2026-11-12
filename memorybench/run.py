"""Run one system over one dataset on one track, and build the run record."""

import datetime
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse

from memorybench import PROTOCOL, __version__, answer, datasets, meter
from memorybench.system import Query, System

TRACKS = ("evidence", "answer")
BUDGETS = {"evidence": (256, 512, 1024, 2048, 4096), "answer": (4096,)}
ROLES = ("maintainer", "donor", "other")
MANIFEST = {"name": str, "version": str, "license": str, "access": str, "class": str, "homepage": str,
            "models": list, "budget_aware": bool, "tuned_on": list}
ACCESS = ("open-source", "licensed")
CLASSES = ("local", "local-model", "remote")
GRANT = ("granted_by", "date", "evidence")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_manifest(m, where="system.json"):
    for key, kind in MANIFEST.items():
        if not isinstance(m.get(key), kind):
            raise ValueError("%s: %r must be a %s" % (where, key, kind.__name__))
    if m["access"] not in ACCESS:
        raise ValueError("%s: access must be one of %s" % (where, ", ".join(ACCESS)))
    if m["class"] not in CLASSES:
        raise ValueError("%s: class must be one of %s" % (where, ", ".join(CLASSES)))
    grant = m.get("publication")
    if grant is not None and not (isinstance(grant, dict) and all(isinstance(grant.get(k), str) and grant[k]
                                                                  for k in GRANT)):
        raise ValueError("%s: publication must be null or an object with %s" % (where, ", ".join(GRANT)))


def load_system(folder):
    """(manifest, Adapter class, SHA-256 of adapter.py) of a system folder."""
    with open(os.path.join(folder, "system.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    check_manifest(manifest, os.path.join(folder, "system.json"))
    path = os.path.join(folder, "adapter.py")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    spec = importlib.util.spec_from_file_location("memorybench_system_" + manifest["name"].replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = getattr(module, "Adapter", None)
    if not (isinstance(adapter, type) and issubclass(adapter, System)):
        raise TypeError("%s: Adapter must be a subclass of memorybench.System" % path)
    return manifest, adapter, digest


def tree_bytes(path):
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            p = os.path.join(base, name)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


def gold_texts(q, inst):
    """The distinct, non-empty matching forms of the gold turns of a question."""
    return list(dict.fromkeys(g for g in (meter.norm(inst.text_of[t]) for t in q.gold) if g))


def eligible(q, inst, track):
    return bool(gold_texts(q, inst)) if track == "evidence" else q.answer is not None


def _recall(system, query, budget, times):
    t0 = time.perf_counter()
    items = list(system.recall(query, budget))
    times.append((time.perf_counter() - t0) * 1000)
    return items


def score_evidence(system, query, q, inst, budgets, aware, times):
    """One question on the evidence track: the gold turns whose text is in the context at each budget."""
    gold = gold_texts(q, inst)
    lists = {b: _recall(system, query, b, times) for b in (budgets if aware else budgets[-1:])}
    hit, tok, first = [], [], None
    for b in budgets:
        shown, used = meter.cut(lists[b] if b in lists else lists[budgets[-1]], b)
        found = set()
        for rank, item in enumerate(shown, 1):
            text = meter.norm(item)
            new = {i for i, g in enumerate(gold) if g in text}
            if new and first is None and b == budgets[-1]:
                first = rank
            found |= new
        hit.append(len(found))
        tok.append(used)
    return {"id": q.id, "cat": q.category, "gold": len(gold), "hit": hit, "tok": tok, "first": first}


def score_answer(system, query, q, budgets, aware, client, models, spent, times):
    """One question on the answer track: the reader answers from the context at each budget, the judge grades."""
    row = {"id": q.id, "cat": q.category, "ok": [], "tok": [], "answer": []}
    base = None if aware else _recall(system, query, budgets[-1], times)
    for b in budgets:
        shown, used = meter.cut(_recall(system, query, b, times) if aware else base, b)
        text, ok, use = answer.grade(client, models["reader"], models["judge"], q, "\n".join(shown))
        row["ok"].append(int(ok))
        row["tok"].append(used)
        row["answer"].append(text)
        for i, n in enumerate(use):
            spent[i] += n
    return row


def _pct(values, p):
    if not values:
        return None
    v = sorted(values)
    return round(v[int(p * (len(v) - 1))], 3)


def _commit():
    """The harness commit, and whether its code differs from that commit."""
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "memorybench", "systems"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": head, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _load():
    try:
        return round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        return None


def summarize(track, budgets, rows):
    n = len(rows)
    out = {"questions": n}
    if not n:
        return out
    cats = sorted({r["cat"] for r in rows}, key=lambda c: (len(c), c))
    if track == "evidence":
        value = lambda r, i: r["hit"][i] / r["gold"]  # noqa: E731
        out["recall"] = [round(statistics.fmean(value(r, i) for r in rows), 4) for i in range(len(budgets))]
        out["mrr"] = round(statistics.fmean(1 / r["first"] if r["first"] else 0.0 for r in rows), 4)
    else:
        value = lambda r, i: r["ok"][i]  # noqa: E731
        out["accuracy"] = [round(statistics.fmean(value(r, i) for r in rows), 4) for i in range(len(budgets))]
    out["mean"] = round(statistics.fmean(value(r, i) for r in rows for i in range(len(budgets))), 4)
    out["tokens"] = [round(statistics.fmean(r["tok"][i] for r in rows), 1) for i in range(len(budgets))]
    out["by_category"] = {c: [round(statistics.fmean(value(r, i) for r in rows if r["cat"] == c), 4)
                              for i in range(len(budgets))] for c in cats}
    out["questions_by_category"] = {c: sum(r["cat"] == c for r in rows) for c in cats}
    return out


def run(folder, dataset, data=None, track="evidence", budgets=None, limit=0, runner="unnamed", role="other",
        reader=None, judge=None, api_base=None, progress=None):
    """The run record of one system on one dataset and track."""
    if track not in TRACKS:
        raise ValueError("track must be one of %s" % ", ".join(TRACKS))
    if role not in ROLES:
        raise ValueError("role must be one of %s" % ", ".join(ROLES))
    manifest, adapter, digest = load_system(folder)
    budgets = tuple(sorted(set(budgets or BUDGETS[track])))
    path = datasets.fetch(dataset, data)
    client, models, spent = None, None, [0, 0, 0, 0]
    if track == "answer":
        if not (reader and judge and api_base):
            raise ValueError("the answer track needs a reader model, a judge model and an API base URL")
        client = answer.Client(api_base, os.environ.get("MEMORYBENCH_API_KEY"))
        models = {"reader": reader, "judge": judge, "api_host": urllib.parse.urlsplit(api_base).hostname,
                  "prompts_sha256": answer.prompts_sha256()}
    rows, times, usage = [], [], {}
    ingest_s, disk, n_inst, n_sess, n_turns = 0.0, 0, 0, 0, 0
    load_before, started, t_start = _load(), datetime.datetime.now(datetime.timezone.utc), time.perf_counter()
    for k, inst in enumerate(datasets.instances(dataset, path)):
        if limit and k >= limit:
            break
        asked = [q for q in inst.questions if eligible(q, inst, track)]
        with tempfile.TemporaryDirectory(prefix="memory-bench-") as work:
            system = adapter(work, manifest)
            try:
                for s in inst.sessions:
                    t0 = time.perf_counter()
                    system.ingest(s)
                    ingest_s += time.perf_counter() - t0
                    n_sess += 1
                    n_turns += len(s.turns)
                for q in asked:
                    query = Query(q.text, q.date)
                    if track == "evidence":
                        rows.append(score_evidence(system, query, q, inst, budgets, manifest["budget_aware"], times))
                    else:
                        rows.append(score_answer(system, query, q, budgets, manifest["budget_aware"], client,
                                                 models, spent, times))
            finally:
                system.close()
            disk += tree_bytes(work)
        for key, v in (system.usage or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                usage[key] = usage.get(key, 0) + v
        n_inst += 1
        if progress:
            progress(n_inst, len(rows))
    src = datasets.SOURCES[dataset]
    summary = summarize(track, budgets, rows)
    summary.update({"instances": n_inst, "sessions": n_sess, "turns": n_turns, "ingest_s": round(ingest_s, 3),
                    "recall_ms_p50": _pct(times, 0.5), "recall_ms_p95": _pct(times, 0.95), "disk_bytes": disk,
                    "usage": usage, "wall_s": round(time.perf_counter() - t_start, 1)})
    if models:
        models["tokens"] = {"reader_in": spent[0], "reader_out": spent[1], "judge_in": spent[2], "judge_out": spent[3]}
    return {
        "protocol": PROTOCOL,
        "harness": dict(version=__version__, **_commit()),
        "dataset": {"name": dataset, "sha256": src["sha256"], "source": src["url"], "license": src["license"]},
        "track": track,
        "budgets": list(budgets),
        "meter": meter.RULE,
        "limit": limit,
        "question_set": hashlib.sha256("\n".join(sorted(r["id"] for r in rows)).encode("utf-8")).hexdigest(),
        "system": dict(manifest, adapter_sha256=digest),
        "answer_models": models,
        "runner": {"name": runner, "role": role, "date": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "platform": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count(),
                   "python": sys.version.split()[0], "load_before": load_before, "load_after": _load()},
        "summary": summary,
        "questions": rows,
    }
