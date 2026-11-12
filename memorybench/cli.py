"""python -m memorybench fetch | run | board"""

import argparse
import datetime
import json
import os
import sys

from memorybench import board, datasets
from memorybench import run as runner


def _progress(instances, questions):
    print("  %d instances, %d questions" % (instances, questions), file=sys.stderr, flush=True)


def write_record(rec, path):
    """The record as indented JSON, with one compact line for each question (small files, readable diffs)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = json.dumps(dict(rec, questions=[]), indent=1, ensure_ascii=False)
    rows = ",\n".join("  " + json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rec["questions"])
    text = text.replace('"questions": []', '"questions": [\n%s\n ]' % rows if rows else '"questions": []')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="memorybench", description="Compare memory systems head to head.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download a dataset and check its SHA-256")
    f.add_argument("dataset", choices=sorted(datasets.SOURCES))
    f.add_argument("--data", help="a local copy to check instead of a download")

    r = sub.add_parser("run", help="run one system and write one run record")
    r.add_argument("system", help="a system folder, for example systems/bm25")
    r.add_argument("--dataset", required=True, choices=sorted(datasets.SOURCES))
    r.add_argument("--data", help="a local copy of the dataset file (it must have the pinned SHA-256)")
    r.add_argument("--track", default="evidence", choices=runner.TRACKS)
    r.add_argument("--budgets", help="token budgets, comma-separated (default: 256..4096 evidence, 4096 answer)")
    r.add_argument("--limit", type=int, default=0, help="the first N instances only (0 = all)")
    r.add_argument("--runner", default="unnamed", help="who runs it")
    r.add_argument("--role", default="other", choices=runner.ROLES)
    r.add_argument("--reader", help="answer track: the reader model")
    r.add_argument("--judge", help="answer track: the judge model")
    r.add_argument("--api-base", help="answer track: an OpenAI-compatible base URL (key: MEMORYBENCH_API_KEY)")
    r.add_argument("--out", help="the record file (default: results/private/<dataset>-<track>-<system>-<time>.json)")

    b = sub.add_parser("board", help="write the board of a set of run records")
    b.add_argument("paths", nargs="*", help="record files or folders (default: results/public)")
    b.add_argument("--private", action="store_true", help="also read results/private and show withheld runs")
    b.add_argument("--out", help="write the board to this file instead of stdout")

    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        print(datasets.fetch(a.dataset, a.data))
        return 0
    if a.cmd == "run":
        budgets = [int(x) for x in a.budgets.split(",")] if a.budgets else None
        rec = runner.run(a.system, a.dataset, a.data, a.track, budgets, a.limit, a.runner, a.role,
                         a.reader, a.judge, a.api_base, _progress)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = a.out or os.path.join("results", "private", "%s-%s-%s-%s.json" % (
            a.dataset, a.track, rec["system"]["name"], stamp))
        write_record(rec, out)
        print(json.dumps({"out": out, "summary": {k: v for k, v in rec["summary"].items() if k != "by_category"}}))
        return 0
    paths = a.paths or ["results/public"]
    if a.private and os.path.isdir("results/private"):
        paths.append("results/private")
    text = board.render(board.load([p for p in paths if os.path.exists(p)]), a.private)
    head = ("# Board\n\nWritten by `python -m memorybench board`. Do not edit it by hand: run records in "
            "`results/public/` make it. README.md says how the numbers are made.\n\n")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(head + text + "\n")
    else:
        print(head + text)
    return 0
