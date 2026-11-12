import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from memorybench import board, cli, datasets, meter
from memorybench import run as runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOCOMO = [{
    "sample_id": "conv-t",
    "conversation": {
        "speaker_a": "Ann", "speaker_b": "Bob",
        "session_1_date_time": "1:00 pm on 1 May, 2023",
        "session_1": [
            {"speaker": "Ann", "dia_id": "D1:1", "text": "I brew green tea at 80 C every morning."},
            {"speaker": "Bob", "dia_id": "D1:2", "text": "My bike chain broke on the hill."},
        ],
        "session_10_date_time": "9:00 am on 1 July, 2023",
        "session_10": [{"speaker": "Ann", "dia_id": "D10:1", "text": "The kettle is new."}],
        "session_2_date_time": "2:00 pm on 8 May, 2023",
        "session_2": [
            {"speaker": "Ann", "dia_id": "D2:1", "text": "The garden tomatoes are red now.",
             "blip_caption": "a photo of tomatoes"},
            {"speaker": "Bob", "dia_id": "D2:2", "text": "I fixed the bike chain with a new link."},
        ],
    },
    "qa": [
        {"question": "What temperature does Ann brew green tea at?", "answer": "80 C", "evidence": ["D1:1"],
         "category": 1},
        {"question": "How did Bob fix the bike chain?", "answer": "with a new link", "evidence": ["D1:2; D2:2"],
         "category": 4},
        {"question": "What color are the garden tomatoes?", "answer": "red", "adversarial_answer": "green",
         "evidence": ["D2:1"], "category": 5},
        {"question": "Who is the queen?", "answer": "nobody", "evidence": [], "category": 2},
    ],
}]

LME = [
    {"question_id": "q1", "question_type": "single-session-user", "question": "What tea do I like best?",
     "answer": "green tea", "question_date": "2023/05/30 (Tue) 23:40",
     "haystack_session_ids": ["s_b", "answer_s_a", "s_b"],
     "haystack_dates": ["2023/05/21 (Sun) 10:00", "2023/05/20 (Sat) 02:21", "2023/05/22 (Mon) 09:00"],
     "haystack_sessions": [
         [{"role": "user", "content": "My bike chain broke."}, {"role": "assistant", "content": "Try a new link."}],
         [{"role": "user", "content": "I like green tea best.", "has_answer": True},
          {"role": "assistant", "content": "Green tea is a good choice."}],
         [{"role": "user", "content": "The garden needs water."}],
     ]},
    {"question_id": "q2_abs", "question_type": "single-session-user", "question": "What is my hamster called?",
     "answer": "You did not mention this information.", "question_date": "2023/05/30 (Tue) 23:40",
     "haystack_session_ids": ["answer_s_c"], "haystack_dates": ["2023/05/20 (Sat) 02:21"],
     "haystack_sessions": [[{"role": "user", "content": "My cat is called Luna.", "has_answer": True}]]},
]

TEST_ADAPTER = '''
import json
from memorybench import System, render

class Adapter(System):
    def __init__(self, workdir, manifest):
        super().__init__(workdir, manifest)
        self.mode, self.log, self.lines = manifest["mode"], manifest.get("log"), []
        open(workdir + "/store.bin", "wb").write(b"x" * 1000)

    def _log(self, *args):
        if self.log:
            with open(self.log, "a") as f:
                f.write(json.dumps(args) + "\\n")

    def ingest(self, session):
        self._log("ingest", session.id, [(t.id, t.speaker, t.text, t.extra) for t in session.turns])
        self.lines += [render(session, t) for t in session.turns]

    def recall(self, query, budget):
        self._log("recall", query.text, query.date, budget)
        self.usage["llm_calls"] = self.usage.get("llm_calls", 0) + 1
        if self.mode == "flood":
            return ["x" * 40] * 100
        if self.mode == "ids":
            return ["D1:1 D1:2 D2:2"]
        if self.mode == "bad":
            return [{"text": "not a string"}]
        return self.lines
'''


def lines(path):
    with open(path) as f:
        return [json.loads(x) for x in f]


def manifest(name, **kw):
    m = {"name": name, "version": "1", "license": "MIT", "access": "open-source", "class": "local",
         "homepage": "https://example.org", "models": [], "budget_aware": False, "tuned_on": [],
         "publication": None}
    m.update(kw)
    return m


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.files = {}
        for name, raw in (("locomo", LOCOMO), ("longmemeval-s", LME)):
            path = os.path.join(self.dir, name + ".json")
            with open(path, "w") as f:
                json.dump(raw, f)
            self.files[name] = path
            sha = datasets.sha256_file(path)
            p = mock.patch.dict(datasets.SOURCES[name], {"sha256": sha})
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def system(self, mode, **kw):
        folder = os.path.join(self.dir, "sys-" + mode)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "adapter.py"), "w") as f:
            f.write(TEST_ADAPTER)
        with open(os.path.join(folder, "system.json"), "w") as f:
            json.dump(manifest("t-" + mode, mode=mode, **kw), f)
        return folder


class TestMeter(unittest.TestCase):
    def test_tokens_and_cut(self):
        self.assertEqual([meter.tokens(s) for s in ("", "a", "abcd", "abcde")], [0, 1, 1, 2])
        shown, used = meter.cut(["a" * 40, "b" * 40, "d" * 40, "c" * 4], 21)
        self.assertEqual((len(shown), used), (2, 20))  # the 4th item would fit, but the cut stops at the 1st miss
        with self.assertRaises(TypeError):
            meter.cut([b"bytes"], 10)

    def test_norm(self):
        self.assertEqual(meter.norm("  Green\n TEA\t80 C "), "green tea 80 c")


class TestDatasets(Fixture):
    def test_pinned_hash_is_enforced(self):
        with open(self.files["locomo"], "a") as f:
            f.write(" ")
        with self.assertRaises(ValueError):
            datasets.fetch("locomo", self.files["locomo"])

    def test_locomo(self):
        (inst,) = datasets.instances("locomo", datasets.fetch("locomo", self.files["locomo"]))
        self.assertEqual([s.id for s in inst.sessions], ["session_1", "session_2", "session_10"])
        q = inst.questions
        self.assertEqual(q[1].gold, ("D1:2", "D2:2"))
        self.assertEqual([x.answer for x in q], ["80 C", "with a new link", None, "nobody"])  # category 5 is out
        self.assertEqual(inst.sessions[1].turns[0].extra, {"image_caption": "a photo of tomatoes"})

    def test_longmemeval_sorted_by_date_and_unique_ids(self):
        a, b = datasets.instances("longmemeval-s", self.files["longmemeval-s"])
        self.assertEqual([s.id for s in a.sessions], ["answer_s_a", "s_b", "s_b~2"])
        self.assertEqual(a.questions[0].gold, ("answer_s_a#1",))
        self.assertEqual(len(a.text_of), 5)
        self.assertEqual(b.questions[0].gold, ())  # abstention: no evidence
        self.assertTrue(runner.eligible(b.questions[0], b, "answer"))


class TestRun(Fixture):
    def test_stream_is_blind_and_ordered(self):
        log = os.path.join(self.dir, "calls.jsonl")
        runner.run(self.system("all", log=log), "locomo", self.files["locomo"])
        calls = lines(log)
        self.assertEqual([c[0] for c in calls], ["ingest"] * 3 + ["recall"] * 3)  # every ingest first
        self.assertEqual([c[1] for c in calls[:3]], ["session_1", "session_2", "session_10"])
        # one call per question, at the largest budget; the query is the question text and nothing else
        self.assertEqual([c[1:] for c in calls[3:]], [[q["question"], "", 4096] for q in LOCOMO[0]["qa"][:3]])

    def test_evidence_scores(self):
        rec = runner.run(self.system("all"), "locomo", self.files["locomo"])
        rows = {r["id"]: r for r in rec["questions"]}
        self.assertEqual(sorted(rows), ["conv-t/0", "conv-t/1", "conv-t/2"])  # no gold: out
        self.assertEqual(rows["conv-t/1"]["hit"], [2] * 5)
        self.assertEqual(rows["conv-t/1"]["first"], 2)
        self.assertEqual(rec["summary"]["recall"], [1.0] * 5)
        self.assertEqual(rec["summary"]["disk_bytes"], 1000)
        self.assertEqual(rec["summary"]["usage"], {"llm_calls": 3})
        self.assertEqual(rec["summary"]["turns"], 5)

    def test_budget_is_enforced(self):
        rec = runner.run(self.system("flood"), "locomo", self.files["locomo"], budgets=[256, 1024])
        self.assertEqual(rec["questions"][0]["tok"], [250, 1000])
        self.assertEqual(rec["summary"]["recall"], [0.0, 0.0])

    def test_ids_alone_do_not_count(self):
        rec = runner.run(self.system("ids"), "locomo", self.files["locomo"])
        self.assertEqual(rec["summary"]["recall"], [0.0] * 5)

    def test_items_must_be_text(self):
        with self.assertRaises(TypeError):
            runner.run(self.system("bad"), "locomo", self.files["locomo"])

    def test_budget_aware_gets_one_call_per_budget(self):
        log = os.path.join(self.dir, "aware.jsonl")
        runner.run(self.system("all", log=log, budget_aware=True), "locomo", self.files["locomo"], budgets=[512, 256])
        budgets = [c[3] for c in lines(log) if c[0] == "recall"]
        self.assertEqual(budgets, [256, 512] * 3)

    def test_shipped_floors(self):
        for name in ("recent", "bm25"):
            rec = runner.run(os.path.join(ROOT, "systems", name), "longmemeval-s", self.files["longmemeval-s"])
            self.assertEqual(rec["summary"]["questions"], 1)
            self.assertEqual(rec["summary"]["recall"], [1.0] * 5, name)
            self.assertTrue(rec["system"]["adapter_sha256"])

    def test_manifest_rules(self):
        runner.check_manifest(manifest("x"))
        for bad in ({"access": "closed"}, {"class": "cloud"}, {"budget_aware": "no"},
                    {"access": "licensed", "publication": {"granted_by": "Vendor"}}):
            with self.assertRaises(ValueError):
                runner.check_manifest(manifest("x", **bad))
        runner.check_manifest(manifest("x", access="licensed", publication={
            "granted_by": "Vendor Inc.", "date": "2026-11-11", "evidence": "https://example.org/grant"}))
        for name in os.listdir(os.path.join(ROOT, "systems")):
            with open(os.path.join(ROOT, "systems", name, "system.json")) as f:
                runner.check_manifest(json.load(f), name)

    def test_cli_run_then_board(self):
        out = os.path.join(self.dir, "public", "rec.json")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cli.main(["run", os.path.join(ROOT, "systems", "bm25"), "--dataset", "locomo", "--data",
                      self.files["locomo"], "--runner", "test", "--out", out])
        with open(out) as f:
            rec = json.load(f)
        self.assertNotIn(self.dir, json.dumps(rec))  # no local path in a record
        self.assertEqual(rec["question_set"], hashlib.sha256(b"conv-t/0\nconv-t/1\nconv-t/2").hexdigest())
        md = os.path.join(self.dir, "BOARD.md")
        cli.main(["board", os.path.dirname(out), "--out", md])
        with open(md) as f:
            self.assertIn("| 1 | bm25 |", f.read())


if __name__ == "__main__":
    unittest.main()
