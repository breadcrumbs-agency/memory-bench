import http.server
import json
import os
import threading
import unittest
from unittest import mock

from memorybench import answer
from memorybench import run as runner

from test_harness import ROOT, Fixture


class FakeModels(http.server.BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint: the reader repeats "80 C" when the context holds it; the judge
    says yes when the gold answer is in the answer."""

    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeModels.seen.append((self.headers.get("Authorization"), body))
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        if system == answer.READER_SYSTEM:
            reply = "80 C" if "80 C" in user else "I do not know."
        else:
            gold = user.split("Gold answer: ")[1].split("\n")[0]
            said = user.split("Answer to grade: ")[1].split("\n")[0]
            reply = "yes" if gold in said else "no"
        data = json.dumps({"choices": [{"message": {"content": reply}}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class TestAnswer(Fixture):
    def setUp(self):
        super().setUp()
        FakeModels.seen = []
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeModels)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = "http://127.0.0.1:%d/v1" % self.server.server_address[1]

    def test_answer_track(self):
        with mock.patch.dict(os.environ, {"MEMORYBENCH_API_KEY": "sk-test-secret"}):
            rec = runner.run(os.path.join(ROOT, "systems", "bm25"), "locomo", self.files["locomo"], track="answer",
                             reader="reader-x", judge="judge-y", api_base=self.base)
        rows = {r["id"]: r for r in rec["questions"]}
        self.assertEqual(sorted(rows), ["conv-t/0", "conv-t/1", "conv-t/3"])  # category 5 is out
        self.assertEqual(rows["conv-t/0"]["ok"], [1])
        self.assertEqual(rows["conv-t/0"]["answer"], ["80 C"])
        self.assertEqual(rec["summary"]["accuracy"], [round(1 / 3, 4)])
        m = rec["answer_models"]
        self.assertEqual((m["reader"], m["judge"], m["api_host"]), ("reader-x", "judge-y", "127.0.0.1"))
        self.assertEqual(m["prompts_sha256"], answer.prompts_sha256())
        self.assertEqual(m["tokens"], {"reader_in": 30, "reader_out": 6, "judge_in": 30, "judge_out": 6})
        self.assertEqual({auth for auth, _ in FakeModels.seen}, {"Bearer sk-test-secret"})
        self.assertEqual({body["temperature"] for _, body in FakeModels.seen}, {0})
        self.assertNotIn("sk-test-secret", json.dumps(rec))  # the key never enters a record

    def test_answer_track_needs_models(self):
        with self.assertRaises(ValueError):
            runner.run(os.path.join(ROOT, "systems", "bm25"), "locomo", self.files["locomo"], track="answer")


if __name__ == "__main__":
    unittest.main()
