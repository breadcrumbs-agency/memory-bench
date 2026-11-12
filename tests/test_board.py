import unittest

from memorybench import board
from memorybench import run as runner

from test_harness import manifest


def record(name, hits, sha="a" * 64, access="open-source", publication=None, date="2026-11-11T00:00:00Z"):
    """An evidence-track record over len(hits) questions with one gold turn each; hits[i] is 0 or 1 at both budgets."""
    budgets = [256, 1024]
    rows = [{"id": "q%d" % i, "cat": "1", "gold": 1, "hit": [h, h], "tok": [200, 900], "first": 1 if h else None}
            for i, h in enumerate(hits)]
    summary = runner.summarize("evidence", budgets, rows)
    summary.update({"ingest_s": 0.5, "recall_ms_p50": 1.0, "disk_bytes": 2_000_000})
    return {"protocol": "memory-bench/1", "dataset": {"name": "locomo", "sha256": sha}, "track": "evidence",
            "budgets": budgets, "question_set": "b" * 64, "answer_models": None,
            "system": manifest(name, access=access, publication=publication),
            "runner": {"name": "test", "role": "other", "date": date, "platform": "Linux-6", "machine": "x86_64"},
            "summary": summary, "questions": rows}


class TestBoard(unittest.TestCase):
    def test_table_order_and_head_to_head(self):
        strong = record("strong", [1] * 90 + [0] * 10)
        weak = record("weak", [1] * 40 + [0] * 60)
        twin = record("twin", [1] * 40 + [0] * 60)
        close = record("close", [1] * 41 + [0] * 59)
        text = board.render([weak, strong, twin, close])
        rows = [line for line in text.splitlines() if line.startswith("| 1 |") or line.startswith("| 2 |")]
        self.assertIn("| 1 | strong |", rows[0])
        self.assertIn("| 0.900 | 0.900 | 0.900 |", rows[0])
        self.assertIn("| strong |  | +0.490 ▲ (49/51/0) | +0.500 ▲ (50/50/0)", text)
        self.assertIn("| close | -0.490 ▼ (0/51/49) |  | +0.010 = (1/99/0)", text)  # ahead on the mean, not clearly
        self.assertIn("+0.000 = (0/100/0)", text)

    def test_groups_and_newest_run(self):
        old = record("x", [0] * 10, date="2026-01-01T00:00:00Z")
        new = record("x", [1] * 10, date="2026-02-01T00:00:00Z")
        other = record("y", [1] * 10, sha="c" * 64)
        text = board.render([old, new, other])
        self.assertEqual(text.count("## locomo, evidence track"), 2)  # other dataset bytes: another group
        self.assertNotIn("| 0.000 |", text)  # the older run of x is replaced

    def test_licensed_needs_a_grant(self):
        closed = record("closed", [1] * 10, access="licensed")
        granted = record("granted", [1] * 10, access="licensed", publication={
            "granted_by": "Vendor Inc.", "date": "2026-11-11", "evidence": "https://example.org/grant"})
        text = board.render([closed, granted])
        self.assertIn("Withheld (licensed, no publication grant): closed.", text)
        self.assertNotIn("| closed |", text)
        self.assertIn("| granted |", text)
        self.assertIn("| closed |", board.render([closed, granted], private=True))


if __name__ == "__main__":
    unittest.main()
