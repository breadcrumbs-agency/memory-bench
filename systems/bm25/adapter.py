"""Floor: SQLite FTS5 BM25, one row for each turn (the search that the Python standard library has).

The query is the distinct words of the question that are not stop words, joined with OR.
"""

import os
import re
import sqlite3

from memorybench import System, render

STOP = set("""a an the and or of to in on at for with by from is are was were be been being do does did
done have has had i you he she it we they me my your his her its our their what when where which who whom
why how that this these those there here as about into over after before during than then so if not no
yes can could would should will shall may might must just also very much many any some more most other""".split())


class Adapter(System):
    def __init__(self, workdir, manifest):
        super().__init__(workdir, manifest)
        self.db = sqlite3.connect(os.path.join(workdir, "bm25.db"))
        self.db.execute("CREATE VIRTUAL TABLE t USING fts5(line UNINDEXED, body)")

    def ingest(self, session):
        rows = []
        for t in session.turns:
            body = "%s: %s" % (t.speaker, t.text)
            if t.extra.get("image_caption"):
                body += " [image: %s]" % t.extra["image_caption"]
            rows.append((render(session, t), body))
        self.db.executemany("INSERT INTO t VALUES (?, ?)", rows)
        self.db.commit()

    def recall(self, query, budget):
        words = list(dict.fromkeys(re.findall(r"\w+", query.text.lower())))
        terms = [w for w in words if w not in STOP] or words
        if not terms:
            return []
        expr = " OR ".join('"%s"' % w for w in terms)
        return [line for (line,) in self.db.execute("SELECT line FROM t WHERE t MATCH ? ORDER BY bm25(t)", (expr,))]

    def close(self):
        self.db.close()
