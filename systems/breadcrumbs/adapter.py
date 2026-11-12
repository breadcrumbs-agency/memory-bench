"""breadcrumbs: a content-addressed store and an FTS5 index, searched with dialogue-neighbor smoothing.

Each turn becomes one crumb (put_turn), linked to the turn before it when the speaker changes. recall() is
Index.search with the shipped weight of the "neighbor" dial (0.3), and it gives the hit turns as text lines.
"""

import importlib.metadata
import os

from breadcrumbs.index import Index
from breadcrumbs.ingest import put_turn
from breadcrumbs.store import Store

from memorybench import System, render

NEIGHBOR = 0.3  # the default of the "neighbor" dial in breadcrumbs 0.4.1 (breadcrumbs/dials.py)
HITS = 400


class Adapter(System):
    def __init__(self, workdir, manifest):
        super().__init__(workdir, manifest)
        have = importlib.metadata.version("breadcrumbs")
        if have != manifest["version"]:
            raise RuntimeError("breadcrumbs %s is installed, but system.json pins %s" % (have, manifest["version"]))
        self.store = Store(os.path.join(workdir, "store"))
        self.index = Index(os.path.join(workdir, "index.db"))
        self.line = {}

    def ingest(self, session):
        prev = None
        for t in session.turns:
            body = "%s: %s" % (t.speaker, t.text)
            if t.extra.get("image_caption"):
                body += " [image: %s]" % t.extra["image_caption"]
            cid = put_turn(self.store, self.index, t.speaker, body, session.id, ts=0, prev=prev)
            self.line[cid] = render(session, t)
            prev = cid
        self.store.flush()
        self.index.commit()

    def recall(self, query, budget):
        return [self.line[c] for c, _ref, _score in self.index.search(query.text, HITS, nb=NEIGHBOR) if c in self.line]

    def close(self):
        self.index.close()
        self.store.close()
