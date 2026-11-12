"""Floor: no search. recall() gives the newest turns first, as a chat window keeps them."""

from memorybench import System, render


class Adapter(System):
    def __init__(self, workdir, manifest):
        super().__init__(workdir, manifest)
        self.lines = []

    def ingest(self, session):
        self.lines.extend(render(session, t) for t in session.turns)

    def recall(self, query, budget):
        return self.lines[::-1]
