"""The contract between the harness and a memory system.

A donor writes one file, systems/<name>/adapter.py, with one class named Adapter: a subclass of System.
For each test instance (one conversation history), the harness makes a new Adapter in an empty folder.
It calls ingest() once for each session, in time order, and then recall() once for each question.
The system never gets a question before ingest ends, and it never gets an answer or a gold turn.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Turn:
    id: str  # unique in the instance
    speaker: str  # a name (LoCoMo) or "user" / "assistant" (LongMemEval)
    text: str
    extra: dict = dataclasses.field(default_factory=dict)  # for example {"image_caption": "..."}


@dataclasses.dataclass(frozen=True)
class Session:
    id: str  # unique in the instance
    date: str  # the date text of the dataset, unchanged
    turns: tuple


@dataclasses.dataclass(frozen=True)
class Query:
    text: str
    date: str = ""  # the date of the question, when the dataset gives one


def render(session, turn):
    """The plain line of a turn: "[date] speaker: text", plus the image caption when the turn has one.

    Adapters may use it; nothing requires it. Every character of an item counts against the budget.
    """
    line = "[%s] %s: %s" % (session.date, turn.speaker, turn.text)
    caption = turn.extra.get("image_caption")
    return line + " [image: %s]" % caption if caption else line


class System:
    """The base class of an adapter. A subclass implements ingest and recall; close is optional."""

    def __init__(self, workdir, manifest):
        self.workdir = workdir  # empty; keep every file of the store here, because the harness measures it
        self.manifest = manifest  # the parsed system.json
        self.usage = {}  # model use that the system reports, for example {"llm_calls": 3, "llm_tokens": 900}

    def ingest(self, session):
        """Store one Session. The harness gives the sessions of an instance in time order."""
        raise NotImplementedError

    def recall(self, query, budget):
        """The memory context for a Query: a list of text items, best first.

        budget is the token budget of the context. The harness counts the tokens of each item and keeps the
        items in order until the next item does not fit. A system with "budget_aware": false gets one call at
        the largest budget, and the harness cuts that list for each smaller budget.
        """
        raise NotImplementedError

    def close(self):
        """Release files and connections. The harness then measures the bytes in workdir and deletes it."""
