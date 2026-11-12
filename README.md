# memory-bench

One stream, one meter, one board for agent memory systems.

Open-source systems and licensed systems run on the same data, under the same token budget, with the same
scoring. The board then puts each pair of systems head to head, question by question.

- The board: [BOARD.md](BOARD.md)
- The contract for a system: [memorybench/system.py](memorybench/system.py)
- Python 3.11 or later. The harness uses only the standard library. Each system brings its own dependencies.

## Quick start

```
git clone https://github.com/breadcrumbs-agency/memory-bench && cd memory-bench
python -m memorybench fetch locomo                        # downloads 2.8 MB and checks its SHA-256
python -m memorybench run systems/bm25 --dataset locomo   # writes results/private/locomo-evidence-bm25-<time>.json
python -m memorybench board --private                     # results/public and results/private, head to head
```

## What the harness does

For each test instance (one long conversation history), the harness:

1. makes a new system in an empty folder,
2. gives it the sessions one at a time, in time order,
3. asks the questions only after the last session,
4. counts the tokens of the context that the system gives back, and cuts the context at the budget,
5. scores the context, measures the folder, and deletes the folder.

The system never gets an answer or the list of evidence turns.

### Datasets

| Name | What | Instances | Questions: evidence / answer | Source, pinned | License |
|---|---|---|---|---|---|
| `locomo` | long conversations between two people: 272 sessions, 5,882 turns | 10 | 1,981 / 1,540 | [snap-research/locomo](https://github.com/snap-research/locomo) at `cbfbc1d` | CC BY-NC 4.0 |
| `longmemeval-s` | chat histories of a user and an assistant, about 490,000 characters each: 23,867 sessions, 246,750 turns | 500 | 470 / 500 | [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) at `98d7416` | MIT |

The repository holds no dataset text. `fetch` downloads each file at the pinned revision and refuses a file
with another SHA-256. `--data PATH` uses a local copy, with the same check. The LongMemEval-S sessions are
sorted by date, because the dataset order is not always the date order.

### Two tracks

- **evidence** (no model; each run gives the same scores). Each question has gold turns: the turns that hold
  the answer. A gold turn counts when its full text is in the context (lower case, spaces made equal). The
  score is the share of gold turns found at each budget: 256, 512, 1,024, 2,048 and 4,096 tokens. "Mean" is
  the mean over the budgets. MRR is 1 / the rank of the first item that holds a gold turn, inside the largest
  budget.
- **answer** (two models). A reader model answers from the context at 4,096 tokens. A judge model compares
  the answer with the gold answer. The score is the share of right answers. Both models run at temperature 0
  through an OpenAI-compatible endpoint. The record keeps the two model names and the SHA-256 of the prompts.
  The API key comes from `MEMORYBENCH_API_KEY` and never goes into a record.

```
MEMORYBENCH_API_KEY=... python -m memorybench run systems/bm25 --dataset locomo --track answer \
    --reader <model> --judge <model> --api-base https://<host>/v1
```

The evidence track measures verbatim recall. A system that rewrites memories into facts cannot show a gold
turn word for word, so its evidence score is low by design. For such a system, read the answer track.

### Costs in each run record

Ingest seconds, recall latency (p50 and p95), the bytes in the system's folder (summed over the instances),
the tokens of each context, the models that the system calls (from its manifest), and the model use that the
system reports. Each record also keeps the runner's machine, Python version, load average and the harness
commit.

## The rules: apples to apples

1. **Same bytes.** Every run reads the pinned dataset file.
2. **Same stream.** Each system gets the same sessions, in the same order, with the same dates.
3. **No peeking.** No system gets an answer or a gold turn. A system that was tuned on a test dataset says so
   in `tuned_on`.
4. **Same meter.** The harness counts the tokens, ceil(characters / 4), and cuts the context. No system can
   use more than the budget. Ids, headers and formats count as text.
5. **Same scoring.** The evidence track has no model. The answer track has one reader, one judge and one
   prompt set for all systems.
6. **Everything declared.** The version, the license, each model that the system calls, and where it runs.
   The adapter code is public, also for a licensed system.
7. **Like with like.** The board compares two runs only when the protocol, the dataset bytes, the track, the
   budgets and the question set are the same. On the answer track, the reader, the judge and the prompts
   must also be the same.
8. **Head to head.** The board compares two systems question by question. It calls a system ahead only
   when the 95% interval of the mean difference does not include zero.

## Donate a system

A donor gives a system to the board: an adapter, and what the run needs. Three files:

```
systems/<name>/system.json        the manifest
systems/<name>/adapter.py         class Adapter(memorybench.System): ingest, recall, close
systems/<name>/requirements.txt   the pinned dependencies, if any
```

The smallest adapter (it is `systems/recent`):

```python
from memorybench import System, render

class Adapter(System):
    def __init__(self, workdir, manifest):
        super().__init__(workdir, manifest)
        self.lines = []

    def ingest(self, session):
        self.lines.extend(render(session, t) for t in session.turns)

    def recall(self, query, budget):
        return self.lines[::-1]  # best first; the harness cuts at the budget
```

| Manifest field | Meaning |
|---|---|
| `name`, `version` | the system, and the exact version that the adapter pins |
| `license` | an SPDX id, or `LicenseRef-<name>` for a closed license |
| `access` | `open-source` (an OSI license) or `licensed` (closed, source-available or hosted) |
| `class` | `local` (no model), `local-model` (models on the runner's machine) or `remote` (a service) |
| `homepage` | where the system lives |
| `models` | each model that the system calls, for example `{"role": "embedder", "id": "...", "where": "local"}` |
| `budget_aware` | `true`: one recall call for each budget; `false`: one call at the largest budget |
| `tuned_on` | each test dataset that tuned a setting of the system, and the setting |
| `publication` | `null`, or the grant that permits publication (licensed systems) |

Then run `python -m memorybench run systems/<name> --dataset locomo` and open a pull request.

### Open-source donors

The pull request holds the three files. The maintainers run the system at the pinned version and commit the
run record to `results/public/`. Anyone can run it again with the same command.

### Licensed donors: vendors and licensees

A licensed system is closed, source-available or hosted. Its donor can be the vendor or a licensee.

- The adapter is public, like every adapter. Keys, binaries and other access never go into the repository.
- Some licenses forbid published benchmarks without the vendor's consent. So the board shows the numbers of
  a licensed system only when its `publication` field holds a grant:
  `{"granted_by": "<rights holder>", "date": "YYYY-MM-DD", "evidence": "<link or document id>"}`.
  Without a grant, the board lists the system as withheld.
- A licensee without a grant can still compare in private. Runs go to `results/private/` (git ignores it),
  and `python -m memorybench board --private` puts them head to head with each public run.
- For a run by the maintainers, the donor gives access for the run: a key with quota, or a binary or a
  container. Otherwise the donor runs it, with `--role donor` in the record.

## Run records

One JSON file per run. `results/public/` holds the records of the board. A record holds the protocol, the
harness commit, the dataset SHA-256, the track, the budgets, a hash of the question set, the manifest with
the SHA-256 of `adapter.py`, the runner (name, role, machine), the summary, and one line per question. The
evidence line holds the gold turns found and the tokens at each budget. The answer line holds the answer, the
verdict and the tokens. A record holds no dataset text. On the answer track, it holds the reader's answers.

## Who keeps the board

breadcrumbs-agency keeps this repository, and it also wrote the `breadcrumbs` entrant. Its runs have the role
`maintainer`. Each record holds what you need to run it again. `recent` (no search) and `bm25` (the search in
the Python standard library) are floors: they show what an agent gets with no memory product.

## Limits

- ceil(characters / 4) is not the tokenizer of a model. It is the same for every system, and it needs no
  download.
- Time and disk depend on the machine. The board says so when a group mixes machines.
- The judge prompt is this project's own. Compare answer-track numbers in this board, not with papers.
- LoCoMo is CC BY-NC 4.0 (non-commercial). The repository holds none of its text. Each runner must make sure
  that their use agrees with the license.
- A change to the stream, the meter or the scoring changes `PROTOCOL` in `memorybench/__init__.py`. Old
  records then stay in their own group.

## Tests

```
python -m unittest discover -s tests
```

## License

The harness and the adapters in `systems/` are MIT ([LICENSE](LICENSE)). Each dataset and each system keeps
its own license.
