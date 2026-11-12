# Board

Written by `python -m memorybench board`. Do not edit it by hand: run records in `results/public/` make it. README.md says how the numbers are made.

## locomo, evidence track

Questions: 1981. Budgets (tokens): 256, 512, 1024, 2048, 4096. Dataset SHA-256: 79fa87e90f04…. Question set: 14dd0b5a6b2e….

| # | System | Version | Access | Class | R@256 | R@512 | R@1024 | R@2048 | R@4096 | Mean | MRR | Ingest s | Recall p50 ms | Disk MB | Runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | breadcrumbs | 0.4.1 | open-source | local | 0.592 | 0.680 | 0.760 | 0.824 | 0.878 | 0.747 | 0.486 | 0.34 | 1.75 | 3.5 | breadcrumbs-agency (maintainer), macOS arm64 |
| 2 | bm25 | 1 | open-source | local | 0.504 | 0.589 | 0.660 | 0.702 | 0.747 | 0.640 | 0.427 | 0.15 | 0.64 | 3.8 | breadcrumbs-agency (maintainer), macOS arm64 |
| 3 | recent | 1 | open-source | local | 0.002 | 0.011 | 0.030 | 0.082 | 0.160 | 0.057 | 0.006 | 0.00 | 0.00 | 0.0 | breadcrumbs-agency (maintainer), macOS arm64 |

Head to head: the mean of (row − column) per question, over all budgets. ▲ the row is ahead and ▼ behind (the 95% interval excludes 0); = no clear difference. In brackets: questions won, tied and lost by the row.

| row − column | breadcrumbs | bm25 | recent |
|---|---|---|---|
| breadcrumbs |  | +0.106 ▲ (618/1218/145) | +0.690 ▲ (1790/143/48) |
| bm25 | -0.106 ▼ (145/1218/618) |  | +0.583 ▲ (1560/337/84) |
| recent | -0.690 ▼ (48/143/1790) | -0.583 ▼ (84/337/1560) |  |

## longmemeval-s, evidence track

Questions: 470. Budgets (tokens): 256, 512, 1024, 2048, 4096. Dataset SHA-256: d6f21ea9d60a…. Question set: 3913f325f99c….

| # | System | Version | Access | Class | R@256 | R@512 | R@1024 | R@2048 | R@4096 | Mean | MRR | Ingest s | Recall p50 ms | Disk MB | Runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | bm25 | 1 | open-source | local | 0.408 | 0.521 | 0.649 | 0.745 | 0.819 | 0.629 | 0.645 | 18.58 | 0.43 | 912.7 | breadcrumbs-agency (maintainer), macOS arm64 |
| 2 | breadcrumbs | 0.4.1 | open-source | local | 0.383 | 0.491 | 0.659 | 0.745 | 0.847 | 0.625 | 0.665 | 37.01 | 0.86 | 409.1 | breadcrumbs-agency (maintainer), macOS arm64 |
| 3 | recent | 1 | open-source | local | 0.000 | 0.000 | 0.002 | 0.007 | 0.031 | 0.008 | 0.005 | 0.07 | 0.00 | 0.0 | breadcrumbs-agency (maintainer), macOS arm64 |

Head to head: the mean of (row − column) per question, over all budgets. ▲ the row is ahead and ▼ behind (the 95% interval excludes 0); = no clear difference. In brackets: questions won, tied and lost by the row.

| row − column | bm25 | breadcrumbs | recent |
|---|---|---|---|
| bm25 |  | +0.004 = (104/267/99) | +0.621 ▲ (419/47/4) |
| breadcrumbs | -0.004 = (99/267/104) |  | +0.617 ▲ (433/35/2) |
| recent | -0.621 ▼ (4/47/419) | -0.617 ▼ (2/35/433) |  |

