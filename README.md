# Battledance Chess MLP-GA Trainer

Battledance Chess MLP-GA Trainer is a self-play training framework for **Battledance Chess**, a custom chesslike game with unusual leaper pieces, drops, royal bishops, and stalemate-as-loss.

The program implements:

- the Battledance Chess rules engine;
- a 3-hidden-layer tanh MLP board evaluator;
- a population-based genetic algorithm for evolving agents;
- per-agent snapshot history;
- resumable multi-process training;
- prelude seed ranking for stronger first-run initialization.

Yes, it is compute-heavy. No, it is not optimized. You have been warned.

---

## Features

### Battledance Chess rules engine

The code implements the full game engine used for training:

- 8×8 board.
- Custom pieces:
  - `K` — Kirin
  - `N` — kNight
  - `F` — Frog
  - `L` — Lancer
  - `P` — Phoenix
  - `G` — roGue
  - `R` — Rook
  - `B` — Bishop
- The royal pieces are bishops.
- Capturing a royal bishop wins.
- No legal moves means the side to move loses.
- Captured non-bishop pieces go into hand and may later be dropped onto that player’s home rows.
- Draw conditions:
  - threefold repetition;
  - 64-move rule, implemented as 128 plies since the last capture or drop;
  - hard long-game cap at 4096 plies.

The canonical starting position is encoded internally as:

```text
rglbblgr/pfnkknfp/8/8/8/8/PFNKKNFP/RGLBBLGR w - - 0 1
```

---

## Neural network evaluator

Each agent uses a feed-forward MLP to evaluate board positions.

Architecture:

```text
Input:  594 features
Hidden: 512 tanh
Hidden: 512 tanh
Hidden: 512 tanh
Output: 1 tanh scalar
```

The output is White-centric:

```text
+1 ≈ good for White
-1 ≈ good for Black
```

Position encoding includes:

- board occupancy and piece planes;
- pieces in hand;
- side to move;
- repetition count;
- 64-move counter;
- total game length.

Move choice is one-ply:

1. Generate all legal moves.
2. Apply each move temporarily.
3. Evaluate the resulting position.
4. Choose probabilistically among high-scoring moves.

This is not a search engine. It is a learned evaluator used greedily with weighted randomness.

---

## Training method

Training uses a population-based genetic algorithm rather than backpropagation.

Each trainable agent has a population of:

```text
260 networks
```

formed from:

```text
256 crossover/mutation children
4 elite direct clones
```

A generation is evaluated in two stages:

1. **Stage 1**: cheap evaluation of all 260 networks against the relevant frozen opponent snapshots.
2. **Stage 2**: heavier evaluation of the top candidates from Stage 1.

The top 8 successful candidates become the next parent set.

A generation only succeeds if the selected parents achieve non-negative margins against every required opponent snapshot at full Stage-2 resolution.

There is no default hard generation cap. A difficult agent can keep evolving until it passes the success gate.

---

## Agents

There are 15 named agents:

```text
Red Grn Blu Cyn Mag Yel NoN deR nrG ulB nyC gaM leY XyZ ZyX
```

Most element agents train against:

```text
6 matrix-selected element opponents + ZyX + self
```

`XyZ` trains broadly against:

```text
all 13 normal element labels + XyZ
```

`ZyX` trains narrowly against:

```text
XyZ + ZyX
```

This means agents do not all train against the same opponent set.

---

## Snapshot files

Each agent uses five snapshot slots:

```text
Name_0.pkl
Name_1.pkl
Name_2.pkl
Name_3.pkl
Name_4.pkl
```

Meanings:

```text
_0 = current trainable parent population / active cycle parents
_1 = newest completed parent set
_2 = previous champion
_3 = older champion
_4 = oldest retained champion
```

After a full cycle completes, snapshots rotate:

```text
_0 → _1
old _1 champion → _2
old _2 champion → _3
old _3 champion → _4
```

The newest `_1` file is a parent-list file, not just one champion. When a single champion is needed, the first parent in the list is used.

---

## Prelude initialization

On a fresh or empty `models` directory, plain execution defaults to prelude initialization:

```bash
python battledance_training.py
```

The prelude creates 60 distinct Xavier-initialized seed networks and ranks them with a round-robin tournament.

Default prelude workload:

```text
8 * 60^2 = 28,800 games
```

For distinct unordered seed pairs, this gives:

```text
16 games per pair
```

Self-intersection games cancel cleanly in scoring.

After ranking, the 60 seed networks are snake-assigned across the 15 labels and four retained snapshot slots using the built-in prelude order:

```text
ZyX XyZ deR nyC Red Cyn nrG gaM Grn Mag ulB leY Blu Yel NoN
```

The assignment shape is:

```text
_1: ranks  1..15
_2: ranks 30..16
_3: ranks 31..45
_4: ranks 60..46
```

Each label receives the same total rank budget.

To skip the prelude and use the older direct Xavier initialization:

```bash
python battledance_training.py --no-prelude-init
```

Prelude options:

```bash
python battledance_training.py --prelude-rounds 8
python battledance_training.py --prelude-workers 5
python battledance_training.py --prelude-seed 12345
```

If a prelude is interrupted before `training_state.json` exists, rerunning the script resumes the prelude from saved progress.

---

## Running training

Basic command:

```bash
python battledance_training.py
```

Default process mode:

```text
--threads-mode 5
```

Accepted process modes:

```bash
python battledance_training.py --threads-mode 1
python battledance_training.py --threads-mode 3
python battledance_training.py --threads-mode 5
```

Meaning:

```text
1 = single process
3 = three worker processes
5 = five worker processes
```

Press `q` while running to request a graceful stop. The program finishes the current game, checkpoints progress, and exits at a safe point.

Ctrl+C once also requests a graceful stop. Ctrl+C again forces interruption.

---

## Resumability

The trainer is designed for aggressive resumability.

Most long-running game passes checkpoint after every game, so a crash or graceful stop usually loses at most the current in-progress game.

Important durable state includes:

```text
models/training_state.json
models/cycle_progress.json
models/ga_done_<Name>.json
models/prelude_progress.json
models/prelude_ranking.json
models/prelude_assignment.json
models/cycle_<n>_rotation_done.json
models/cycle_<n>_rotation_summary.json
models/cycle_<n>_champions_rr_done.json
```

High-churn per-game progress markers are kept outside `models`:

```text
ga_progress/
```

Sample games and matrices are also kept outside `models`:

```text
sample_games/
```

This keeps `models` easier to browse while training is active.

---

## Output directories

The script creates these directories beside `battledance_training.py`:

```text
models/
ga_progress/
sample_games/
```

### `models/`

Contains model snapshots and durable training state.

Typical contents:

```text
Red_0.pkl
Red_1.pkl
Red_2.pkl
Red_3.pkl
Red_4.pkl
...
training_state.json
cycle_progress.json
prelude_progress.json
prelude_ranking.json
prelude_assignment.json
cycle_<n>_rotation_done.json
cycle_<n>_rotation_summary.json
cycle_<n>_champions_rr_done.json
training_log.txt
```

### `ga_progress/`

Contains frequently updated per-game progress files.

Typical contents:

```text
ga_progress_<Name>.json
champion_progress_<Name>.json
prelude_progress_worker_01.json
prelude_progress_worker_02.json
...
cycle_<n>_champions_rr_progress_worker_01.json
...
```

These files are intentionally separate from `models` because they update often during active training.

### `sample_games/`

Contains human-readable sample game logs and cycle-end champion round-robin outputs.

Typical contents:

```text
champion_matches_<Name>.txt
cycle_<n>_champions_rr_worker_01.txt
cycle_<n>_champions_rr.txt
cycle_<n>_matrix.txt
```

The cycle matrix files summarize directed results between the newly minted `_1` champions for a completed cycle.

---

## Cycle-end champion round-robin

After all agents complete a cycle and snapshots rotate, the script runs a champion round-robin between the newly minted `_1` champions.

Default workload:

```text
4 * 15^2 = 900 games
```

Outputs:

```text
sample_games/cycle_<n>_champions_rr.txt
sample_games/cycle_<n>_matrix.txt
```

The matrix is directed:

```text
row = White champion
column = Black champion
cell = summed result from White's perspective
```

With 4 reps, each cell is in:

```text
[-4, +4]
```

---

## Verified snapshot rotation

Snapshot rotation is treated as a safety-critical step.

Before any cycle is marked rotated, the script:

1. Builds per-agent full-payload rotation plans.
2. Writes those plans to disk.
3. Applies the planned snapshot rotation.
4. Reloads and verifies the written destinations.
5. Writes the rotation-done marker.
6. Writes a best-effort rotation summary.
7. Cleans up rotation plans best-effort.

This prevents a mid-rotation interruption from accidentally double-rotating snapshots on resume.

Rotation summary files are written to:

```text
models/cycle_<n>_rotation_summary.json
```

They include total and per-agent timing, verification status, and machine-readable snapshot transition metadata.

---

## Drive-blink tolerance

The file I/O layer uses temp-file atomic replacement and retry logic.

Relevant environment variables:

```text
BD_IO_RETRY_SECONDS
BD_IO_RETRY_INITIAL_DELAY
BD_IO_RETRY_MAX_DELAY
```

Defaults:

```text
BD_IO_RETRY_SECONDS=300
BD_IO_RETRY_INITIAL_DELAY=0.5
BD_IO_RETRY_MAX_DELAY=5.0
```

Example for a flaky external drive:

```bash
set BD_IO_RETRY_SECONDS=1800
python battledance_training.py
```

That gives the script up to 30 minutes to wait for temporarily missing storage before giving up on an I/O operation.

---

## Requirements

Python:

```text
3.10+
```

Python packages:

```text
numpy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A syntax check can be run with:

```bash
python -S -m py_compile battledance_training.py
```

---

## Suggested `.gitignore`

Generated training state and model files can become very large and should usually not be committed.

Recommended:

```gitignore
models/
ga_progress/
sample_games/
__pycache__/
*.pyc
*.tmp.*
```

---

## License

Copyright (C) 2025 Jacob Scow

This project is licensed under the GNU General Public License v3.0 or later.

See `LICENSE` for the full license text.

---

## Disclaimer

This was purely vibe-coded, right down to even the 4 sections above this one in this README.md being AI-written. I, Jacob Scow, just kept reiterating (parts of) the specs to the AI until it; actually gave me something that does the thing. Hopefully. I better hope so, because: Responsibility for the design, behavior, bugs, and licensing of this repository rests with me, not the AI.

Stability: Experimental. This is a “works on my machine, generated with AI (+ human prodding)” project. Expect bugs, weird edge cases, and rough edges. Even though I don't seem to experience any myself when running it on my machine.
