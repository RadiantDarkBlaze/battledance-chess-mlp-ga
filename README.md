# Battledance Chess MLP-GA Trainer

This repository contains two connected things:

1. **Battledance Chess** — a custom chesslike board game.
2. **An MLP-GA trainer** — a program that tries to evolve neural-network players for that game.

The trainer does not use human game records, opening books, or hand-written strategy. It creates many simple neural networks, lets them play Battledance Chess, keeps the better performers, mutates and recombines them, and repeats.

The short version:

```text
Battledance Chess = the game.
MLP = a small feed-forward neural network that scores board positions.
GA = a genetic algorithm that breeds better networks from better performers.
```

This is experimental toy/research code. It is built to run for a long time, save progress often, and resume after ordinary interruptions.

---

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

If the requirements file is not present, the important package is:

```bash
pip install numpy
```

Run the trainer:

```bash
python battledance_training.py
```

Request a graceful stop while it is running:

```text
q
```

On some terminals, use `q` and then Enter. The program finishes the current game, saves progress, and exits without pretending the current cycle is complete.

To resume, run the same command again:

```bash
python battledance_training.py
```

---

## The game in one page

Battledance Chess is played on an **8×8 board**. It looks like chess, but the pieces and capture rules are custom.

Each side starts with two rows of pieces. White uses uppercase letters. Black uses lowercase letters.

```text
xx a= b= c= d= e= f= g= h= xx
=8 ,r .g ,l .b ,b .l ,g .r =8
=7 .p ,f .n ,k .k ,n .f ,p =7
=6 ,, .. ,, .. ,, .. ,, .. =6
=5 .. ,, .. ,, .. ,, .. ,, =5
=4 ,, .. ,, .. ,, .. ,, .. =4
=3 .. ,, .. ,, .. ,, .. ,, =3
=2 ,P .F ,N .K ,K .N ,F .P =2
=1 .R ,G .L ,B .B ,L .G ,R =1
xx a= b= c= d= e= f= g= h= xx
```

The internal starting-position string is:

```text
rglbblgr/pfnkknfp/8/8/8/8/PFNKKNFP/RGLBBLGR w - - 0 1
```

### Objective

The **Bishop** is royal. Capturing any opposing Bishop wins immediately.

A player also loses if they have no legal moves.

If a Bishop capture is available, the engine treats Bishop-capturing moves as the only legal moves for that turn. In practice: if you can take a royal Bishop, you must end the game.

### Pieces

Most pieces are **leapers**: they jump directly to a target square. A leaper may land on an empty square or capture an enemy piece, but it may not land on a friendly piece.

Leaps are symmetric. For example, a `(2,1)` leap means every reflected/rotated version of that shape, like a chess knight.

| Letter | Name | Movement |
|---|---|---|
| `K` | Kirin | Leaps `(2,0)` and `(1,1)` |
| `N` | kNight | Leaps `(2,1)` |
| `F` | Frog | Leaps `(3,0)` and `(2,2)` |
| `L` | Lancer | Leaps `(3,1)` |
| `P` | Phoenix | Leaps `(1,0)` and `(3,3)` |
| `G` | roGue | Leaps `(3,2)` |
| `R` | Rook | Slides orthogonally, like a chess rook |
| `B` | Bishop | Slides diagonally, like a chess bishop; also royal |

A sliding piece keeps moving in one legal direction until it stops, captures, or hits the edge of the board.

### Check and legal moves

A side may not make a move that leaves its own Bishop attacked.

Because each side starts with two Bishops, this means both royal Bishops matter. Losing either opposing Bishop wins the game, and exposing your own Bishop is illegal.

### Captures and drops

When a player captures a **non-Bishop** piece, that piece goes into the capturing player’s hand.

On a later turn, instead of moving a board piece, the player may **drop** one held piece onto an empty square in their own home rows:

```text
White home rows: ranks 1 and 2
Black home rows: ranks 7 and 8
```

Drops do not capture. They place a held piece back onto the board.

### Draws

The engine declares a draw by any of these rules:

```text
threefold repetition
128 plies without a capture or drop
4096 total plies
```

A **ply** is one individual turn by one player. So 128 plies equals 64 full moves.

---

## What the neural player is doing

Each neural network is a board evaluator.

It does not understand the game in words. It receives a list of numbers describing the position, then outputs one score.

The default board encoding has **594 numbers**:

```text
piece locations
pieces in hand
side to move
draw/repetition counters
```

The default network shape is:

```text
594 -> 512 -> 512 -> 512 -> 1
```

That means:

```text
594 inputs
three hidden layers of 512 values each
1 output score
```

The hidden layers use `tanh`, so values are pushed into a smooth `-1` to `+1` range.

When choosing a move, the agent:

1. lists its legal moves;
2. temporarily plays each move;
3. evaluates the resulting board;
4. gives immediate wins a huge score and terminal draws a neutral score;
5. chooses randomly among the high-scoring moves, weighted toward the better ones.

That last bit matters. The agent is not perfectly deterministic, so repeated games between the same two networks can still differ.

---

## Neuroevolution, in plain English

This trainer does **not** use backpropagation. It does not compare the network’s move to a known correct answer.

Instead, it uses survival pressure:

```text
make many networks
let them play games
score their results
keep the better ones
breed new networks from them
mutate the children
repeat
```

That is the “GA” part: Genetic Algorithm.

The networks are not breeding because they are alive. It just means the program copies numbers from parent networks into child networks, then randomly tweaks some of those numbers.

### Crossover

A child network is built from two parents. For each weight or bias, the child gets that value from one parent or the other.

Plainly:

```text
child = mixed copy of parent A and parent B
```

### Mutation

After crossover, some values are randomly nudged.

The default mutation rate is:

```text
1/256
```

So each individual weight or bias has a small chance to be changed.

### Selection

After candidates play games, the trainer keeps the strongest candidates as parents for the next generation.

A network is not judged only by one lucky win. It is judged across many games against configured opponent snapshots.

---

## What a training run looks like

A fresh run starts with a **prelude**.

The prelude creates fresh Xavier-initialized seed networks, makes them play a seed round-robin, ranks them, and distributes them into the starting snapshot history.

With the default configuration:

```text
15 agents × 4 retained non-_0 snapshots = 60 prelude seed networks
```

After prelude, training proceeds in cycles.

A cycle roughly does this:

1. Train each agent against its configured opponent snapshots.
2. Evaluate each agent’s candidate networks.
3. Save a passing parent set when that agent succeeds.
4. Log champion audit games for that agent.
5. Once every agent is done, rotate snapshots.
6. Optionally run a cycle-end champion round-robin.
7. Advance the cycle counter.

If the program stops before the cycle counter advances, running it again resumes the unfinished work instead of treating the cycle as complete.

---

## Agents and snapshots

The default run has 15 named training lineages:

```text
Red, Grn, Blu, Cyn, Mag, Yel, NoN, deR, nrG, ulB, nyC, gaM, leY, XyZ, ZyX
```

You do not need to know the naming theme to understand the trainer. Each name is just one evolving player lineage.

Each agent has snapshot slots:

```text
Name_0.pkl   Active training population or current parent checkpoint.
Name_1.pkl   Most recent retained trained parents.
Name_2.pkl   Older retained champion snapshot.
Name_3.pkl   Older retained champion snapshot.
Name_4.pkl   Older retained champion snapshot.
```

The `_0` slot is active during training. The `_1` through `_4` slots are opponent history.

After a successful cycle, snapshots rotate:

```text
_0 -> _1
_1 -> _2
_2 -> _3
_3 -> _4
```

The rotation is designed to be resumable. If an interruption happens partway through rotation, the script uses its saved rotation plan rather than guessing.

---

## Population size and evaluation

The default population settings are:

```ini
[population]
parents = 8
children_per_parent_intersection = 4
elites = 4
```

That creates:

```text
8 × 8 × 4 = 256 children
256 children + 4 elite parent clones = 260 candidate networks
```

Evaluation happens in two stages:

```ini
[evaluation]
stage1_rounds = 2
stage2_finalists = 12
stage2_rounds = 16
```

Plain-English version:

```text
Stage 1: test every candidate with a smaller game budget.
Stage 2: spend more games only on the best finalists.
```

The success gate is strict: the chosen parents must have non-negative margins against every required opponent snapshot at the configured evaluation depth. A candidate that is great on average but clearly loses to one required opponent can fail the gate.

---

## Files and generated output

Main repository files:

```text
battledance_training.py   Game engine and training program.
training_config.ini       Editable defaults for fresh runs.
README.md                 This explanation.
LICENSE.txt               GPL-3.0-or-later license text.
requirements.txt          Python dependencies, if present.
.gitignore                Suggested ignores for generated files.
```

Runtime directories are created beside the script:

```text
models/        Model snapshots, training state, rotation state, and logs.
ga_progress/   Frequently updated progress JSON files.
sample_games/  Champion game logs and round-robin result matrices.
```

These generated directories can become large. They should normally stay out of Git.

Recommended `.gitignore` entries:

```gitignore
models/
ga_progress/
sample_games/
__pycache__/
*.pyc
*.tmp.*
```

---

## Fresh-run and resume safety

For a clean default run, start with these generated directories absent or empty:

```text
models/
ga_progress/
sample_games/
```

Once a run exists, do not manually edit, move, delete, or replace generated files unless you are intentionally abandoning or repairing that run.

The script is conservative. If the existing files do not look like a coherent run, it may refuse to continue instead of guessing what should be overwritten.

---

## Storage robustness

The trainer is meant for long runs, so it saves progress frequently.

It uses:

```text
atomic replace-style writes
frequent JSON progress markers
verified saves for important snapshots
resumable champion logs
resumable rotation plans
retry loops for temporary storage failures
```

It does **not** create persistent `.bak` duplicate files.

Temporary drive or storage loss is treated as something to wait through. If the storage root stays unavailable beyond the retry window, the script requests a graceful stop where possible.

The retry timing can be controlled with environment variables:

```bat
set BD_IO_RETRY_SECONDS=1800
python battledance_training.py
```

That example gives the script up to 30 minutes to wait for temporarily missing storage before giving up on an I/O operation.

The code also supports optional config values such as:

```ini
[runtime]
io_retry_seconds = 300
io_retry_initial_delay = 0.5
io_retry_max_delay = 5.0
```

---

## Useful commands

Run with a specific thread grouping:

```bash
python battledance_training.py --threads-mode 5
```

Override prelude worker count for a fresh or resumable prelude:

```bash
python battledance_training.py --prelude-workers 5
```

Override prelude rounds:

```bash
python battledance_training.py --prelude-rounds 8
```

Syntax-check the main script:

```bash
python -S -m py_compile battledance_training.py
```

---

## What to change carefully

`training_config.ini` exposes many knobs. Most are safe to experiment with only before a fresh run.

Usually safe between resumes:

```text
thread mode
I/O retry timing
prelude worker count, while still in prelude
```

Fresh-run-only unless you know exactly what you are doing:

```text
agent names
opponent lists
snapshot counts
hidden-layer architecture
population shape
evaluation game counts
prelude snake order
```

Changing structural settings mid-run can make existing snapshots or progress files incompatible.

---

## License

Copyright (C) 2025 Jacob Scow

This project is licensed under the GNU General Public License v3.0 or later.

See `LICENSE.txt` for the full license text.

---
## Disclaimer

This was purely vibe-coded, right down to even all sections above this one in this README.md being AI-written. I, Jacob Scow, just kept reiterating (parts of) the specs to the AI until it; actually gave me something that does the thing. Hopefully. I better hope so, because: Responsibility for the design, behavior, bugs, and licensing of this repository rests with me, not the AI.

Stability: Experimental. This is a “works on my machine, generated with AI (+ human prodding)” project. Expect bugs, weird edge cases, and rough edges. Even though I don't seem to experience any myself when running it on my machine.
