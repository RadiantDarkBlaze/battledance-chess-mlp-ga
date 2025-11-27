## License

Copyright (C) 2025 Jacob Scow

This project is licensed under the GNU General Public License v3.0 or later.  
See the `LICENSE.txt` file for details.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Jacob Scow, a.k.a., RadiantDarkBlaze

---

# Battledance Chess Trainer

Battledance Chess Trainer is a self-play training framework for the **Battledance Chess** variant.
It implements the full game rules, a neural network evaluation model, and a population-based
evolutionary training loop with disk-based snapshots and resumability.

Yes, it’s compute-heavy. No, it’s not optimized. You’ve been warned.

---

## Features

- **Full Battledance Chess rules**
  - Custom leap and slide pieces (K, N, F, L, P, G, R, B)
  - Drop rules using captured pieces
  - Draw conditions:
    - Threefold repetition
    - 64-move rule (128 plies since last capture/drop)
    - Long game cap (4096 plies)

- **Neural network evaluator**
  - Encodes each position as a **594-dimensional** feature vector:
    - 9 piece/all-piece planes over 8×8
    - 14 “in-hand” piece counters
    - 4 scalar features (side to move, 64-move counter, repetition, game length)
  - 3-layer tanh MLP:
    - Input: 594
    - Hidden: 512 → 512 → 512
    - Output: 1 (tanh scalar, White-centric)

- **Neuro-evolution training**
  - Population of **260 networks**:
    - 256 children from uniform crossover + mutation
    - 4 elite clones carried over
  - Two-stage evaluation:
    - Stage 1: cheap evaluation vs frozen snapshots (few games per colour)
    - Stage 2: heavy evaluation on top candidates (many games per colour)
  - Success condition: 8 selected parents must have **non-negative margin vs all opponent snapshots**

- **Snapshot and cycle management**
  - Per-agent snapshots: `Name_0.pkl` .. `Name_3.pkl`
    - `_0`: current parents (8 nets)
    - `_1`: parents from previous cycle
    - `_2`, `_3`: individual champion snapshots from earlier cycles
  - After each full cycle:
    - Snapshots are rotated (`_0 → _1`, champions extracted, older champions shifted down)
    - Cycle counter is incremented

- **Resumable and crash-tolerant**
  - Progress files:
    - `training_state.json` – global cycle counter
    - `cycle_progress.json` – per-agent cycle state
    - `ga_progress_<Name>.json` – per-generation GA progress
    - `ga_done_<Name>.json` – GA completion markers per agent/cycle
    - `champion_progress_<Name>.json` – champion-match progress
  - At most **one in-progress game** is lost upon interruption.

- **Multi-process training**
  - Training cycles can run with:
    - `--threads-mode 1` – single process, sequential
    - `--threads-mode 3` – three worker processes with fixed agent groupings
    - `--threads-mode 5` – five worker processes with fixed agent groupings

---

## Requirements

- **Python**: 3.10+
  (Uses `dataclasses.dataclass(slots=True)`, which requires 3.10 or later.)
- **Python packages**:
  - `numpy`

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

Run a training cycle (example):

```bash
python battledance_training.py
```
Default `--threads-mode` is `5`, accepted args:
```bash
python battledance_training.py --threads-mode 1
python battledance_training.py --threads-mode 3
python battledance_training.py --threads-mode 5
```
All other `--threads-mode` args will print an error and exit (with a non-zero status).

---

## Disclaimer

This was purely vibe-coded, right down to even the 4 sections above this one
in this README.md being AI-written. I, Jacob Scow, just kept reiterating (parts of)
the specs to the AI until it; actually gave me something that does the thing.
Hopefully. I better hope so, because: Responsibility for the design, behavior,
bugs, and licensing of this repository rests with me, not the AI.

**Stability:** Experimental.
This is a “works on my machine, generated with AI (+ human prodding)” project.
Expect bugs, weird edge cases, and rough edges. Even though I don't seem to
experience any myself when running it on my machine.
