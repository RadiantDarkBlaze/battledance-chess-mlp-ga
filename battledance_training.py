# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Jacob Scow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 or
# later as published by the Free Software Foundation.
#
# See the LICENSE file in the project root for details.

# Jacob Scow, a.k.a., RadiantDarkBlaze

r"""
Battledance Chess training program
================================

This module implements a self-play training framework for the
Battledance Chess variant.  The code follows a detailed
specification that defines the game rules, a neural network
evaluation model, a population-based evolutionary training scheme,
snapshot semantics, and champion evaluation.  It is intended to run
locally (using one or several worker processes) and saves all state
to disk so training can be resumed between runs.

Key features:

* Implements the full rules of Battledance Chess, including custom
  leap and slide pieces, drop rules, and draw conditions (threefold
  repetition, 64-move rule, and a long game cap).
* Encodes the game state as a 594-dimensional feature vector and
  evaluates positions using a three-layer tanh MLP.
* Uses a neuro-evolution regimen: a population of 260 networks is
  evolved via selection, crossover and mutation until eight parents
  meet a win-rate threshold against frozen opponent snapshots.
* Maintains four snapshots per agent (\_0..\_3) with strict
  semantics for parents and champions across cycles.
* After each agent finishes training in a cycle, its champion plays
  evaluation games against opponent champions and logs the results.
* After all agents finish training in a cycle, snapshots are rotated
  and a new population is seeded from the latest parents.

Note: This implementation prioritises correctness and adherence to
the specification over runtime performance.  Running this program
will require significant computation time.

This project is licensed under the GNU General Public License v3.0 or later.
See the `LICENSE.txt` file for details.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
import json
import pickle
import random
import hashlib
import time
import argparse
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np

###############################################################################
#  Simple file-based training log
###############################################################################

LOG_PATH: Optional[str] = None
CURRENT_AGENT_NAME: str = ""
CURRENT_GEN: int = 0
CURRENT_CYCLE: int = 0


def setup_logging(base_dir: str, cycle: int) -> None:
    """
    Initialise global log file for this run. Appends a session header to
    models/training_log.txt. Safe to call multiple times; only the first
    call in a process sets LOG_PATH.
    """
    global LOG_PATH, CURRENT_CYCLE
    CURRENT_CYCLE = cycle
    if LOG_PATH is None:
        os.makedirs(base_dir, exist_ok=True)
        LOG_PATH = os.path.join(base_dir, "training_log.txt")
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n=== New session cycle={cycle} started {ts} ===\n")


def log(msg: str, also_print: bool = True) -> None:
    """
    Append a timestamped line to the log file and optionally echo to stdout.
    Logging must never crash training; I/O errors are swallowed.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}] {msg}"
    if also_print:
        print(line, flush=True)
    if LOG_PATH is not None:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # Logging must never crash training
            pass

_last_status_len: int = 0


def status(msg: str) -> None:
    """
    Update a single carriage-return status line on the console.

    Uses only '\r' and spaces (no ANSI escape sequences), so it behaves
    sensibly on plain Windows consoles. Nothing from here is written to
    the log file.
    """
    global _last_status_len
    line = msg[:200]  # cap length to something sane

    # Move cursor to column 0 and write the new text
    sys.stdout.write("\r" + line)

    # If the previous line was longer, overwrite the tail with spaces
    if _last_status_len > len(line):
        sys.stdout.write(" " * (_last_status_len - len(line)))
        # Move back to start and re-write the new line (so we don't end on spaces)
        sys.stdout.write("\r" + line)

    sys.stdout.flush()
    _last_status_len = len(line)


###############################################################################
#  Cycle progress tracking (for resumability)
###############################################################################

def load_cycle_progress(
    base_dir: str,
    cycle: int,
    agent_names: List[str],
) -> Dict:
    """
    Load or initialise per-cycle, per-agent progress.

    Layout of cycle_progress.json:

    {
      "cycle": <int>,
      "agents": {
        "Red": {"state": "pending"|"done", "last_gen": <int>, "last_cycle": <int>},
        ...
      }
    }

    Semantics:
      * "pending": GA training and champion matches for this agent in this
        cycle have not been completed (or must be redone).
      * "done": GA training and champion matches completed for this agent
        in this cycle.

    Any legacy or unknown state values are treated as "pending" by the
    caller.
    """
    path = os.path.join(base_dir, "cycle_progress.json")
    data: Dict[str, object]

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    if data.get("cycle") != cycle:
        # New cycle or corrupted file; start fresh.
        agents_progress: Dict[str, Dict[str, object]] = {}
    else:
        agents_progress = data.get("agents", {}) or {}

    # Ensure every agent has at least a stub entry.
    for name in agent_names:
        if name not in agents_progress:
            agents_progress[name] = {"state": "pending"}

    return {"cycle": cycle, "agents": agents_progress}


def save_cycle_progress(base_dir: str, progress: Dict) -> None:
    """
    Persist the current per-cycle, per-agent progress state.

    I/O errors are swallowed to avoid crashing training.
    """
    path = os.path.join(base_dir, "cycle_progress.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
    except Exception:
        # Progress logging must not crash training.
        pass

def load_champion_progress(base_dir: str, cycle: int, name: str) -> Dict:
    """
    Load or initialise per-cycle champion-match progress for a single agent.

    Layout of champion_progress_<name>.json:

    {
      "cycle": <int>,
      "next_index": <int>
    }

    Semantics:
      * next_index = number of champion-match games already completed
        for this agent in this cycle (0-based; games [0..next_index-1]
        are considered done).
    """
    path = os.path.join(base_dir, f"champion_progress_{name}.json")
    data: Dict[str, object]

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    if data.get("cycle") != cycle:
        # New cycle or corrupted file; start fresh for this agent.
        return {"cycle": cycle, "next_index": 0}

    # Ensure next_index is sane
    next_index = data.get("next_index", 0)
    if not isinstance(next_index, int):
        next_index = 0
    data["next_index"] = int(next_index)
    return data


def save_champion_progress(base_dir: str, name: str, progress: Dict) -> None:
    """
    Persist per-cycle champion-match progress state for a single agent.

    I/O errors are swallowed to avoid crashing training.
    """
    path = os.path.join(base_dir, f"champion_progress_{name}.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
    except Exception:
        # Progress logging must not crash training.
        pass

###############################################################################
#  GA generation progress tracking (for resumability)
###############################################################################

GA_BASE_DIR: Optional[str] = None  # set by run_training_cycle


def _ga_progress_path() -> Optional[str]:
    """
    Return the path to the GA progress file for the current agent, or None
    if GA_BASE_DIR or CURRENT_AGENT_NAME is not yet set.
    """
    if GA_BASE_DIR is None or not CURRENT_AGENT_NAME:
        return None
    return os.path.join(GA_BASE_DIR, f"ga_progress_{CURRENT_AGENT_NAME}.json")


def load_ga_progress() -> Optional[Dict]:
    """
    Load GA progress JSON for the current agent, or None if unavailable.

    This file tracks per-generation GA evaluation progress so that at most
    one in-progress Battledance game needs to be replayed after interruption.
    """
    path = _ga_progress_path()
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def save_ga_progress(progress: Dict) -> None:
    """
    Persist GA progress JSON. I/O errors are swallowed; failure here must
    never crash training.
    """
    path = _ga_progress_path()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
    except Exception:
        pass


def reset_ga_progress() -> None:
    """
    Remove any GA progress file for the current agent. Called after a
    generation successfully completes so that the next generation starts
    with a clean state.
    """
    path = _ga_progress_path()
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _ga_done_file(base_dir: str, name: str) -> str:
    """
    Return the path to the GA 'done' marker file for this agent.

    This records that GA training for `name` has successfully completed
    for a specific cycle, so we can skip re-running GA on later resumes.
    """
    return os.path.join(base_dir, f"ga_done_{name}.json")


def load_ga_done_state(base_dir: str, name: str) -> Optional[Dict[str, object]]:
    """
    Load the GA 'done' marker for this agent, or None if absent/invalid.

    Layout:

      {
        "cycle": <int>,
        "agent": <str>,
        "last_gen": <int>,
        "timestamp": <str>
      }
    """
    path = _ga_done_file(base_dir, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def save_ga_done_state(base_dir: str, name: str, cycle: int, last_gen: int) -> None:
    """
    Persist a GA 'done' marker for this agent and cycle.

    I/O errors are swallowed to avoid crashing training.
    """
    path = _ga_done_file(base_dir, name)
    payload = {
        "cycle": int(cycle),
        "agent": name,
        "last_gen": int(last_gen),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        # Marker is purely an optimisation; never crash training.
        pass


###############################################################################
#  Game representation
###############################################################################

# Shared constants for feature encoding; avoids rebuilding these every call.
PIECE_ORDER: List[str] = ['K', 'N', 'F', 'L', 'P', 'G', 'R', 'B']
PIECE_INDEX: Dict[str, int] = {k: i for i, k in enumerate(PIECE_ORDER)}
IN_HAND_TYPES: List[str] = ['K', 'N', 'F', 'L', 'P', 'G', 'R']
FEATURE_PLANES = 9 * 8 * 8  # 9 planes × 8×8
FEATURE_TOTAL = FEATURE_PLANES + 14 + 4  # board + in-hand + scalars = 594

class Piece:
    """A single Battledance chess piece."""

    __slots__ = ("kind", "color")

    def __init__(self, kind: str, color: str) -> None:
        # kind: uppercase letter identifying the piece type
        # color: 'w' for white, 'b' for black
        self.kind = kind
        self.color = color


@dataclass(slots=True)
class Move:
    """
    Representation of a single move on the Battledance board.

    * ``kind``: either ``'move'`` for a normal piece move or ``'drop'`` for a drop.
    * ``from_pos``: (row, col) origin for normal moves; ``None`` for drops.
    * ``to_pos``: (row, col) destination for the move.
    * ``drop_type``: for drops, the piece type being dropped (uppercase for white,
      lowercase for black); ignored for normal moves.
    """
    kind: str
    from_pos: Optional[Tuple[int, int]]
    to_pos: Tuple[int, int]
    drop_type: Optional[str] = None


class BattledanceBoard:
    """
    Implements the rules of Battledance Chess and manages game state.

    The board is always 8×8. The constructor accepts a Battledance FEN
    string of the form:

        "<pieces>/... x8 ... <pieces> <side_to_move> - - 0 1"

    Only the piece placement and side-to-move fields are parsed; the
    numeric counters are tracked internally.  Repetition information is
    maintained so that threefold repetition can be detected.
    """

    BASE_LEAPS: Dict[str, List[Tuple[int, int]]] = {
        'K': [(2, 0), (1, 1)],  # Kirin
        'N': [(2, 1)],          # Knight
        'F': [(3, 0), (2, 2)],  # Frog
        'L': [(3, 1)],          # Lancer
        'P': [(1, 0), (3, 3)],  # Phoenix
        'G': [(3, 2)],          # Rogue
        'R': [],                # Rook slides only
        'B': [],                # Bishop slides only
    }

    BASE_SLIDES: Dict[str, List[Tuple[int, int]]] = {
        'K': [],
        'N': [],
        'F': [],
        'L': [],
        'P': [],
        'G': [],
        'R': [(1, 0)],
        'B': [(1, 1)],
    }

    # Small caches so we don't rebuild symmetry tables constantly
    _LEAP_CACHE: Dict[str, List[Tuple[int, int]]] = {}
    _SLIDE_CACHE: Dict[str, List[Tuple[int, int]]] = {}

    def __init__(self, fen: str) -> None:
        """
        Create a Battledance board from a FEN-like string.

        Only the piece placement and side-to-move fields are honoured.
        All repetition / draw counters start at zero and the initial
        position is recorded for threefold-repetition tracking.
        """
        # 8×8 board initialised to empty
        self.board: List[List[Optional[Piece]]] = [
            [None for _ in range(8)] for _ in range(8)
        ]
        self.captured_w: List[str] = []
        self.captured_b: List[str] = []

        parts = fen.split()
        board_part = parts[0] if parts else "8/8/8/8/8/8/8/8"
        side = parts[1] if len(parts) > 1 else "w"

        rows = board_part.split("/")
        if len(rows) != 8:
            raise ValueError(f"Invalid Battledance FEN (expected 8 ranks): {fen!r}")

        # FEN ranks go from 8 (top) to 1 (bottom), so r = 0 is Black's back rank.
        for r, row_str in enumerate(rows):
            c = 0
            for ch in row_str:
                if ch.isdigit():
                    c += int(ch)
                    continue
                if c >= 8:
                    raise ValueError(f"Too many files in rank {r} of FEN {fen!r}")
                color = "w" if ch.isupper() else "b"
                kind = ch.upper()
                self.board[r][c] = Piece(kind, color)
                c += 1
            if c > 8:
                raise ValueError(f"Too many files in rank {r} of FEN {fen!r}")

        self.turn = "w" if side == "w" else "b"
        self.plys_since_capture_or_drop = 0
        self.plys_since_game_start = 0

        # Map position keys to number of occurrences so far
        self.repetition_counts: Dict[str, int] = {}

        # Record the initial position for repetition tracking.
        self.record_state()

    def home_rows(self, color: str) -> List[int]:
        """
        Return the list of row indices that count as the home rows for `color`.

        By convention:
          * White's home rows are ranks 1–2 (internal rows 6 and 7).
          * Black's home rows are ranks 7–8 (internal rows 0 and 1).

        These rows are the only ones eligible for drop moves.
        """
        if color == "w":
            return [6, 7]
        else:
            return [0, 1]

    def _position_string(self) -> str:
        """
        Return a canonical string representation of the current position
        (board, side to move, and pieces in hand) for repetition tracking.
        """
        rows: List[str] = []
        for r in range(8):
            row_chars: List[str] = []
            for c in range(8):
                cell = self.board[r][c]
                if cell is None:
                    row_chars.append(".")
                else:
                    ch = cell.kind.upper() if cell.color == "w" else cell.kind.lower()
                    row_chars.append(ch)
            rows.append("".join(row_chars))
        board_str = "/".join(rows)

        # Use multiset of in-hand pieces (order-independent)
        w_hand = "".join(sorted(self.captured_w))
        b_hand = "".join(sorted(self.captured_b))

        return f"{board_str} {self.turn} W[{w_hand}] B[{b_hand}]"

    def _position_key(self) -> str:
        """Return a stable hash key for the current position."""
        s = self._position_string()
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def record_state(self) -> None:
        """
        Update the internal repetition table for the current position.

        Called after the initial setup and after every applied move.
        """
        key = self._position_key()
        self.repetition_counts[key] = self.repetition_counts.get(key, 0) + 1

    def repetition_count(self) -> int:
        """
        Return how many times the current position has occurred so far
        in this game (including the current occurrence).
        """
        if not hasattr(self, "repetition_counts"):
            return 1
        key = self._position_key()
        return self.repetition_counts.get(key, 0)

    @classmethod
    def symmetrical_leaps(cls, kind: str) -> List[Tuple[int, int]]:
        """Return all unique leap offsets for the given piece kind (cached)."""
        cached = cls._LEAP_CACHE.get(kind)
        if cached is not None:
            return cached

        base = cls.BASE_LEAPS.get(kind, [])
        offsets: set[Tuple[int, int]] = set()
        for dx, dy in base:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    offsets.add((sx * dx, sy * dy))
                    if dx != dy:
                        offsets.add((sx * dy, sy * dx))

        result = list(offsets)
        cls._LEAP_CACHE[kind] = result
        return result

    @classmethod
    def symmetrical_slides(cls, kind: str) -> List[Tuple[int, int]]:
        """Return all unique slide directions for the given piece kind (cached)."""
        cached = cls._SLIDE_CACHE.get(kind)
        if cached is not None:
            return cached

        base = cls.BASE_SLIDES.get(kind, [])
        dirs: set[Tuple[int, int]] = set()
        for dx, dy in base:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    dirs.add((sx * dx, sy * dy))
                    if dx != dy:
                        dirs.add((sx * dy, sy * dx))

        result = list(dirs)
        cls._SLIDE_CACHE[kind] = result
        return result

    def get_bishop_positions(self, color: str) -> List[Tuple[int, int]]:
        """Return positions of all royal bishops of the given colour."""
        positions: List[Tuple[int, int]] = []
        for r in range(8):
            for c in range(8):
                cell = self.board[r][c]
                if cell is not None and cell.kind == 'B' and cell.color == color:
                    positions.append((r, c))
        return positions

    def is_square_attacked(self, row: int, col: int, by_color: str) -> bool:
        """
        Determine if (row, col) is attacked by a piece of colour ``by_color``.
        Drops cannot capture, so only leaps and slides are considered.
        """
        for r in range(8):
            for c in range(8):
                cell = self.board[r][c]
                if cell is None or cell.color != by_color:
                    continue
                kind = cell.kind
                # Check leap attacks
                for dx, dy in self.symmetrical_leaps(kind):
                    nr, nc = r + dx, c + dy
                    if nr == row and nc == col:
                        return True
                # Check slide attacks
                for dx, dy in self.symmetrical_slides(kind):
                    step = 1
                    while True:
                        nr = r + dx * step
                        nc = c + dy * step
                        if not (0 <= nr < 8 and 0 <= nc < 8):
                            break
                        target = self.board[nr][nc]
                        if nr == row and nc == col:
                            return True
                        if target is not None:
                            break
                        step += 1
        return False

    def is_in_check(self, color: str) -> bool:
        """Return True if any bishop of the given colour is under attack."""
        enemy = 'b' if color == 'w' else 'w'
        for (r, c) in self.get_bishop_positions(color):
            if self.is_square_attacked(r, c, enemy):
                return True
        return False

    def generate_moves_unfiltered(self, color: str) -> List[Move]:
        """
        Generate all possible moves (including drops) for ``color``
        without considering whether the move leaves a royal bishop
        attacked.  These moves will be filtered later.
        """
        moves: List[Move] = []
        # Piece moves
        for r in range(8):
            for c in range(8):
                cell = self.board[r][c]
                if cell is None or cell.color != color:
                    continue
                kind = cell.kind
                leaps = self.symmetrical_leaps(kind)
                slides = self.symmetrical_slides(kind)

                # Leap moves
                for dx, dy in leaps:
                    nr, nc = r + dx, c + dy
                    if not (0 <= nr < 8 and 0 <= nc < 8):
                        continue
                    target = self.board[nr][nc]
                    if target is None or target.color != color:
                        moves.append(Move('move', (r, c), (nr, nc)))

                # Slide moves
                for dx, dy in slides:
                    step = 1
                    while True:
                        nr = r + dx * step
                        nc = c + dy * step
                        if not (0 <= nr < 8 and 0 <= nc < 8):
                            break
                        target = self.board[nr][nc]
                        if target is None:
                            moves.append(Move('move', (r, c), (nr, nc)))
                            step += 1
                            continue
                        elif target.color != color:
                            moves.append(Move('move', (r, c), (nr, nc)))
                        break

        # Drop moves
        captured_list = self.captured_w if color == 'w' else self.captured_b
        for ch in captured_list:
            for hr in self.home_rows(color):
                for col in range(8):
                    if self.board[hr][col] is None:
                        moves.append(Move('drop', None, (hr, col), drop_type=ch))
        return moves

    def generate_legal_moves(self, color: str) -> List[Move]:
        cand = self.generate_moves_unfiltered(color)

        # First: collect all bishop-capture moves.
        bishop_caps: List[Move] = []
        for move in cand:
            if move.kind == 'move':
                tr, tc = move.to_pos
                target = self.board[tr][tc]
                if target is not None and target.kind == 'B' and target.color != color:
                    bishop_caps.append(move)

        # If any bishop captures exist, they are the only legal moves.
        if bishop_caps:
            return bishop_caps

        # Otherwise fall back to normal “don’t leave my bishops in check” logic.
        legal: List[Move] = []
        for move in cand:
            clone = self.copy()
            clone.apply_move(move)
            if clone.is_in_check(color):
                continue
            legal.append(move)
        return legal

    def copy(self) -> 'BattledanceBoard':
        """Deep copy the board, captured lists, counters and repetition history."""
        clone = object.__new__(BattledanceBoard)
        clone.board = [
            [None if cell is None else Piece(cell.kind, cell.color) for cell in row]
            for row in self.board
        ]
        clone.captured_w = list(self.captured_w)
        clone.captured_b = list(self.captured_b)
        clone.turn = self.turn
        clone.plys_since_capture_or_drop = self.plys_since_capture_or_drop
        clone.plys_since_game_start = self.plys_since_game_start
        clone.repetition_counts = dict(self.repetition_counts)
        return clone

    def apply_move(self, move: Move) -> Optional[str]:
        """
        Apply a move and update counters and repetition history.  Returns
        'w' if White wins, 'b' if Black wins, or None if the game
        continues.  Capturing a royal bishop ends the game.  Counters
        are reset after any capture or drop.
        """
        color = self.turn
        enemy = 'b' if color == 'w' else 'w'
        winner: Optional[str] = None
        capture_or_drop = False
        if move.kind == 'move':
            fr, fc = move.from_pos
            tr, tc = move.to_pos
            piece = self.board[fr][fc]
            target = self.board[tr][tc]
            # Move piece
            self.board[fr][fc] = None
            if target is not None:
                # capturing bishop ends game
                if target.kind == 'B':
                    winner = color
                captured_char = target.kind.upper() if color == 'w' else target.kind.lower()
                if winner is None:
                    # captured piece becomes in hand
                    if color == 'w':
                        self.captured_w.append(captured_char)
                    else:
                        self.captured_b.append(captured_char)
                capture_or_drop = True
            # Place piece
            self.board[tr][tc] = Piece(piece.kind, color)
        else:
            # Drop move
            tr, tc = move.to_pos
            ch = move.drop_type
            drop_color = 'w' if ch.isupper() else 'b'
            drop_kind = ch.upper()
            self.board[tr][tc] = Piece(drop_kind, drop_color)
            if drop_color == 'w':
                self.captured_w.remove(ch)
            else:
                self.captured_b.remove(ch)
            capture_or_drop = True
        # Switch turn
        self.turn = enemy
        # Update counters
        self.plys_since_game_start += 1
        if capture_or_drop:
            self.plys_since_capture_or_drop = 0
        else:
            self.plys_since_capture_or_drop += 1
        # Record new state
        self.record_state()
        return winner

    def check_draw(self) -> bool:
        """Return True if any draw condition triggers."""
        # Threefold repetition
        if self.repetition_count() >= 3:
            return True
        # 64-move rule
        if self.plys_since_capture_or_drop >= 128:
            return True
        # Long game (2048 full moves = 4096 plies)
        if self.plys_since_game_start >= 4096:
            return True
        return False

    def encode_features(self) -> np.ndarray:
        """
        Encode the current state into a 594-dimensional float32 vector.

        Features:
        - 8 piece-specific planes for K,N,F,L,P,G,R,B (white +1, black −1)
        - 1 plane for any piece (white +1, black −1)
        - 14 in-hand scalars (White K,N,F,L,P,G,R, then Black K,N,F,L,P,G,R)
          clipped at 4 and divided by 4
        - side to move: +1 if White, −1 if Black
        - 64-move counter: plys_since_capture_or_drop / 128
        - repetition count: min(rep_count,3) / 3
        - long game counter: plys_since_game_start / 4096
        """
        # Board planes: shape (9, 8, 8)
        planes = np.zeros((9, 8, 8), dtype=np.float32)

        # Pieces on board
        for r in range(8):
            row = self.board[r]
            for c in range(8):
                cell = row[c]
                if cell is None:
                    continue
                idx = PIECE_INDEX.get(cell.kind)
                if idx is None:
                    continue
                val = 1.0 if cell.color == 'w' else -1.0
                planes[idx, r, c] = val
                planes[8, r, c] = val  # "any piece" plane

        # Flatten planes once
        feat = planes.reshape(FEATURE_PLANES)

        # In-hand features
        in_hand = np.zeros(14, dtype=np.float32)

        # white captured pieces (uppercase in list)
        for i, t in enumerate(IN_HAND_TYPES):
            count = 0
            for ch in self.captured_w:
                if ch.upper() == t:
                    count += 1
            in_hand[i] = min(count, 4) / 4.0

        # black captured pieces (lowercase in list)
        for i, t in enumerate(IN_HAND_TYPES):
            count = 0
            for ch in self.captured_b:
                if ch.lower() == t.lower():
                    count += 1
            in_hand[7 + i] = min(count, 4) / 4.0

        # Scalar features
        stm = 1.0 if self.turn == 'w' else -1.0
        m64 = min(self.plys_since_capture_or_drop / 128.0, 1.0)
        rep = min(self.repetition_count(), 3) / 3.0
        glen = min(self.plys_since_game_start / 4096.0, 1.0)
        scalars = np.array([stm, m64, rep, glen], dtype=np.float32)

        # Single allocation for full feature vector
        out = np.empty(FEATURE_TOTAL, dtype=np.float32)
        out[:FEATURE_PLANES] = feat
        out[FEATURE_PLANES:FEATURE_PLANES + 14] = in_hand
        out[FEATURE_PLANES + 14:] = scalars
        return out


###############################################################################
#  Neural network
###############################################################################

class MLP:
    """
    Multi‑layer perceptron with three hidden layers and tanh activations.

    The network architecture is fixed: input_dim=594, hidden_dim=512,
    output_dim=1.  Xavier/Glorot initialisation is used.  Mutation
    applies Gaussian noise to a fraction of weights and biases.
    """

    def __init__(self, input_dim: int = 594, hidden_dim: int = 512, seed: Optional[int] = None) -> None:
        rnd = np.random.RandomState(seed)
        def xavier(shape: Tuple[int, int]):
            fan_in, fan_out = shape
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rnd.uniform(-limit, limit, shape).astype(np.float32)
        self.W1 = xavier((input_dim, hidden_dim))
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.W2 = xavier((hidden_dim, hidden_dim))
        self.b2 = np.zeros((hidden_dim,), dtype=np.float32)
        self.W3 = xavier((hidden_dim, hidden_dim))
        self.b3 = np.zeros((hidden_dim,), dtype=np.float32)
        self.W4 = xavier((hidden_dim, 1))
        self.b4 = np.zeros((1,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> float:
        """Compute forward pass returning tanh scalar."""
        z1 = np.tanh(x.dot(self.W1) + self.b1)
        z2 = np.tanh(z1.dot(self.W2) + self.b2)
        z3 = np.tanh(z2.dot(self.W3) + self.b3)
        out = np.tanh(z3.dot(self.W4) + self.b4)
        return float(out[0])

    def copy(self) -> 'MLP':
        new = object.__new__(MLP)
        new.W1 = self.W1.copy()
        new.b1 = self.b1.copy()
        new.W2 = self.W2.copy()
        new.b2 = self.b2.copy()
        new.W3 = self.W3.copy()
        new.b3 = self.b3.copy()
        new.W4 = self.W4.copy()
        new.b4 = self.b4.copy()
        return new

    def mutate(self, mutation_rate: float, weight_decay: float, rnd: np.random.RandomState) -> None:
        """Mutate the network in place using Xavier-scaled Gaussian noise."""
        def xavier_scale(arr: np.ndarray, is_weight: bool) -> float:
            if not is_weight:
                # Biases: just use a small fixed scale
                return 0.05
            fan_in, fan_out = arr.shape
            limit = np.sqrt(6.0 / float(fan_in + fan_out))
            # Convert uniform limit to a rough Gaussian std; factor ~0.5 keeps noise modest
            return float(limit * 0.5)

        for attr in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']:
            arr: np.ndarray = getattr(self, attr)
            is_weight = attr.startswith('W')
            scale = xavier_scale(arr, is_weight=is_weight)

            if scale <= 0.0:
                continue

            mask = rnd.rand(*arr.shape) < mutation_rate
            if not mask.any():
                # still apply weight decay
                arr -= weight_decay * arr
                continue

            noise = rnd.randn(*arr.shape).astype(np.float32) * scale
            arr += mask * noise
            arr -= weight_decay * arr

    def crossover(self, other: 'MLP', rnd: np.random.RandomState) -> 'MLP':
        """Return a child network from this and another network via uniform crossover."""
        child = MLP()
        for attr in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']:
            a: np.ndarray = getattr(self, attr)
            b: np.ndarray = getattr(other, attr)
            mask = rnd.rand(*a.shape) < 0.5
            setattr(child, attr, np.where(mask, a, b).astype(np.float32))
        return child

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {attr: getattr(self, attr) for attr in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']}

    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> 'MLP':
        net = cls()
        for attr in data:
            setattr(net, attr, data[attr].astype(np.float32))
        return net


###############################################################################
#  Agent
###############################################################################

@dataclass
class Agent:
    """
    An agent consisting of a neural network and a name.  The network
    evaluates positions from White’s perspective; sign flips are
    applied externally.  Agents can be trainable or frozen.
    """
    name: str
    net: MLP = field(default_factory=lambda: MLP())
    trainable: bool = True

    def clone(self) -> 'Agent':
        return Agent(
            name=self.name,
            net=self.net.copy(),
            trainable=self.trainable,
        )

    def evaluate_state(self, board: BattledanceBoard) -> float:
        features = board.encode_features()
        return self.net.forward(features)

    def choose_move(self, board: BattledanceBoard) -> Optional[Move]:
        color = board.turn
        legal = board.generate_legal_moves(color)
        if not legal:
            return None

        candidate_moves = legal
        scores: List[float] = []
        for mv in candidate_moves:
            clone = board.copy()
            result = clone.apply_move(mv)
            if result is not None:
                # Capturing bishop → immediate win/loss
                score = 1e6 if result == color else -1e6
            else:
                v = self.evaluate_state(clone)
                score = v if color == 'w' else -v
            scores.append(score)

        best = max(scores)
        worst = min(scores)
        if abs(best - worst) < 1e-9:
            return random.choice(candidate_moves)

        norms = [(s - worst) / (best - worst) for s in scores]
        filtered: List[Tuple[Move, float]] = []
        for mv, n in zip(candidate_moves, norms):
            if n > 0.8:
                weight = (n - 0.8) / 0.2
                filtered.append((mv, weight))
        if not filtered:
            return random.choice(candidate_moves)

        total = sum(w for _, w in filtered)
        r = random.random() * total
        cum = 0.0
        for mv, w in filtered:
            cum += w
            if r <= cum:
                return mv
        return filtered[-1][0]


###############################################################################
#  Environment wrapper
###############################################################################

class BattledanceEnvironment:
    def __init__(self) -> None:
        self.initial_fen = "rglbblgr/pfnkknfp/8/8/8/8/PFNKKNFP/RGLBBLGR w - - 0 1"
        # Pre-parse once; games start from copies of this board
        self.initial_board = BattledanceBoard(self.initial_fen)

    def play_game(self, agent1: Agent, agent2: Agent) -> int:
        """Play a game and return result from agent1’s (White’s) perspective (+1/-1/0)."""
        board = self.initial_board.copy()
        players = {'w': agent1, 'b': agent2}
        while True:
            color = board.turn
            agent = players[color]

            mv = agent.choose_move(board)
            if mv is None:
                # no legal moves; stalemate = loss for side to move
                loser = color
                return 1 if loser == 'b' else -1

            winner = board.apply_move(mv)
            if winner is not None:
                return 1 if winner == 'w' else -1

            if board.check_draw():
                return 0


###############################################################################
#  Population and snapshot helpers
###############################################################################

def save_parents(parents: List[Agent], filepath: str) -> None:
    """
    Save a list of parent agents to disk.

    In normal operation this will be a list of 8 parents, but the
    function does not enforce the length.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    serial = []
    for agent in parents:
        serial.append({
            'name': agent.name,
            'trainable': False,
            'net': agent.net.to_dict(),
        })
    with open(filepath, 'wb') as f:
        pickle.dump(serial, f)


def load_parents(filepath: str) -> List[Agent]:
    try:
        with open(filepath, 'rb') as f:
            serial = pickle.load(f)
    except Exception:
        return []

    if not isinstance(serial, list):
        return []  # not a parents file

    parents: List[Agent] = []
    for data in serial:
        if not isinstance(data, dict) or 'net' not in data:
            continue
        agent = Agent(data.get('name', 'unknown'))
        agent.trainable = False
        agent.net = MLP.from_dict(data['net'])
        parents.append(agent)
    return parents


def save_champion(agent: Agent, filepath: str) -> None:
    """Save a single champion agent to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        'name': agent.name,
        'trainable': False,
        'net': agent.net.to_dict(),
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_champion(filepath: str) -> Optional[Agent]:
    """
    Load a champion agent from a snapshot file.

    This is tolerant of both storage formats:
      * Single-champion snapshot: a dict with keys {'name', 'net', ...}.
      * Parent snapshot: a list of 8 dicts, where element 0 is the champion.

    Returns None if the file cannot be read or does not match either format.
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return None

    # Case 1: single champion dict
    if isinstance(data, dict) and 'net' in data:
        name = data.get('name', 'unknown')
        agent = Agent(name)
        agent.trainable = False
        agent.net = MLP.from_dict(data['net'])
        return agent

    # Case 2: list of parent dicts (8 parents); first entry is champion
    if isinstance(data, list) and data:
        champ_data = data[0]
        if isinstance(champ_data, dict) and 'net' in champ_data:
            name = champ_data.get('name', 'unknown')
            agent = Agent(name)
            agent.trainable = False
            agent.net = MLP.from_dict(champ_data['net'])
            return agent

    # Unknown format
    return None

def save_population_state(name: str, population: List[Agent], gen: int, base_dir: str) -> None:
    """
    Persist the full GA population for `name` into the `_0` slot.

    While an agent is in state "pending" for the current cycle, its
    `{name}_0.pkl` file contains this population snapshot (kind="population").
    Once training succeeds for that agent, `_0` is overwritten with an
    8-parent list via `save_parents`.
    """
    path = os.path.join(base_dir, f"{name}_0.pkl")
    serial = {
        "kind": "population",
        "generation": int(gen),
        "agents": [
            {
                "name": agent.name,
                "trainable": agent.trainable,
                "net": agent.net.to_dict(),
            }
            for agent in population
        ],
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(serial, f)
    except Exception:
        # Population snapshots are for resumability only; never crash training.
        pass


def load_population_state(name: str, base_dir: str) -> Optional[Tuple[List[Agent], int]]:
    """
    Load a saved GA population for `name` from the `_0` slot, if present.

    Returns (population, generation) if `_0` currently stores a
    population snapshot (kind="population"), otherwise returns None.
    """
    path = os.path.join(base_dir, f"{name}_0.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return None

    if not (isinstance(data, dict) and data.get("kind") == "population" and "agents" in data):
        return None

    gen = int(data.get("generation", 0))
    agents_data = data["agents"]
    population: List[Agent] = []
    for ad in agents_data:
        agent = Agent(ad.get("name", "unknown"))
        agent.trainable = bool(ad.get("trainable", True))
        net_dict = ad.get("net")
        if isinstance(net_dict, dict):
            agent.net = MLP.from_dict(net_dict)
        population.append(agent)
    return population, gen


def normalise(values: Sequence[float]) -> List[float]:
    """Normalise a sequence of values to [0,1]; if all equal, return 1 for all."""
    if not values:
        return []
    best = max(values)
    worst = min(values)
    if best == worst:
        return [1.0 for _ in values]
    denom = float(best - worst)
    return [float(v - worst) / denom for v in values]


def evaluate_population_stage1(
    pop: List[Agent],
    opponents: List[Agent],
    env: BattledanceEnvironment,
    n_per_colour: int,
) -> Tuple[List[int], List[int], List[List[Tuple[int, int]]]]:
    """
    Stage-1 population evaluation with per-game resumability.

    For each agent in `pop` and each opponent in `opponents`, play
    `n_per_colour` games as White and `n_per_colour` games as Black,
    accumulating (margin, draws) from the agent's perspective.

    Progress is checkpointed after every single Battledance game into
    ga_progress_<Agent>.json so that at most one game has to be replayed
    after interruption.
    """
    n = len(pop)
    m = len(opponents)

    # Trivial cases: no population or no opponents or no games per colour.
    if n == 0 or n_per_colour <= 0 or m == 0:
        sum_margins = [0] * n
        min_margins = [0] * n
        snapshot_stats: List[List[Tuple[int, int]]] = [[(0, 0) for _ in range(m)] for _ in range(n)]
        return sum_margins, min_margins, snapshot_stats

    # ------------------------------------------------------------------
    # Load or initialise GA Stage-1 progress
    # ------------------------------------------------------------------
    progress = load_ga_progress()
    valid = False
    if progress is not None:
        if (
            progress.get("cycle") == CURRENT_CYCLE
            and progress.get("agent") == CURRENT_AGENT_NAME
            and progress.get("generation") == CURRENT_GEN
            and progress.get("n_candidates") == n
            and progress.get("n_opponents") == m
            and progress.get("n1") == n_per_colour
        ):
            valid = True

    if not valid:
        # Fresh Stage-1 progress
        margins = [[0 for _ in range(m)] for _ in range(n)]
        draws = [[0 for _ in range(m)] for _ in range(n)]
        progress = {
            "cycle": CURRENT_CYCLE,
            "agent": CURRENT_AGENT_NAME,
            "generation": CURRENT_GEN,
            "n_candidates": n,
            "n_opponents": m,
            "n1": n_per_colour,
            # Default n2; Stage-2 may overwrite/extend this.
            "n2": 8,
            "stage": "stage1",
            "stage1": {
                "game_index": 0,
                "margins": margins,
                "draws": draws,
            },
        }
        save_ga_progress(progress)
    else:
        # Ensure Stage-1 arrays exist even if Stage-2 has already started
        s1 = progress.get("stage1") or {}
        margins = s1.get("margins")
        draws = s1.get("draws")
        if margins is None or draws is None or len(margins) != n or any(len(row) != m for row in margins):
            margins = [[0 for _ in range(m)] for _ in range(n)]
            draws = [[0 for _ in range(m)] for _ in range(n)]
            s1["margins"] = margins
            s1["draws"] = draws
            s1.setdefault("game_index", 0)
            progress["stage1"] = s1
            progress.setdefault("stage", "stage1")
            save_ga_progress(progress)

    stage = progress.get("stage", "stage1")
    s1 = progress["stage1"]
    game_index = int(s1.get("game_index", 0) or 0)

    games_per_pair = 2 * n_per_colour
    total_pairs = n * m
    total_games = total_pairs * games_per_pair

    # ------------------------------------------------------------------
    # If Stage-1 already complete, reconstruct sums and stats without
    # re-playing any games.
    # ------------------------------------------------------------------
    if stage in ("stage1_complete", "stage2", "done") and game_index >= total_games:
        sum_margins = [sum(row) for row in margins]
        min_margins: List[int] = []
        for i in range(n):
            row = margins[i]
            min_margins.append(min(row) if row else 0)
        snapshot_stats: List[List[Tuple[int, int]]] = []
        for i in range(n):
            row_stats = []
            for j in range(m):
                row_stats.append((margins[i][j], draws[i][j]))
            snapshot_stats.append(row_stats)
        return sum_margins, min_margins, snapshot_stats

    # ------------------------------------------------------------------
    # Stage-1: play/resume all games with per-game checkpointing
    # ------------------------------------------------------------------
    while game_index < total_games:
        pair_index = game_index // games_per_pair
        i = pair_index // m
        j = pair_index % m
        offset = game_index % games_per_pair
        colour_block = offset // n_per_colour  # 0 = agent as White, 1 = agent as Black

        agent = pop[i]
        opp = opponents[j]

        # Status line
        current_game = game_index + 1
        if CURRENT_AGENT_NAME:
            role = "W" if colour_block == 0 else "B"
            status(
                f"[{CURRENT_AGENT_NAME}] cyc {CURRENT_CYCLE} gen {CURRENT_GEN} "
                f"stage1 game {current_game}/{total_games} "
                f"(agent {i+1}/{n} vs {opp.name}, {role})"
            )

        # Play a single Battledance game and update margin/draws.
        if colour_block == 0:
            # Agent as White
            res = env.play_game(agent, opp)
            if res == 1:
                margins[i][j] += 1
            elif res == -1:
                margins[i][j] -= 1
            else:
                draws[i][j] += 1
        else:
            # Agent as Black
            res = env.play_game(opp, agent)
            # From agent's perspective: invert result
            if res == 1:       # opponent (as White) wins
                margins[i][j] -= 1
            elif res == -1:    # opponent loses
                margins[i][j] += 1
            else:
                draws[i][j] += 1

        # Advance global Stage-1 game index and checkpoint progress.
        game_index += 1
        s1["game_index"] = game_index
        s1["margins"] = margins
        s1["draws"] = draws
        progress["stage1"] = s1
        progress["stage"] = "stage1"
        save_ga_progress(progress)

    # Clean newline after the status bar
    print()

    # ------------------------------------------------------------------
    # Stage-1 finished: compute sum/min and snapshot stats, mark complete
    # ------------------------------------------------------------------
    sum_margins = [sum(row) for row in margins]
    min_margins: List[int] = []
    for i in range(n):
        row = margins[i]
        min_margins.append(min(row) if row else 0)

    snapshot_stats: List[List[Tuple[int, int]]] = []
    for i in range(n):
        row_stats = []
        for j in range(m):
            row_stats.append((margins[i][j], draws[i][j]))
        snapshot_stats.append(row_stats)

    s1["sum_margins"] = sum_margins
    s1["min_margins"] = min_margins
    s1["snapshot_stats"] = snapshot_stats
    progress["stage1"] = s1
    progress["stage"] = "stage1_complete"
    save_ga_progress(progress)

    return sum_margins, min_margins, snapshot_stats


def refine_top_candidates(
    pop: List[Agent],
    opponents: List[Agent],
    env: BattledanceEnvironment,
    top_indices: List[int],
    snapshot_stats_stage1: List[List[Tuple[int, int]]],
    n1_per_colour: int,
    n2_per_colour: int,
    use_worst_only: bool = False,
) -> Tuple[List[int], dict[int, int], dict[int, int], dict[int, List[Tuple[int, int]]]]:
    """
    Stage-2 heavy evaluation on a small subset of candidates with per-game
    resumability.

    Reuses Stage-1 results at `n1_per_colour`, adds the remaining games to
    reach `n2_per_colour` per colour for the `top_indices` subset, and
    computes final metrics.

    All Stage-2 games are checkpointed into ga_progress_<Agent>.json so
    that at most one Battledance game is repeated after interruption.
    """
    if n2_per_colour < n1_per_colour:
        raise ValueError("n2_per_colour must be >= n1_per_colour")

    m = len(opponents)
    additional = n2_per_colour - n1_per_colour

    # No extra work required if no finalists, no opponents, or no additional games.
    if not top_indices or m == 0 or additional <= 0:
        final_sum_margins: dict[int, int] = {}
        final_min_margins: dict[int, int] = {}
        snapshot_stats_stage2: dict[int, List[Tuple[int, int]]] = {}
        # Just treat Stage-1 numbers as Stage-2 numbers for the subset.
        for idx in top_indices:
            row = snapshot_stats_stage1[idx]
            ssum = sum(margin for (margin, _) in row)
            smin = min((margin for (margin, _) in row), default=0)
            final_sum_margins[idx] = ssum
            final_min_margins[idx] = smin
            snapshot_stats_stage2[idx] = list(row)

        ordered_indices = list(top_indices)
        if use_worst_only:
            ordered_indices.sort(
                key=lambda i: (final_min_margins.get(i, 0), final_sum_margins.get(i, 0), -i),
                reverse=True,
            )
        else:
            sums = [final_sum_margins[i] for i in ordered_indices]
            mins = [final_min_margins[i] for i in ordered_indices]
            N1 = normalise(sums)
            N2 = normalise(mins)
            fitness2 = {idx: N1[k] * N2[k] for k, idx in enumerate(ordered_indices)}
            ordered_indices.sort(key=lambda i: (fitness2.get(i, 0.0), i), reverse=True)

        return ordered_indices, final_sum_margins, final_min_margins, snapshot_stats_stage2

    # ------------------------------------------------------------------
    # Load/initialise GA Stage-2 progress
    # ------------------------------------------------------------------
    progress = load_ga_progress()
    n = len(pop)

    # We assume Stage-1 was run via evaluate_population_stage1, which
    # stored margins/draws arrays under progress["stage1"].
    if not progress:
        raise RuntimeError("GA Stage-2 progress missing; Stage-1 was not recorded.")

    if (
        progress.get("cycle") != CURRENT_CYCLE
        or progress.get("agent") != CURRENT_AGENT_NAME
        or progress.get("generation") != CURRENT_GEN
        or progress.get("n_candidates") != n
        or progress.get("n_opponents") != m
        or progress.get("n1") != n1_per_colour
        or progress.get("n2") != n2_per_colour
    ):
        # Mis-match: safest is to restart Stage-2 from scratch (Stage-1
        # results are available through snapshot_stats_stage1).
        s1_margins = [[margin for (margin, _) in row] for row in snapshot_stats_stage1]
        s1_draws = [[draws for (_, draws) in row] for row in snapshot_stats_stage1]
        progress = {
            "cycle": CURRENT_CYCLE,
            "agent": CURRENT_AGENT_NAME,
            "generation": CURRENT_GEN,
            "n_candidates": n,
            "n_opponents": m,
            "n1": n1_per_colour,
            "n2": n2_per_colour,
            "stage": "stage2",
            "stage1": {
                "margins": s1_margins,
                "draws": s1_draws,
                "game_index": 2 * n1_per_colour * n * m,
                "sum_margins": [sum(row) for row in s1_margins],
                "min_margins": [
                    min(row) if row else 0
                    for row in s1_margins
                ],
                "snapshot_stats": snapshot_stats_stage1,
            },
            "stage2": {
                "game_index": 0,
                "top_indices": list(top_indices),
                # per-candidate, per-opponent margins/draws at full n2 budget
                "snapshot_margins": {
                    str(idx): [margin for (margin, _) in snapshot_stats_stage1[idx]]
                    for idx in top_indices
                },
                "snapshot_draws": {
                    str(idx): [draws for (_, draws) in snapshot_stats_stage1[idx]]
                    for idx in top_indices
                },
            },
        }
        save_ga_progress(progress)

    stage = progress.get("stage", "stage1_complete")

    # Ensure Stage-1 snapshot stats are present (for candidates outside top_indices)
    if "stage1" not in progress or "snapshot_stats" not in progress["stage1"]:
        s1_margins = [[margin for (margin, _) in row] for row in snapshot_stats_stage1]
        s1_draws = [[draws for (_, draws) in row] for row in snapshot_stats_stage1]
        progress.setdefault("stage1", {})
        progress["stage1"]["margins"] = s1_margins
        progress["stage1"]["draws"] = s1_draws
        progress["stage1"]["snapshot_stats"] = snapshot_stats_stage1
        save_ga_progress(progress)

    s1_snapshot_stats: List[List[Tuple[int, int]]] = progress["stage1"]["snapshot_stats"]

    # Initialise Stage-2 section if not present or if we are just leaving Stage-1.
    if stage in ("stage1_complete",) or "stage2" not in progress:
        snapshot_margins_stage2: Dict[str, List[int]] = {
            str(idx): [margin for (margin, _) in s1_snapshot_stats[idx]]
            for idx in top_indices
        }
        snapshot_draws_stage2: Dict[str, List[int]] = {
            str(idx): [draws for (_, draws) in s1_snapshot_stats[idx]]
            for idx in top_indices
        }
        progress["stage2"] = {
            "game_index": 0,
            "top_indices": list(top_indices),
            "snapshot_margins": snapshot_margins_stage2,
            "snapshot_draws": snapshot_draws_stage2,
        }
        progress["stage"] = "stage2"
        save_ga_progress(progress)

    s2 = progress["stage2"]
    # Ensure top_indices match
    if s2.get("top_indices") != list(top_indices):
        # If mismatch, safest is to restart Stage-2 completely with the given top_indices
        snapshot_margins_stage2 = {
            str(idx): [margin for (margin, _) in s1_snapshot_stats[idx]]
            for idx in top_indices
        }
        snapshot_draws_stage2 = {
            str(idx): [draws for (_, draws) in s1_snapshot_stats[idx]]
            for idx in top_indices
        }
        s2 = {
            "game_index": 0,
            "top_indices": list(top_indices),
            "snapshot_margins": snapshot_margins_stage2,
            "snapshot_draws": snapshot_draws_stage2,
        }
        progress["stage2"] = s2
        progress["stage"] = "stage2"
        save_ga_progress(progress)

    snapshot_margins_stage2: Dict[str, List[int]] = s2["snapshot_margins"]
    snapshot_draws_stage2: Dict[str, List[int]] = s2["snapshot_draws"]
    game_index = int(s2.get("game_index", 0) or 0)

    T = len(top_indices)
    games_per_pair = 2 * additional
    total_pairs = T * m
    total_games = total_pairs * games_per_pair

    # ------------------------------------------------------------------
    # If Stage-2 already finished, reconstruct final metrics from stored
    # arrays without replaying games.
    # ------------------------------------------------------------------
    if stage == "done" and game_index >= total_games and "final_sum_margins" in s2 and "final_min_margins" in s2:
        final_sum_margins = {int(k): v for k, v in s2["final_sum_margins"].items()}
        final_min_margins = {int(k): v for k, v in s2["final_min_margins"].items()}
        snapshot_stats_stage2: dict[int, List[Tuple[int, int]]] = {}
        for idx in top_indices:
            row_m = snapshot_margins_stage2[str(idx)]
            row_d = snapshot_draws_stage2[str(idx)]
            snapshot_stats_stage2[idx] = list(zip(row_m, row_d))

        ordered_indices = list(top_indices)
        if use_worst_only:
            ordered_indices.sort(
                key=lambda i: (final_min_margins.get(i, 0), final_sum_margins.get(i, 0), -i),
                reverse=True,
            )
        else:
            sums = [final_sum_margins[i] for i in ordered_indices]
            mins = [final_min_margins[i] for i in ordered_indices]
            N1 = normalise(sums)
            N2 = normalise(mins)
            fitness2 = {idx: N1[k] * N2[k] for k, idx in enumerate(ordered_indices)}
            ordered_indices.sort(key=lambda i: (fitness2.get(i, 0.0), i), reverse=True)

        return ordered_indices, final_sum_margins, final_min_margins, snapshot_stats_stage2

    # ------------------------------------------------------------------
    # Stage-2: play/resume additional heavy games with per-game checkpoint
    # ------------------------------------------------------------------
    while game_index < total_games:
        pair_index = game_index // games_per_pair
        t_idx = pair_index // m      # index in top_indices list
        j = pair_index % m           # opponent index
        offset = game_index % games_per_pair
        colour_block = offset // additional  # 0 = agent as White, 1 = agent as Black

        idx = top_indices[t_idx]
        agent = pop[idx]
        opp = opponents[j]

        # Status
        current_game = game_index + 1
        role = "W" if colour_block == 0 else "B"
        status(
            f"[{CURRENT_AGENT_NAME}] cyc {CURRENT_CYCLE} gen {CURRENT_GEN} "
            f"stage2 game {current_game}/{total_games} "
            f"(cand_idx {idx} vs {opp.name}, {role})"
        )

        # Play a single Battledance game and update Stage-2 snapshot stats
        row_m = snapshot_margins_stage2[str(idx)]
        row_d = snapshot_draws_stage2[str(idx)]

        if colour_block == 0:
            # Candidate as White
            res = env.play_game(agent, opp)
            if res == 1:
                row_m[j] += 1
            elif res == -1:
                row_m[j] -= 1
            else:
                row_d[j] += 1
        else:
            # Candidate as Black
            res = env.play_game(opp, agent)
            if res == 1:
                row_m[j] -= 1
            elif res == -1:
                row_m[j] += 1
            else:
                row_d[j] += 1

        # Write back and checkpoint progress
        snapshot_margins_stage2[str(idx)] = row_m
        snapshot_draws_stage2[str(idx)] = row_d
        s2["snapshot_margins"] = snapshot_margins_stage2
        s2["snapshot_draws"] = snapshot_draws_stage2

        game_index += 1
        s2["game_index"] = game_index
        progress["stage2"] = s2
        progress["stage"] = "stage2"
        save_ga_progress(progress)

    # Clean newline after status line
    print()

    # ------------------------------------------------------------------
    # Stage-2 finished: compute final metrics for top_indices
    # ------------------------------------------------------------------
    final_sum_margins: dict[int, int] = {}
    final_min_margins: dict[int, int] = {}
    snapshot_stats_stage2: dict[int, List[Tuple[int, int]]] = {}

    for idx in top_indices:
        row_m = snapshot_margins_stage2[str(idx)]
        row_d = snapshot_draws_stage2[str(idx)]
        snapshot_stats_stage2[idx] = list(zip(row_m, row_d))
        ssum = sum(row_m)
        smin = min(row_m) if row_m else 0
        final_sum_margins[idx] = ssum
        final_min_margins[idx] = smin

    # Persist final metrics for future resumes (if any)
    s2["final_sum_margins"] = {str(k): v for k, v in final_sum_margins.items()}
    s2["final_min_margins"] = {str(k): v for k, v in final_min_margins.items()}
    progress["stage2"] = s2
    progress["stage"] = "done"
    save_ga_progress(progress)

    # ------------------------------------------------------------------
    # Build ordering within the finalists by Stage-2 fitness
    # ------------------------------------------------------------------
    ordered_indices = list(top_indices)
    if not ordered_indices:
        return ordered_indices, final_sum_margins, final_min_margins, snapshot_stats_stage2

    if use_worst_only:
        ordered_indices.sort(
            key=lambda i: (final_min_margins.get(i, 0), final_sum_margins.get(i, 0), -i),
            reverse=True,
        )
    else:
        sums = [final_sum_margins[i] for i in ordered_indices]
        mins = [final_min_margins[i] for i in ordered_indices]
        N1 = normalise(sums)
        N2 = normalise(mins)
        fitness2 = {idx: N1[k] * N2[k] for k, idx in enumerate(ordered_indices)}
        ordered_indices.sort(key=lambda i: (fitness2.get(i, 0.0), i), reverse=True)

    return ordered_indices, final_sum_margins, final_min_margins, snapshot_stats_stage2


def build_children(parents: List[Agent], rnd: np.random.RandomState) -> List[Agent]:
    """
    Generate 256 non-elite children from 8 parents using a uniform
    crossover grid:

      * For each ordered pair (i, j) with i,j in {0..7}, generate 4 children.
      * Each child is produced by crossover(p_i, p_j) plus mutation.

    Total: 8 * 8 * 4 = 256 children. Elites are handled separately.
    """
    if len(parents) != 8:
        raise ValueError(f"build_children expects exactly 8 parents, got {len(parents)}")

    children: List[Agent] = []
    for i in range(8):
        for j in range(8):
            p1 = parents[i]
            p2 = parents[j]
            for _ in range(4):
                child_net = p1.net.crossover(p2.net, rnd)
                child_net.mutate(mutation_rate=0.03125, weight_decay=0.0005, rnd=rnd)
                child = Agent(p1.name)
                child.net = child_net
                child.trainable = True
                children.append(child)

    # Sanity: 256 children
    assert len(children) == 256, f"Expected 256 children, got {len(children)}"
    return children


def build_population_from_parents(
    name: str,
    parents: List[Agent],
    rnd: np.random.RandomState,
) -> List[Agent]:
    """
    Given an 8-net parent set, build a 260-net GA population:

      * 256 non-elite children produced via crossover/mutation of all 8 parents.
      * 4 elite direct clones of parents[0..3].

    The 4 elites are preserved as-is (never mutated) and are appended to
    the population for evaluation. The remaining 4 parents are used only
    as genetic sources when generating children; they are not injected
    into the evaluation population.
    """
    if not parents:
        raise ValueError("build_population_from_parents: empty parent list")

    # Use at most the first 8 parents; this should always be exactly 8.
    if len(parents) > 8:
        parents = parents[:8]

    # 256 mutated children from the full 8-parent grid
    children = build_children(parents, rnd)

    pop: List[Agent] = []
    pop.extend(children)

    # 4 elites as direct clones of the top four parents
    elites: List[Agent] = []
    for i in range(min(4, len(parents))):
        elite = parents[i].clone()
        elite.trainable = True
        elites.append(elite)

    pop.extend(elites)

    # pop size is now 256 + 4 = 260
    return pop


def train_population_once(
    pop: List[Agent],
    opponents: List[Agent],
    env: BattledanceEnvironment,
    rnd: np.random.RandomState,
    use_worst_only: bool = False,
) -> Tuple[List[Agent], List[Agent], bool]:
    """
    Perform one evolutionary generation on `pop` using a two-stage
    evaluation scheme with reuse of the initial games.

    Stage 1:
      * Evaluate the whole population vs all opponent snapshots with
        n1 = 2 games per colour (4 per snapshot).
      * Compute a coarse fitness and select the top K candidates
        (default K = 12) as heavy-eval finalists.

    Stage 2:
      * For those K finalists, play the remaining games needed to reach
        n2 = 16 per colour (32 per snapshot), reusing the Stage 1
        results so the final stats correspond to a full n2 budget.
      * Rank the finalists by Stage-2 fitness and pick the top 8 as
        parents. parents[0..3] are elites; parents[4..7] are additional
        genetic sources.

    Success criterion:
      * A generation is considered successful if each of the selected
        parents has margin >= 0 vs every opponent snapshot at the full
        n2 budget.
    """
    if not pop:
        return pop, [], False

    n_candidates = len(pop)
    n1 = 2   # cheap pass: 4 games per snapshot
    n2 = 16  # heavy pass: 32 games per snapshot

    # ------------------------------------------------------------------
    # Stage 1: cheap evaluation for the whole population
    # ------------------------------------------------------------------
    sum_margins_1, min_margins_1, snapshot_stats_1 = evaluate_population_stage1(
        pop,
        opponents,
        env,
        n_per_colour=n1,
    )

    # Build Stage-1 fitness scores
    if use_worst_only:
        fitness_base_1 = normalise(min_margins_1)
    else:
        N1 = normalise(sum_margins_1)
        N2 = normalise(min_margins_1)
        raw = [N1[i] * N2[i] for i in range(n_candidates)]
        fitness_base_1 = normalise(raw)

    indices = list(range(n_candidates))
    indices.sort(key=lambda i: (-fitness_base_1[i], i))

    # Top K finalists for heavy evaluation
    K = min(12, n_candidates)
    top_indices = indices[:K]

    # ------------------------------------------------------------------
    # Stage 2: heavy evaluation for the top K, adding 14/colour to reach 32
    # ------------------------------------------------------------------
    ordered_indices, sum_margins_2, min_margins_2, snapshot_stats_2 = refine_top_candidates(
        pop,
        opponents,
        env,
        top_indices,
        snapshot_stats_1,
        n1_per_colour=n1,
        n2_per_colour=n2,
        use_worst_only=use_worst_only,
    )

    if not ordered_indices:
        # Should not happen, but be defensive.
        return pop, [], False

    # Parents = top 8 of Stage-2 ordering (or fewer if population is tiny)
    parent_count = min(8, len(ordered_indices))
    parent_indices = ordered_indices[:parent_count]
    parents: List[Agent] = [pop[i].clone() for i in parent_indices]

    if not parents:
        return pop, [], False

    # ------------------------------------------------------------------
    # Success gate: each parent must achieve margin >= 0 vs every
    # opponent snapshot at full n2 resolution.
    # ------------------------------------------------------------------
    success = True
    parent_metrics: List[List[Tuple[str, int, int]]] = []

    for local_rank, idx in enumerate(parent_indices):
        row_stats = snapshot_stats_2.get(idx, [])
        row: List[Tuple[str, int, int]] = []
        for snap_idx, (margin, draws) in enumerate(row_stats):
            opp_name = opponents[snap_idx].name if snap_idx < len(opponents) else f"opp_{snap_idx}"
            row.append((opp_name, margin, draws))
            if margin < 0:
                success = False
                break
        parent_metrics.append(row)
        if not success:
            break

    # Log a compact summary on success.
    if success and parents:
        header = (
            f"[{CURRENT_AGENT_NAME}] cycle {CURRENT_CYCLE} gen {CURRENT_GEN}: "
            f"GA success (two-stage) with {len(parents)} parents vs {len(opponents)} opponents."
        )
        log(header)
        for local_idx, row in enumerate(parent_metrics):
            parts = [f"  parent[{local_idx}] idx={parent_indices[local_idx]}:"]
            for opp_name, margin, draws in row:
                parts.append(f"{opp_name}=margin{margin},draws{draws}")
            # Keep these in the file log only; stdout would get noisy.
            log(" ".join(parts), also_print=False)

    # ------------------------------------------------------------------
    # Build next generation: 256 children + 4 elites → 260 nets
    # ------------------------------------------------------------------
    new_pop = build_population_from_parents(parents[0].name, parents, rnd)
    return new_pop, parents, success


###############################################################################
#  Champion matches and rotation
###############################################################################

def run_champion_matches(
    name: str,
    parents: List[Agent],
    opponent_lists: Dict[str, List[str]],
    base_dir: str,
    cycle: int,
    env: BattledanceEnvironment,
    outfile_path: str,
) -> None:
    """
    Run champion matches for a single agent `name` immediately after its GA
    training completes, with per-agent logging and per-agent resumability.

    Champion of `name` is parents[0]. For each Opp in opponent_lists[name],
    and for each snapshot index s ∈ {1,2,3}, we attempt to load the champion
    from Opp_s and, if successful, schedule two games (champ as White, champ
    as Black).

    Checkpointing semantics (per agent):

      * champion_progress_<name>.json tracks, for this cycle, how many
        champion-match games have already been completed.
      * Games are flattened into a linear schedule of length T:
            [ (Opp1, s1, W), (Opp1, s1, B),
              (Opp1, s2, W), (Opp1, s2, B),
              ...
            ]
        The index in this schedule is `game_index`.
      * progress["next_index"] = n means that games [0..n-1] have been fully
        completed and logged for this agent in this cycle.

    Logging:

      * Each agent logs to its own file: champion_matches_<Name>.txt
        (two lines per game: header + moves).
      * On resume, we truncate that log to the first `next_index` games
        to drop any partial tails.
    """
    if not parents:
        log(f"[{name}] cycle {cycle}: no parents available for champion matches.")
        return

    champ = parents[0]

    # Preload all usable opponent snapshots once and build the schedule.
    snapshot_agents: Dict[Tuple[str, int], Agent] = {}
    schedule: List[Tuple[str, int, str]] = []

    for opp_name in opponent_lists.get(name, []):
        for s in (1, 2, 3):
            opp_path = os.path.join(base_dir, f"{opp_name}_{s}.pkl")
            if not os.path.exists(opp_path):
                continue

            opp_champ = load_champion(opp_path)
            if opp_champ is None:
                plist = load_parents(opp_path)
                if not plist:
                    continue
                opp_champ = plist[0]

            snapshot_agents[(opp_name, s)] = opp_champ
            schedule.append((opp_name, s, "W"))
            schedule.append((opp_name, s, "B"))

    total_games = len(schedule)
    if total_games == 0:
        log(f"[{name}] cycle {cycle}: no opponent snapshots available for champion matches.")
        print()
        return

    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)

    # Load per-agent champion progress for this cycle
    progress = load_champion_progress(base_dir, cycle, name)
    next_index = int(progress.get("next_index", 0) or 0)
    if next_index < 0:
        next_index = 0
    if next_index > total_games:
        next_index = total_games
    progress["cycle"] = cycle
    progress["next_index"] = next_index
    save_champion_progress(base_dir, name, progress)

    # Truncate this agent's log file to the first `next_index` complete games
    existing: List[str] = []
    if os.path.exists(outfile_path):
        try:
            with open(outfile_path, "r", encoding="utf-8") as f:
                existing = f.read().splitlines()
        except Exception:
            existing = []

    # Two lines per complete game: header + moves
    max_pairs = len(existing) // 2
    completed_pairs = min(next_index, max_pairs)
    trimmed: List[str] = existing[: 2 * completed_pairs]

    try:
        with open(outfile_path, "w", encoding="utf-8") as f:
            if trimmed:
                f.write("\n".join(trimmed) + "\n")
    except Exception:
        # Logging must not crash training
        pass

    # Replay / resume over the schedule.
    game_index = 0
    for opp_name, s, color in schedule:
        # Skip games already recorded as completed.
        if game_index < next_index:
            game_index += 1
            continue

        opp_champ = snapshot_agents[(opp_name, s)]

        # Status line for this game
        status(
            f"[{name}] cyc {cycle} champ game {game_index + 1}/{total_games} "
            f"vs {opp_name}_{s} ({color})"
        )

        # Play game with correct colour assignment
        if color == "W":
            result_raw, moves_str = play_game_with_moves(champ, opp_champ, env)
            result = result_raw
        else:
            result_raw, moves_str = play_game_with_moves(opp_champ, champ, env)
            result = -result_raw

        header = (
            f"Cycle: {cycle} Name: {name} Opp: {opp_name} "
            f"Snapshot: {s} ChampColor: {color} Result: {result}"
        )

        # Append header + moves to the per-agent log (append-only, per-game)
        try:
            with open(outfile_path, "a", encoding="utf-8") as f:
                f.write(header + "\n")
                f.write(moves_str + "\n")
        except Exception:
            # Logging must not crash training; still advance progress.
            pass

        # Silent detailed log line
        log(
            f"[{name}] cycle {cycle}: champ vs {opp_name}_{s} as {color} -> result {result}",
            also_print=False,
        )

        game_index += 1

        # Update champion_progress after each completed game
        progress["next_index"] = game_index
        save_champion_progress(base_dir, name, progress)

    # Clean up the status line with a newline at the end of champion matches
    print()


def play_game_with_moves(agent1: Agent, agent2: Agent, env: BattledanceEnvironment) -> Tuple[int, str]:
    """
    Play a single game between agents and return (result, moves_str).

    result:
      +1 = agent1 (as White) wins
      -1 = agent2 (as Black) wins
       0 = draw

    moves_str:
      comma-separated list of moves in notation like "Pc2-c4" or "P-@-b2" for drops.
    """
    board = env.initial_board.copy()
    players = {'w': agent1, 'b': agent2}
    moves_list: List[str] = []

    while True:
        color = board.turn
        agent = players[color]

        mv = agent.choose_move(board)
        if mv is None:
            # no legal moves (stalemate is loss for side to move)
            loser = color
            result = 1 if loser == 'b' else -1
            break

        # Record move in notation
        if mv.kind == 'move':
            fr, fc = mv.from_pos
            tr, tc = mv.to_pos
            move_str = f"{board.board[fr][fc].kind}{chr(97 + fc)}{8 - fr}-{chr(97 + tc)}{8 - tr}"
        else:
            # drop notation: X-@-sq
            tr, tc = mv.to_pos
            move_str = f"{mv.drop_type}-@-{chr(97 + tc)}{8 - tr}"
        moves_list.append(move_str)

        winner = board.apply_move(mv)
        if winner is not None:
            result = 1 if winner == 'w' else -1
            break
        if board.check_draw():
            result = 0
            break

    return result, ', '.join(moves_list)

###############################################################################
#  Per-agent training helper (for single- and multi-thread modes)
###############################################################################

def train_single_agent(
    name: str,
    opponent_lists: Dict[str, List[str]],
    base_dir: str,
    cycle: int,
    seed: int,
) -> int:
    """
    Train a single agent `name` for the given cycle until GA success.

    This is a per-agent wrapper around the old run_training_cycle
    per-agent block. It:
      * resumes GA from a saved population if available,
      * otherwise seeds from existing parents or Xavier,
      * trains until a parent set passes,
      * saves parents to Name_0.pkl,
      * resets GA progress for this agent,
      * writes a GA 'done' marker so later resumes can skip GA entirely.

    It does NOT touch cycle_progress or champion_progress; the caller
    is responsible for updating those and running champion matches.
    Returns the final generation index for this agent.
    """
    global CURRENT_AGENT_NAME, CURRENT_GEN, CURRENT_CYCLE, GA_BASE_DIR

    CURRENT_AGENT_NAME = name
    CURRENT_CYCLE = cycle
    GA_BASE_DIR = base_dir

    # Fast path: GA already completed for this agent in this cycle.
    done_state = load_ga_done_state(base_dir, name)
    if done_state is not None and done_state.get("cycle") == cycle:
        # Only trust the marker if the parents snapshot still exists.
        pop_path = os.path.join(base_dir, f"{name}_0.pkl")
        parents_existing = load_parents(pop_path)
        if parents_existing:
            last_gen = int(done_state.get("last_gen", 0) or 0)
            log(
                f"[{name}] cycle {cycle}: GA already marked done at generation {last_gen}; "
                f"skipping GA.",
            )
            return last_gen
        else:
            log(
                f"[{name}] cycle {cycle}: GA done marker present but parents missing; "
                f"ignoring marker and retraining.",
            )

    # Fresh RNG for this agent/process
    rnd = np.random.RandomState(seed)

    env = BattledanceEnvironment()

    # Try to resume GA from a saved full population in `_0`.
    population: List[Agent]
    parents: List[Agent] = []
    gen_start = 0

    pop_state = load_population_state(name, base_dir)
    if pop_state is not None:
        population, gen_start = pop_state
        log(
            f"[{name}] cycle {cycle}: resuming GA from saved population at generation {gen_start}.",
        )
    else:
        # No population snapshot: seed new GA population from parents in `_0` or `_1`,
        # or from fresh Xavier seeds if neither exists.
        parent_path0 = os.path.join(base_dir, f"{name}_0.pkl")
        parent_path1 = os.path.join(base_dir, f"{name}_1.pkl")

        seed_parents = load_parents(parent_path0)
        source_path = parent_path0
        if not seed_parents:
            seed_parents = load_parents(parent_path1)
            source_path = parent_path1

        if seed_parents:
            log(f"[{name}] cycle {cycle}: seeding from existing parents in {source_path}.")
            population = build_population_from_parents(name, seed_parents, rnd)
        else:
            # True "generation 0": 4 Xavier seeds as elites, plus 4 straight copies as other parents.
            log(
                f"[{name}] cycle {cycle}: no prior parents, creating fresh Xavier seed parents "
                f"(4 elites + 4 copies)."
            )
            seeds: List[Agent] = []
            for _ in range(4):
                a = Agent(name)
                a.net = MLP(input_dim=594, hidden_dim=512, seed=rnd.randint(0, 2**31 - 1))
                a.trainable = True
                seeds.append(a)

            parents_seed: List[Agent] = []
            parents_seed.extend(seeds)
            parents_seed.extend([s.clone() for s in seeds])

            population = build_population_from_parents(name, parents_seed, rnd)

        # Newly seeded population is "generation 0" baseline.
        gen_start = 0
        save_population_state(name, population, gen_start, base_dir)

    # Build opponent snapshots for this cycle (read-only; safe in parallel)
    opponents: List[Agent] = []
    for opp_name in opponent_lists.get(name, []):
        for s in (1, 2, 3):
            path = os.path.join(base_dir, f"{opp_name}_{s}.pkl")
            if os.path.exists(path):
                champ = load_champion(path)
                if champ is None:
                    plist = load_parents(path)
                    if plist:
                        champ = plist[0]
                    else:
                        continue
                opponents.append(champ)
    log(f"[{name}] cycle {cycle}: loaded {len(opponents)} opponent snapshots.")

    # Train population until success
    gen = gen_start
    unsuccessful = 0
    success = False
    parents = []

    while not success:
        gen += 1
        CURRENT_GEN = gen
        # Use worst-case-only fitness every 1024 unsuccessful generations
        use_worst = unsuccessful > 0 and unsuccessful % 1024 == 0
        population, parents, success = train_population_once(
            population,
            opponents,
            env,
            rnd,
            use_worst_only=use_worst,
        )

        # Persist the full population to `_0` after each generation so we
        # can resume GA without rebuilding it.
        save_population_state(name, population, gen, base_dir)

        status_str = "success" if success else "continue"
        suffix = " (worst-only fitness)" if use_worst else ""
        log(f"[{name}] cycle {cycle} gen {gen}: {status_str}{suffix}")
        if success:
            break
        unsuccessful += 1

    log(f"[{name}] cycle {cycle}: finished after {gen} generations.")

    # At success: prune `_0` down to the 8 selected parents for this cycle.
    pop_path = os.path.join(base_dir, f"{name}_0.pkl")
    save_parents(parents, pop_path)
    log(f"[{name}] cycle {cycle}: parents saved to {pop_path}")

    # Clear GA progress for this agent now that the generation is complete
    reset_ga_progress()

    # Mark GA as done for this agent and cycle so future resumes can skip GA.
    save_ga_done_state(base_dir, name, cycle, gen)

    return gen


def train_group_agent_sequence(
    group_names: List[str],
    opponent_lists: Dict[str, List[str]],
    base_dir: str,
    cycle: int,
    name_to_seed: Dict[str, int],
) -> None:
    """
    Worker entry point for a single OS process responsible for one group
    of agents. Agents in `group_names` are trained sequentially in the
    given order.

    NEW BEHAVIOUR:
      * For each agent in this group:
          - Run GA training via train_single_agent(...).
          - Immediately run champion matches for that agent, writing to
            champion_matches_<Name>.txt, before moving to the next agent.
      * This ensures per-agent champion matches happen as soon as that
        agent finishes training in this worker.

    The caller ensures that group_names only contains agents that still
    need training for this cycle.
    """
    # Each process gets its own logging header; harmless duplication.
    setup_logging(base_dir, cycle)

    # Single environment reused for champion matches in this process
    env = BattledanceEnvironment()

    for name in group_names:
        seed = name_to_seed.get(name, 0)

        # Train this agent until GA success; saves parents to Name_0.pkl
        train_single_agent(name, opponent_lists, base_dir, cycle, seed)

        # Load parents (8 nets) from Name_0.pkl for champion matches
        pop_path = os.path.join(base_dir, f"{name}_0.pkl")
        parents = load_parents(pop_path)
        if not parents:
            log(
                f"[{name}] cycle {cycle}: WARNING: parents missing after GA success; "
                f"skipping champion matches for this agent.",
            )
            continue

        # Per-agent champion match log
        champ_log_path = os.path.join(base_dir, f"champion_matches_{name}.txt")

        # Run champion matches immediately after training this agent
        run_champion_matches(
            name,
            parents,
            opponent_lists,
            base_dir,
            cycle,
            env,
            champ_log_path,
        )


###############################################################################
#  Main training loop and rotation
###############################################################################

def run_training_cycle(
    agent_names: List[str],
    opponent_lists: Dict[str, List[str]],
    base_dir: str,
    rnd: np.random.RandomState,
    cycle: int,
    thread_mode: str = "3",
) -> bool:
    """
    Run one training cycle across all agents.

    Modes:
      * thread_mode == "1": single-process behaviour (sequential),
        but internally uses train_single_agent for per-agent GA.
      * thread_mode == "3": three worker processes, with fixed agent
        assignment order:
            A = ["deR", "Red", "Mag", "ulB", "Blu"]
            B = ["nyC", "Cyn", "gaM", "leY", "Yel"]
            C = ["ZyX", "XyZ", "nrG", "Grn", "NoN"]
      * thread_mode == "5": five worker processes, with fixed agent
        assignment order:
            A = ["Red", "Grn", "Blu"]
            B = ["Cyn", "Mag", "Yel"]
            C = ["ZyX", "XyZ", "NoN"]
            D = ["deR", "nrG", "ulB"]
            E = ["nyC", "gaM", "leY"]

    Champion matches are run per-agent, immediately after GA training
    for that agent, inside each worker process (see train_group_agent_sequence).

    Returns:
      True  if all required agents in this cycle completed GA + champion
            matches successfully (no worker failures).
      False if any worker process failed; in that case snapshots must
            NOT be rotated and the cycle counter must NOT be advanced.
    """
    log(
        f"[global] cycle {cycle}: champion match logs will be written per-agent as "
        f"champion_matches_<Name>.txt in {base_dir}"
    )

    # GA_BASE_DIR is per-process, but we set it in the main process too.
    global GA_BASE_DIR, CURRENT_AGENT_NAME, CURRENT_CYCLE
    GA_BASE_DIR = base_dir
    CURRENT_AGENT_NAME = ""
    CURRENT_CYCLE = cycle

    # Load or initialise per-cycle, per-agent progress state.
    progress = load_cycle_progress(base_dir, cycle, agent_names)
    agents_progress: Dict[str, Dict[str, object]] = progress["agents"]  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Multi-thread modes: 3 or 5 worker processes, or 1.
    # We split agents into fixed groups and train each group sequentially
    # inside its own process. Champion matches run inside each worker
    # immediately after that agent's GA training.
    # ------------------------------------------------------------------

    # Define the fixed group layouts.
    if thread_mode == "3":
        thread_groups: List[List[str]] = [
            ["deR", "Red", "Mag", "ulB", "Blu"],           # A
            ["nyC", "Cyn", "gaM", "leY", "Yel"],           # B
            ["ZyX", "XyZ", "nrG", "Grn", "NoN"],           # C
        ]
    elif thread_mode == "5":
        thread_groups = [
            ["Red", "Grn", "Blu"],                         # A
            ["Cyn", "Mag", "Yel"],                         # B
            ["ZyX", "XyZ", "NoN"],                         # C
            ["deR", "nrG", "ulB"],                         # D
            ["nyC", "gaM", "leY"],                         # E
        ]
    elif thread_mode == "1":
        thread_groups = [
            [
                "ZyX", "nyC", "deR", "Cyn", "Red",
                "XyZ", "gaM", "nrG", "Mag", "Grn",
                "leY", "ulB", "NoN", "Yel", "Blu",
            ],
        ]
    else:
        raise ValueError(f"Unsupported thread_mode {thread_mode!r}; expected '1', '3', or '5'.")

    # Determine which agents need GA+champ training in this cycle.
    # Anything not marked 'done' is treated as needing work.
    agents_to_train: List[str] = []
    for name in agent_names:
        entry = agents_progress.get(name, {"state": "pending"})
        state = entry.get("state", "pending")
        if state != "done":
            agents_to_train.append(name)

    # Nothing to do for this cycle; all agents already marked done.
    if not agents_to_train:
        log(f"[global] cycle {cycle}: all agents already marked done; skipping training.")
        return True

    # Build a seed per agent to keep randomness per-agent stable-ish.
    name_to_seed: Dict[str, int] = {
        name: int(rnd.randint(0, 2**31 - 1)) for name in agents_to_train
    }

    # Build per-group lists that only include agents that actually need training.
    grouped_agents: List[List[str]] = []
    for group in thread_groups:
        g = [name for name in group if name in agents_to_train]
        if g:
            grouped_agents.append(g)

    if thread_mode == "1":
        # Single-process mode: run groups inline, no multiprocessing.
        for group in grouped_agents:
            group_seed_map = {name: name_to_seed[name] for name in group}
            train_group_agent_sequence(
                group,
                opponent_lists,
                base_dir,
                cycle,
                group_seed_map,
            )
    else:
        # Spawn one process per non-empty group.
        processes: List[multiprocessing.Process] = []
        for idx, group in enumerate(grouped_agents):
            group_seed_map = {name: name_to_seed[name] for name in group}
            p = multiprocessing.Process(
                target=train_group_agent_sequence,
                args=(group, opponent_lists, base_dir, cycle, group_seed_map),
                name=f"BDGroup-{idx+1}",
            )
            processes.append(p)

        # Start all workers.
        for p in processes:
            p.start()

        # Wait for all workers to finish their entire group workload.
        for p in processes:
            p.join()

        # If any worker died, leave progress as-is so this cycle can be safely retried.
        failed = [p.name for p in processes if p.exitcode not in (0, None)]
        if failed:
            log(
                f"[global] cycle {cycle}: WARNING: worker failures in {', '.join(failed)}; "
                f"leaving agents in non-'done' states for retry. Snapshots will NOT be rotated.",
            )
            save_cycle_progress(base_dir, progress)
            return False

    # Mark all trained agents as "done" in cycle_progress, since champion
    # matches have already been run per-agent inside the workers.
    for name in agents_to_train:
        entry = agents_progress.get(name, {"state": "pending"})
        entry["state"] = "done"
        entry["last_cycle"] = cycle
        # last_gen is not used elsewhere; preserve if present, else set -1.
        entry["last_gen"] = entry.get("last_gen", -1)
        agents_progress[name] = entry

    progress["agents"] = agents_progress
    save_cycle_progress(base_dir, progress)

    return True


def rotate_snapshots(agent_names: List[str], base_dir: str) -> None:
    """
    Perform snapshot rotation after all agents have completed their cycle
    training and champion matches.

    Before rotation (after finishing cycle c), for each agent Name:
      * Name_0: 8 parents from cycle c (including the cycle-c champion).
      * Name_1: 8 parents from cycle c-1 (including the cycle-(c-1) champion).
      * Name_2: single champion from cycle c-2 (if it exists).
      * Name_3: single champion from cycle c-3 (if it exists).

    After rotation:
      * Name_3: single champion that used to be in Name_2
                (i.e. champion from cycle c-2).
      * Name_2: champion extracted from Name_1's 8 parents
                (i.e. champion from cycle c-1).
      * Name_1: the 8 parents from Name_0 (i.e. parents from cycle c).
      * Name_0: left as the pruned 8-parent set from cycle c; a fresh GA
                population will be built to disk from Name_0 when the
                next cycle starts.
    """
    for name in agent_names:
        path0 = os.path.join(base_dir, f"{name}_0.pkl")
        path1 = os.path.join(base_dir, f"{name}_1.pkl")
        path2 = os.path.join(base_dir, f"{name}_2.pkl")
        path3 = os.path.join(base_dir, f"{name}_3.pkl")

        # 1) Discard old _3
        if os.path.exists(path3):
            os.remove(path3)
            log(f"[{name}] rotation: removed old {path3}", also_print=False)

        # 2) Move current _2 champion to _3 (if present)
        champ_from_2 = load_champion(path2)
        if champ_from_2 is not None:
            save_champion(champ_from_2, path3)
            log(f"[{name}] rotation: moved champion from _2 to _3.", also_print=False)

        # 3) Extract prior-cycle champion from _1 into _2
        parents_prev = load_parents(path1)
        if parents_prev:
            champ_prev = parents_prev[0]
            save_champion(champ_prev, path2)
            log(f"[{name}] rotation: extracted champion from _1 into _2.", also_print=False)

        # 4) Move this cycle's parents (from _0) into _1
        parents_this = load_parents(path0)
        if parents_this:
            save_parents(parents_this, path1)
            log(f"[{name}] rotation: copied parents from _0 into _1.", also_print=False)


def update_cycle_counter(base_dir: str) -> int:
    """Increment cycle counter in state file and return new cycle value."""
    state_path = os.path.join(base_dir, 'training_state.json')
    data = {'cycle': 0}
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {'cycle': 0}
        except Exception:
            data = {'cycle': 0}

    data['cycle'] = int(data.get('cycle', 0) or 0) + 1

    try:
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception:
        # Failing to persist the counter should not crash training.
        pass

    return data['cycle']


def initialize_agents(agent_names: List[str], base_dir: str) -> None:
    """
    Initialise snapshot files on first run.  Creates initial parent sets
    and snapshot copies.  Does not overwrite existing files.

    For each agent, we create:

      * 4 Xavier-initialised seed networks (conceptual generation 0 elites).
      * 4 direct copies of those seeds (conceptual generation 0 other parents).

    These 8 parents are saved to `_0` and `_1`.  Snapshot champions for
    indices 1, 2, and 3 are three distinct nets drawn from the seed set,
    so even in cycle 0 the three opponent snapshots are not identical.
    """
    os.makedirs(base_dir, exist_ok=True)
    # Check if already initialised
    state_path = os.path.join(base_dir, 'training_state.json')
    if os.path.exists(state_path):
        return

    rnd = np.random.RandomState()
    for name in agent_names:
        # 4 Xavier seeds
        seeds: List[Agent] = []
        for _ in range(4):
            a = Agent(name)
            a.net = MLP(input_dim=594, hidden_dim=512, seed=rnd.randint(0, 2**31 - 1))
            a.trainable = True
            seeds.append(a)

        # 4 straight copies as additional parents
        parents: List[Agent] = []
        parents.extend(seeds)
        parents.extend([s.clone() for s in seeds])

        # Stored parents are frozen snapshots
        for p in parents:
            p.trainable = False

        # Save parent sets for snapshot 0 and 1
        save_parents(parents, os.path.join(base_dir, f"{name}_0.pkl"))
        save_parents(parents, os.path.join(base_dir, f"{name}_1.pkl"))

        # Champions:
        # - Snapshot 1 champion = parents[0] = seeds[0] (via load_champion on the parent list)
        # - Snapshot 2 champion = seeds[1]
        # - Snapshot 3 champion = seeds[2]
        # This guarantees three distinct parameter sets for s=1,2,3.
        champ_s2 = seeds[1] if len(seeds) > 1 else seeds[0]
        champ_s3 = seeds[2] if len(seeds) > 2 else seeds[0]

        save_champion(champ_s2, os.path.join(base_dir, f"{name}_2.pkl"))
        save_champion(champ_s3, os.path.join(base_dir, f"{name}_3.pkl"))

    # Write initial state
    with open(state_path, 'w') as f:
        json.dump({'cycle': 0}, f)


def main() -> None:
    # Define agent names and opponent lists
    agent_names = [
        "Red", "Grn", "Blu", "Cyn", "Mag", "Yel",
        "NoN", "deR", "nrG", "ulB", "nyC", "gaM",
        "leY", "XyZ", "ZyX",
    ]
    opponent_lists: Dict[str, List[str]] = {
        "Red": ["NoN", "Grn", "ulB", "Cyn", "gaM", "Yel", "ZyX", "Red"],
        "Grn": ["deR", "NoN", "Blu", "Cyn", "Mag", "leY", "ZyX", "Grn"],
        "Blu": ["Red", "nrG", "NoN", "nyC", "Mag", "Yel", "ZyX", "Blu"],
        "Cyn": ["deR", "nrG", "Blu", "NoN", "gaM", "Yel", "ZyX", "Cyn"],
        "Mag": ["Red", "nrG", "ulB", "Cyn", "NoN", "leY", "ZyX", "Mag"],
        "Yel": ["deR", "Grn", "ulB", "nyC", "Mag", "NoN", "ZyX", "Yel"],
        "NoN": ["deR", "nrG", "ulB", "nyC", "gaM", "leY", "ZyX", "NoN"],
        "deR": ["Red", "nrG", "Blu", "nyC", "Mag", "leY", "ZyX", "deR"],
        "nrG": ["Red", "Grn", "ulB", "nyC", "gaM", "Yel", "ZyX", "nrG"],
        "ulB": ["deR", "Grn", "Blu", "Cyn", "gaM", "leY", "ZyX", "ulB"],
        "nyC": ["Red", "Grn", "ulB", "Cyn", "Mag", "leY", "ZyX", "nyC"],
        "gaM": ["deR", "Grn", "Blu", "nyC", "Mag", "Yel", "ZyX", "gaM"],
        "leY": ["Red", "nrG", "Blu", "Cyn", "gaM", "Yel", "ZyX", "leY"],
        "XyZ": [
            "Red", "Grn", "Blu", "Cyn", "Mag", "Yel",
            "NoN", "deR", "nrG", "ulB", "nyC", "gaM", "leY", "XyZ",
        ],
        "ZyX": ["XyZ", "ZyX"],
    }

    # CLI argument for thread mode
    parser = argparse.ArgumentParser(description="Battledance Chess training driver")
    parser.add_argument(
        "--threads-mode",
        choices=["1", "3", "5"],
        default="5",
        help="Thread grouping mode: '1', '3', or '5'.",
    )
    args = parser.parse_args()
    thread_mode = args.threads_mode

    # Base directory for models
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    # Initialise agents on first run only
    initialize_agents(agent_names, base_dir)

    # Read current cycle from state, robust against partial/corrupt JSON.
    state_path = os.path.join(base_dir, "training_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cycle = int(data.get("cycle", 0) or 0)
            else:
                cycle = 0
        except Exception:
            cycle = 0
    else:
        cycle = 0

    # Set up logging for this run
    setup_logging(base_dir, cycle)

    rnd = np.random.RandomState()

    log(f"=== Training cycle {cycle} starting (threads-mode={thread_mode}) ===")
    completed = run_training_cycle(agent_names, opponent_lists, base_dir, rnd, cycle, thread_mode=thread_mode)

    if not completed:
        log(f"=== Training cycle {cycle} aborted; snapshots NOT rotated and cycle counter NOT advanced. ===")
        return

    rotate_snapshots(agent_names, base_dir)
    update_cycle_counter(base_dir)
    log(f"=== Training cycle {cycle} completed; snapshots rotated. ===")


if __name__ == '__main__':
    main()
