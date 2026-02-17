# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Jacob Scow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 or
# later as published by the Free Software Foundation.
#
# See the LICENSE.txt file in the project root for details.

# Jacob Scow, a.k.a., RadiantDarkBlaze

r"""
Battledance Chess training program.

Implements a self-play training framework for the Battledance Chess variant:
- Full rules engine (custom leap/slide pieces, drops, threefold, 64-move, long-game cap).
- State encoding: 594-dim float feature vector.
- Evaluator: 3-hidden-layer tanh MLP (512 wide), scalar output.
- Population-based neuroevolution (260 nets), selection/crossover/mutation.
- Snapshot semantics: four slots per agent (_0.._3), parents/champions across cycles.
- Per-cycle resumability via progress JSONs; per-agent champion evaluation logs.

Licensed under GPL-3.0-or-later. See LICENSE.txt.
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
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
###############################################################################
#  Graceful stop (main-process keypress + cross-process stop event)
###############################################################################

class GracefulStop(Exception):
    """Raised to request a clean stop at the next safe checkpoint."""
    pass


def check_stop(stop_event, *, ga_progress: Optional[Dict] = None) -> None:
    """
    If stop_event is set, raise GracefulStop.

    If ga_progress is provided, attempt one final durable write of the current
    GA progress before raising (best-effort).
    """
    if stop_event is None:
        return
    try:
        is_set = stop_event.is_set()
    except Exception:
        return
    if not is_set:
        return

    # Best-effort: force a durable GA progress checkpoint on stop.
    if ga_progress is not None:
        try:
            path = _ga_progress_path()
            if path:
                _safe_write_json(path, ga_progress, indent=2, durable=True)
        except Exception:
            pass

    raise GracefulStop()


def start_stop_listener(stop_event) -> threading.Thread:
    """
    Start a daemon thread in the main process that listens for a keypress to
    request a graceful stop. On Windows it listens for 'q' without Enter using
    msvcrt; otherwise it falls back to reading a line ('q' + Enter).
    """
    def _thread() -> None:
        # Prefer Windows non-blocking keypress (no Enter).
        try:
            import msvcrt  # type: ignore
            log("[global] Press 'q' to request graceful stop (finish current game, save, exit).")
            while not stop_event.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch and ch.lower() == "q":
                        stop_event.set()
                        log("[global] Graceful stop requested (q).")
                        return
                time.sleep(0.1)
            return
        except Exception:
            pass

        # Portable fallback: requires Enter.
        try:
            log("[global] Type 'q' + Enter to request graceful stop (finish current game, save, exit).")
            while not stop_event.is_set():
                line = sys.stdin.readline()
                if not line:
                    return
                if line.strip().lower() in ("q", "quit", "stop"):
                    stop_event.set()
                    log("[global] Graceful stop requested (stdin).")
                    return
        except Exception:
            return

    t = threading.Thread(target=_thread, daemon=True)
    t.start()
    return t


def install_sigint_as_graceful(stop_event) -> None:
    """
    Convert the first Ctrl+C into a graceful stop request; a second Ctrl+C
    triggers the normal KeyboardInterrupt.
    """
    try:
        import signal
    except Exception:
        return

    try:
        old_handler = signal.getsignal(signal.SIGINT)
    except Exception:
        old_handler = None

    def _handler(sig, frame):
        if not stop_event.is_set():
            stop_event.set()
            log("[global] Ctrl+C received -> graceful stop requested. Press Ctrl+C again to force.")
        else:
            if callable(old_handler):
                old_handler(sig, frame)
            raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        return


###############################################################################
#  Simple file-based training log
###############################################################################

LOG_PATH: Optional[str] = None
CURRENT_AGENT_NAME: str = ""
CURRENT_GEN: int = 0
CURRENT_CYCLE: int = 0
STATUS_QUEUE: Optional[object] = None
LOG_QUEUE: Optional[object] = None

def setup_logging(base_dir: str, cycle: int) -> None:
    global LOG_PATH, CURRENT_CYCLE
    CURRENT_CYCLE = cycle
    if LOG_PATH is None:
        try:
            os.makedirs(base_dir, exist_ok=True)
            LOG_PATH = os.path.join(base_dir, "training_log.txt")
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n=== New session cycle={cycle} started {ts} ===\n")
        except Exception:
            LOG_PATH = None
            # swallow: logging must never crash training
            pass

def log(msg: str, also_print: bool = True) -> None:
    """
    Append a timestamped line to the log file and optionally echo to stdout.
    In multi-process mode, printing is performed only by the main log consumer.
    """
    # Only suppress printing if someone configured status redirection WITHOUT log redirection.
    # (If LOG_QUEUE is active, printing is safe: it happens in the main consumer.)
    if STATUS_QUEUE is not None and LOG_QUEUE is None:
        also_print = False

    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}] {msg}"

    if LOG_QUEUE is not None:
        try:
            LOG_QUEUE.put_nowait((os.getpid(), line, bool(also_print)))
        except Exception:
            pass
        return

    if also_print:
        print(line, flush=True)

    if LOG_PATH is not None:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

_last_status_len: int = 0

_status_active: bool = False

def init_ipc(status_queue: Optional[object], log_queue: Optional[object]) -> None:
    """
    Called inside worker processes to redirect status() and log() updates
    to the main process.
    """
    global STATUS_QUEUE, LOG_QUEUE
    STATUS_QUEUE = status_queue
    LOG_QUEUE = log_queue

def status_newline() -> None:
    """
    End the current status line with a newline.

    In multi-process mode (STATUS_QUEUE set), this sends a newline request
    to the main process. Otherwise it prints locally.

    IMPORTANT: this is a no-op if this process has not emitted any status()
    updates since the last status_newline(). This prevents stray blank lines
    when a phase completes without printing a status bar.
    """
    global _last_status_len, _status_active

    if not _status_active and _last_status_len == 0:
        return

    if STATUS_QUEUE is not None:
        try:
            # msg=None is treated as "print newline" by the main consumer.
            STATUS_QUEUE.put_nowait((os.getpid(), None))
        except Exception:
            pass
    else:
        print()

    _last_status_len = 0
    _status_active = False


def status(msg: str) -> None:
    """
    Update a single carriage-return status line on the console.

    In multi-process modes, workers push status updates to the main process
    through STATUS_QUEUE so only one writer touches stdout (prevents smear).
    """
    line = (msg or "")[:200]  # cap length to something sane

    global _status_active
    _status_active = True

    # Worker redirection path
    if STATUS_QUEUE is not None:
        try:
            STATUS_QUEUE.put_nowait((os.getpid(), line))
        except Exception:
            pass
        return

    # Single-process path (original behavior)
    global _last_status_len
    sys.stdout.write("\r" + line)

    if _last_status_len > len(line):
        sys.stdout.write(" " * (_last_status_len - len(line)))
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
    path = os.path.join(base_dir, "cycle_progress.json")
    data = _safe_read_json(path)
    if not isinstance(data, dict):
        data = {}

    if data.get("cycle") != cycle:
        agents_progress: Dict[str, Dict[str, object]] = {}
    else:
        agents_progress = data.get("agents", {}) or {}
        if not isinstance(agents_progress, dict):
            agents_progress = {}

    for name in agent_names:
        if name not in agents_progress or not isinstance(agents_progress.get(name), dict):
            agents_progress[name] = {"state": "pending"}

    return {"cycle": cycle, "agents": agents_progress}

def save_cycle_progress(base_dir: str, progress: Dict) -> None:
    """
    Persist the current per-cycle, per-agent progress state atomically.
    """
    path = os.path.join(base_dir, "cycle_progress.json")
    try:
        _safe_write_json(path, progress, indent=2, durable=True)
    except Exception:
        pass

def load_champion_progress(base_dir: str, cycle: int, name: str) -> Dict:
    path = os.path.join(base_dir, f"champion_progress_{name}.json")
    data = _safe_read_json(path)
    if not isinstance(data, dict) or data.get("cycle") != cycle:
        return {"cycle": cycle, "next_index": 0}

    next_index = data.get("next_index", 0)
    if not isinstance(next_index, int):
        next_index = 0
    data["next_index"] = int(next_index)
    return data

def save_champion_progress(base_dir: str, name: str, progress: Dict) -> None:
    path = os.path.join(base_dir, f"champion_progress_{name}.json")
    try:
        _safe_write_json(path, progress, indent=2, durable=True)
    except Exception:
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
    path = _ga_progress_path()
    if not path:
        return None
    data = _safe_read_json(path)
    return data if isinstance(data, dict) else None

def save_ga_progress(progress: Dict) -> None:
    path = _ga_progress_path()
    if not path:
        return
    try:
        _safe_write_json(path, progress, indent=2, durable=False)
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
    path = _ga_done_file(base_dir, name)
    data = _safe_read_json(path)
    return data if isinstance(data, dict) else None

def save_ga_done_state(base_dir: str, name: str, cycle: int, last_gen: int) -> None:
    path = _ga_done_file(base_dir, name)
    payload = {
        "cycle": int(cycle),
        "agent": name,
        "last_gen": int(last_gen),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    try:
        _safe_write_json(path, payload, indent=2, durable=True)
    except Exception:
        pass

def _atomic_write_bytes(path: str, payload: bytes, *, make_backup: bool = True, durable: bool = True) -> None:
    """
    Write bytes atomically:
      - write to a temp file in the same directory
      - flush (+ optional fsync)
      - optionally copy existing target to .bak (best effort)
      - os.replace temp -> target (atomic on Windows/POSIX)

    If anything fails, the existing target is left untouched.
    """
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)

    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            f.write(payload)
            f.flush()
            if durable:
                os.fsync(f.fileno())

        if make_backup and os.path.exists(path):
            try:
                # Best-effort backup; do not remove/rename the live file.
                import shutil
                shutil.copy2(path, path + ".bak")
            except Exception:
                pass

        os.replace(tmp, path)  # atomic replace
    finally:
        # Clean up temp if it still exists
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _safe_write_json(path: str, obj: object, *, indent: int = 2, durable: bool = True) -> None:
    payload = json.dumps(obj, indent=indent).encode("utf-8")
    _atomic_write_bytes(path, payload, make_backup=True, durable=durable)


def _safe_write_pickle(path: str, obj: object, *, durable: bool = True) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(path, payload, make_backup=True, durable=durable)


def _safe_read_json(path: str) -> Optional[object]:
    for p in (path, path + ".bak"):
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return None


def _safe_read_pickle(path: str) -> Optional[object]:
    for p in (path, path + ".bak"):
        if not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            continue
    return None

###############################################################################
#  Game representation
###############################################################################

# Shared constants for feature encoding; avoids rebuilding these every call.
PIECE_ORDER: List[str] = ['K', 'N', 'F', 'L', 'P', 'G', 'R', 'B']
PIECE_INDEX: Dict[str, int] = {k: i for i, k in enumerate(PIECE_ORDER)}
IN_HAND_TYPES: List[str] = ['K', 'N', 'F', 'L', 'P', 'G', 'R']
HAND_INDEX: Dict[str, int] = {k: i for i, k in enumerate(IN_HAND_TYPES)}  # K N F L P G R
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

@dataclass(slots=True)
class _UndoFull:
    prev_turn: str
    prev_plys_since_capture_or_drop: int
    prev_plys_since_game_start: int
    prev_last_rep_key: Optional[str]

    move: Move
    moved_piece: Optional['Piece']          # None for drop
    captured_piece: Optional['Piece']       # only for normal moves

    # Drop undo (restore removed hand piece at exact index)
    drop_removed_index: Optional[int]       # only for drops

    # Capture undo (remove appended-in-hand piece if any)
    hand_added_char: Optional[str]          # char appended (case-preserving)
    hand_added_to: Optional[str]            # 'w' or 'b'
    hand_added_index: Optional[int]         # index inserted/appended at time of capture undo

    # Repetition undo (decrement this key)
    rep_key_after: Optional[str]


@dataclass(slots=True)
class _UndoMinimal:
    move: Move
    moved_piece: Optional['Piece']
    captured_piece: Optional['Piece']

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

        self._rebuild_fast_state()

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
        Also caches the last position key so undo logic doesn't re-hash.
        """
        key = self._position_key()
        self.repetition_counts[key] = self.repetition_counts.get(key, 0) + 1
        self._last_rep_key = key

    def _rebuild_fast_state(self) -> None:
        """(Re)build fast caches: occupied squares, bishop squares, hand counts."""
        self._pos_w: set[Tuple[int, int]] = set()
        self._pos_b: set[Tuple[int, int]] = set()
        self._bishops_w: set[Tuple[int, int]] = set()
        self._bishops_b: set[Tuple[int, int]] = set()

        for r in range(8):
            for c in range(8):
                cell = self.board[r][c]
                if cell is None:
                    continue
                if cell.color == "w":
                    self._pos_w.add((r, c))
                    if cell.kind == "B":
                        self._bishops_w.add((r, c))
                else:
                    self._pos_b.add((r, c))
                    if cell.kind == "B":
                        self._bishops_b.add((r, c))

        self._hand_w_counts: List[int] = [0] * 7
        self._hand_b_counts: List[int] = [0] * 7

        for ch in self.captured_w:
            idx = HAND_INDEX.get(ch.upper())
            if idx is not None:
                self._hand_w_counts[idx] += 1

        for ch in self.captured_b:
            idx = HAND_INDEX.get(ch.upper())
            if idx is not None:
                self._hand_b_counts[idx] += 1

        if not hasattr(self, "_last_rep_key"):
            self._last_rep_key = None

    def repetition_count(self) -> int:
        """
        Return how many times the current position has occurred so far
        in this game (including the current occurrence).

        Fast path: use cached _last_rep_key set by record_state().
        Fallback: compute key if cache missing (should be rare).
        """
        if not hasattr(self, "repetition_counts"):
            return 1

        key = getattr(self, "_last_rep_key", None)
        if key is None:
            # Fallback: if someone calls repetition_count before record_state()
            key = self._position_key()

        cnt = self.repetition_counts.get(key, 0)
        # If record_state() wasn't called for this position, treat as first occurrence.
        return cnt if cnt > 0 else 1

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

        result = sorted(offsets)
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

        result = sorted(dirs)
        cls._SLIDE_CACHE[kind] = result
        return result

    def get_bishop_positions(self, color: str) -> List[Tuple[int, int]]:
        """Return positions of all royal bishops of the given colour."""
        if not hasattr(self, "_bishops_w"):
            self._rebuild_fast_state()
        s = self._bishops_w if color == "w" else self._bishops_b
        return list(s)

    def is_square_attacked(self, row: int, col: int, by_color: str) -> bool:
        """
        Determine if (row, col) is attacked by a piece of colour ``by_color``.
        Drops cannot capture, so only leaps and slides are considered.
        """
        if not hasattr(self, "_pos_w"):
            self._rebuild_fast_state()

        positions = self._pos_w if by_color == "w" else self._pos_b
        board = self.board
        sym_leaps = self.symmetrical_leaps
        sym_slides = self.symmetrical_slides

        for r, c in positions:
            cell = board[r][c]
            if cell is None or cell.color != by_color:
                continue
            kind = cell.kind

            for dx, dy in sym_leaps(kind):
                if r + dx == row and c + dy == col:
                    return True

            for dx, dy in sym_slides(kind):
                step = 1
                while True:
                    nr = r + dx * step
                    nc = c + dy * step
                    if not (0 <= nr < 8 and 0 <= nc < 8):
                        break
                    if nr == row and nc == col:
                        return True
                    if board[nr][nc] is not None:
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
        attacked. These moves will be filtered later.

        Optimized: iterate cached piece positions instead of scanning 64 squares.
        """
        if not hasattr(self, "_pos_w"):
            self._rebuild_fast_state()

        moves: List[Move] = []
        board = self.board
        positions = self._pos_w if color == "w" else self._pos_b

        # Piece moves (iterate only occupied squares for this color)
        for r, c in sorted(positions):
            cell = board[r][c]
            if cell is None or cell.color != color:
                # Defensive against any cache drift
                continue

            kind = cell.kind
            leaps = self.symmetrical_leaps(kind)
            slides = self.symmetrical_slides(kind)

            # Leap moves
            for dx, dy in leaps:
                nr, nc = r + dx, c + dy
                if not (0 <= nr < 8 and 0 <= nc < 8):
                    continue
                target = board[nr][nc]
                if target is None or target.color != color:
                    moves.append(Move("move", (r, c), (nr, nc)))

            # Slide moves
            for dx, dy in slides:
                step = 1
                while True:
                    nr = r + dx * step
                    nc = c + dy * step
                    if not (0 <= nr < 8 and 0 <= nc < 8):
                        break
                    target = board[nr][nc]
                    if target is None:
                        moves.append(Move("move", (r, c), (nr, nc)))
                        step += 1
                        continue
                    elif target.color != color:
                        moves.append(Move("move", (r, c), (nr, nc)))
                    break

        # Drop moves (dedupe by piece type, preserving first-seen order)
        captured_list = self.captured_w if color == "w" else self.captured_b
        seen = set()
        for ch in captured_list:
            if ch in seen:
                continue
            seen.add(ch)
            for hr in self.home_rows(color):
                for col in range(8):
                    if board[hr][col] is None:
                        moves.append(Move("drop", None, (hr, col), drop_type=ch))

        return moves

    def generate_legal_moves(self, color: str) -> List[Move]:
        cand = self.generate_moves_unfiltered(color)

        bishop_caps: List[Move] = []
        for move in cand:
            if move.kind == 'move':
                tr, tc = move.to_pos
                target = self.board[tr][tc]
                if target is not None and target.kind == 'B' and target.color != color:
                    bishop_caps.append(move)

        if bishop_caps:
            return bishop_caps

        legal: List[Move] = []
        for move in cand:
            undo = self._apply_move_minimal(move)
            try:
                if not self.is_in_check(color):
                    legal.append(move)
            finally:
                self._unapply_move_minimal(undo)

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

        # fast caches
        clone._pos_w = set(self._pos_w)
        clone._pos_b = set(self._pos_b)
        clone._bishops_w = set(self._bishops_w)
        clone._bishops_b = set(self._bishops_b)
        clone._hand_w_counts = list(self._hand_w_counts)
        clone._hand_b_counts = list(self._hand_b_counts)
        clone._last_rep_key = getattr(self, "_last_rep_key", None)

        return clone

    def apply_move(self, move: Move) -> Optional[str]:
        """
        Apply a move and update counters and repetition history.  Returns
        'w' if White wins, 'b' if Black wins, or None if the game continues.
        Capturing a royal bishop ends the game.  Counters are reset after
        any capture or drop.
        """
        if not hasattr(self, "_pos_w"):
            self._rebuild_fast_state()

        color = self.turn
        enemy = 'b' if color == 'w' else 'w'
        winner: Optional[str] = None
        capture_or_drop = False

        pos_self = self._pos_w if color == "w" else self._pos_b
        pos_enemy = self._pos_b if color == "w" else self._pos_w
        bishops_self = self._bishops_w if color == "w" else self._bishops_b
        bishops_enemy = self._bishops_b if color == "w" else self._bishops_w

        if move.kind == 'move':
            fr, fc = move.from_pos
            tr, tc = move.to_pos
            piece = self.board[fr][fc]
            target = self.board[tr][tc]

            pos_self.discard((fr, fc))
            pos_self.add((tr, tc))
            if piece.kind == "B":
                bishops_self.discard((fr, fc))
                bishops_self.add((tr, tc))

            if target is not None:
                pos_enemy.discard((tr, tc))
                if target.kind == "B":
                    bishops_enemy.discard((tr, tc))
                    winner = color

                if winner is None:
                    captured_char = target.kind.upper() if color == 'w' else target.kind.lower()
                    if color == 'w':
                        self.captured_w.append(captured_char)
                        idx = HAND_INDEX.get(captured_char.upper())
                        if idx is not None:
                            self._hand_w_counts[idx] += 1
                    else:
                        self.captured_b.append(captured_char)
                        idx = HAND_INDEX.get(captured_char.upper())
                        if idx is not None:
                            self._hand_b_counts[idx] += 1
                capture_or_drop = True

            self.board[fr][fc] = None
            self.board[tr][tc] = piece

        else:
            tr, tc = move.to_pos
            ch = move.drop_type
            drop_color = 'w' if ch.isupper() else 'b'
            drop_kind = ch.upper()
            if drop_color != color:
                raise ValueError(
                    f"Illegal drop: side to move is {color} but drop_type implies {drop_color} ({ch!r})."
                )

            self.board[tr][tc] = Piece(drop_kind, drop_color)

            if drop_color == "w":
                self._pos_w.add((tr, tc))
                if drop_kind == "B":
                    self._bishops_w.add((tr, tc))
            else:
                self._pos_b.add((tr, tc))
                if drop_kind == "B":
                    self._bishops_b.add((tr, tc))

            if drop_color == 'w':
                self.captured_w.remove(ch)
                idx = HAND_INDEX.get(drop_kind)
                if idx is not None:
                    self._hand_w_counts[idx] -= 1
            else:
                self.captured_b.remove(ch)
                idx = HAND_INDEX.get(drop_kind)
                if idx is not None:
                    self._hand_b_counts[idx] -= 1

            capture_or_drop = True

        self.turn = enemy

        self.plys_since_game_start += 1
        if capture_or_drop:
            self.plys_since_capture_or_drop = 0
        else:
            self.plys_since_capture_or_drop += 1

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

    def apply_move_with_undo(self, move: Move) -> Tuple[Optional[str], _UndoFull]:
        """
        Apply a move *in-place* and return (winner, undo_record) so the move
        can be undone exactly (including hand order, counters, repetition).
        """
        if not hasattr(self, "_pos_w"):
            self._rebuild_fast_state()

        prev_turn = self.turn
        prev_pcd = self.plys_since_capture_or_drop
        prev_pgs = self.plys_since_game_start
        prev_last_rep_key = getattr(self, "_last_rep_key", None)

        moved_piece: Optional[Piece] = None
        captured_piece: Optional[Piece] = None
        drop_removed_index: Optional[int] = None
        hand_added_char: Optional[str] = None
        hand_added_to: Optional[str] = None
        hand_added_index: Optional[int] = None

        if move.kind == "move":
            fr, fc = move.from_pos
            tr, tc = move.to_pos
            moved_piece = self.board[fr][fc]
            captured_piece = self.board[tr][tc]
            if captured_piece is not None and captured_piece.kind != "B":
                hand_added_char = captured_piece.kind.upper() if prev_turn == "w" else captured_piece.kind.lower()
                hand_added_to = prev_turn
                # apply_move() appends to the end; record the exact index it will occupy.
                hand_added_index = len(self.captured_w) if prev_turn == "w" else len(self.captured_b)
        else:
            ch = move.drop_type
            lst = self.captured_w if ch.isupper() else self.captured_b
            drop_removed_index = lst.index(ch)

        winner = self.apply_move(move)
        rep_key_after = getattr(self, "_last_rep_key", None)

        undo = _UndoFull(
            prev_turn=prev_turn,
            prev_plys_since_capture_or_drop=prev_pcd,
            prev_plys_since_game_start=prev_pgs,
            prev_last_rep_key=prev_last_rep_key,

            move=move,
            moved_piece=moved_piece,
            captured_piece=captured_piece,

            drop_removed_index=drop_removed_index,

            hand_added_char=hand_added_char,
            hand_added_to=hand_added_to,
            hand_added_index=hand_added_index,

            rep_key_after=rep_key_after,
        )
        return winner, undo

    def unapply_move(self, undo: _UndoFull) -> None:
        """Undo a move previously applied by apply_move_with_undo()."""
        if not hasattr(self, "_pos_w"):
            self._rebuild_fast_state()

        key = undo.rep_key_after
        if key is not None:
            cnt = self.repetition_counts.get(key, 0)
            if cnt <= 1:
                self.repetition_counts.pop(key, None)
            else:
                self.repetition_counts[key] = cnt - 1

        self.turn = undo.prev_turn
        self.plys_since_capture_or_drop = undo.prev_plys_since_capture_or_drop
        self.plys_since_game_start = undo.prev_plys_since_game_start
        self._last_rep_key = undo.prev_last_rep_key

        move = undo.move
        board = self.board

        if move.kind == "move":
            fr, fc = move.from_pos
            tr, tc = move.to_pos

            mover = undo.prev_turn
            pos_self = self._pos_w if mover == "w" else self._pos_b
            pos_enemy = self._pos_b if mover == "w" else self._pos_w
            bishops_self = self._bishops_w if mover == "w" else self._bishops_b
            bishops_enemy = self._bishops_b if mover == "w" else self._bishops_w

            board[fr][fc] = undo.moved_piece
            board[tr][tc] = undo.captured_piece

            pos_self.discard((tr, tc))
            pos_self.add((fr, fc))
            if undo.moved_piece is not None and undo.moved_piece.kind == "B":
                bishops_self.discard((tr, tc))
                bishops_self.add((fr, fc))

            if undo.captured_piece is not None:
                pos_enemy.add((tr, tc))
                if undo.captured_piece.kind == "B":
                    bishops_enemy.add((tr, tc))

            if undo.hand_added_char is not None and undo.hand_added_to is not None:
                if undo.hand_added_to == "w":
                    idx0 = undo.hand_added_index
                    if isinstance(idx0, int) and 0 <= idx0 < len(self.captured_w) and self.captured_w[idx0] == undo.hand_added_char:
                        del self.captured_w[idx0]
                    else:
                        # Fallback to previous behavior
                        self.captured_w.pop()
                    hi = HAND_INDEX.get(undo.hand_added_char.upper())
                    if hi is not None:
                        self._hand_w_counts[hi] -= 1
                else:
                    idx0 = undo.hand_added_index
                    if isinstance(idx0, int) and 0 <= idx0 < len(self.captured_b) and self.captured_b[idx0] == undo.hand_added_char:
                        del self.captured_b[idx0]
                    else:
                        self.captured_b.pop()
                    hi = HAND_INDEX.get(undo.hand_added_char.upper())
                    if hi is not None:
                        self._hand_b_counts[hi] -= 1

        else:
            tr, tc = move.to_pos
            ch = move.drop_type
            drop_color = "w" if ch.isupper() else "b"
            drop_kind = ch.upper()

            board[tr][tc] = None

            if drop_color == "w":
                self._pos_w.discard((tr, tc))
                if drop_kind == "B":
                    self._bishops_w.discard((tr, tc))
            else:
                self._pos_b.discard((tr, tc))
                if drop_kind == "B":
                    self._bishops_b.discard((tr, tc))

            idx = int(undo.drop_removed_index or 0)
            if drop_color == "w":
                self.captured_w.insert(idx, ch)
                hi = HAND_INDEX.get(drop_kind)
                if hi is not None:
                    self._hand_w_counts[hi] += 1
            else:
                self.captured_b.insert(idx, ch)
                hi = HAND_INDEX.get(drop_kind)
                if hi is not None:
                    self._hand_b_counts[hi] += 1

    def _apply_move_minimal(self, move: Move) -> _UndoMinimal:
        """
        Apply a move in-place for legality checking only.
        Does NOT touch: hands, counters, repetition, turn.
        Does touch: board + fast caches (pos/bishops).
        """
        if not hasattr(self, "_pos_w"):
            self._rebuild_fast_state()

        board = self.board

        if move.kind == "move":
            fr, fc = move.from_pos
            tr, tc = move.to_pos
            moved = board[fr][fc]
            captured = board[tr][tc]

            mover = moved.color
            pos_self = self._pos_w if mover == "w" else self._pos_b
            pos_enemy = self._pos_b if mover == "w" else self._pos_w
            bishops_self = self._bishops_w if mover == "w" else self._bishops_b
            bishops_enemy = self._bishops_b if mover == "w" else self._bishops_w

            # Robust cache ops (no KeyError if drift ever occurs)
            pos_self.discard((fr, fc))
            pos_self.add((tr, tc))
            if moved.kind == "B":
                bishops_self.discard((fr, fc))
                bishops_self.add((tr, tc))

            if captured is not None:
                pos_enemy.discard((tr, tc))
                if captured.kind == "B":
                    bishops_enemy.discard((tr, tc))

            board[fr][fc] = None
            board[tr][tc] = moved

            return _UndoMinimal(move=move, moved_piece=moved, captured_piece=captured)

        else:
            tr, tc = move.to_pos
            ch = move.drop_type
            drop_color = "w" if ch.isupper() else "b"
            drop_kind = ch.upper()
            dropped = Piece(drop_kind, drop_color)

            board[tr][tc] = dropped
            if drop_color == "w":
                self._pos_w.add((tr, tc))
                if drop_kind == "B":
                    self._bishops_w.add((tr, tc))
            else:
                self._pos_b.add((tr, tc))
                if drop_kind == "B":
                    self._bishops_b.add((tr, tc))

            return _UndoMinimal(move=move, moved_piece=None, captured_piece=None)


    def _unapply_move_minimal(self, undo: _UndoMinimal) -> None:
        if not hasattr(self, "_pos_w"):
            self._rebuild_fast_state()

        board = self.board
        move = undo.move

        if move.kind == "move":
            fr, fc = move.from_pos
            tr, tc = move.to_pos
            moved = undo.moved_piece
            captured = undo.captured_piece

            mover = moved.color
            pos_self = self._pos_w if mover == "w" else self._pos_b
            pos_enemy = self._pos_b if mover == "w" else self._pos_w
            bishops_self = self._bishops_w if mover == "w" else self._bishops_b
            bishops_enemy = self._bishops_b if mover == "w" else self._bishops_w

            board[fr][fc] = moved
            board[tr][tc] = captured

            pos_self.discard((tr, tc))
            pos_self.add((fr, fc))
            if moved.kind == "B":
                bishops_self.discard((tr, tc))
                bishops_self.add((fr, fc))

            if captured is not None:
                pos_enemy.add((tr, tc))
                if captured.kind == "B":
                    bishops_enemy.add((tr, tc))

        else:
            tr, tc = move.to_pos
            ch = move.drop_type
            drop_color = "w" if ch.isupper() else "b"
            drop_kind = ch.upper()

            board[tr][tc] = None
            if drop_color == "w":
                self._pos_w.discard((tr, tc))
                if drop_kind == "B":
                    self._bishops_w.discard((tr, tc))
            else:
                self._pos_b.discard((tr, tc))
                if drop_kind == "B":
                    self._bishops_b.discard((tr, tc))

    def encode_features(self) -> np.ndarray:
        """
        Encode the current state into a 594-dimensional float32 vector.
        Same semantics as before, but faster:
          * no (9,8,8) allocation
          * in-hand counts maintained incrementally
        """
        if not hasattr(self, "_hand_w_counts"):
            self._rebuild_fast_state()

        out = np.zeros(FEATURE_TOTAL, dtype=np.float32)

        for r in range(8):
            for c in range(8):
                cell = self.board[r][c]
                if cell is None:
                    continue
                idx = PIECE_INDEX.get(cell.kind)
                if idx is None:
                    continue
                val = 1.0 if cell.color == 'w' else -1.0
                sq = r * 8 + c
                out[idx * 64 + sq] = val
                out[8 * 64 + sq] = val

        base = FEATURE_PLANES

        hw = self._hand_w_counts
        hb = self._hand_b_counts
        for i in range(7):
            out[base + i] = min(hw[i], 4) / 4.0
            out[base + 7 + i] = min(hb[i], 4) / 4.0

        stm = 1.0 if self.turn == 'w' else -1.0
        m64 = min(self.plys_since_capture_or_drop / 128.0, 1.0)
        rep = min(self.repetition_count(), 3) / 3.0
        glen = min(self.plys_since_game_start / 4096.0, 1.0)

        out[base + 14 + 0] = stm
        out[base + 14 + 1] = m64
        out[base + 14 + 2] = rep
        out[base + 14 + 3] = glen

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
        # Avoid calling __init__ (which does Xavier init we immediately overwrite).
        child = MLP.__new__(MLP)

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
        """
        Construct an MLP directly from parameter arrays without calling __init__.
        Avoids wasted Xavier initialisation when loading from disk.
        """
        net = cls.__new__(cls)
        for attr in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']:
            arr = np.asarray(data[attr], dtype=np.float32)
            # Ensure we own writable storage (pickle can give odd flags sometimes)
            if not arr.flags.writeable:
                arr = arr.copy()
            setattr(net, attr, arr)
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

        scores: List[float] = []
        for mv in legal:
            winner, undo = board.apply_move_with_undo(mv)
            try:
                if winner is not None:
                    score = 1e6 if winner == color else -1e6
                else:
                    v = self.evaluate_state(board)
                    score = v if color == 'w' else -v
                scores.append(score)
            finally:
                board.unapply_move(undo)

        best = max(scores)
        worst = min(scores)
        if abs(best - worst) < 1e-9:
            return random.choice(legal)

        norms = [(s - worst) / (best - worst) for s in scores]
        filtered: List[Tuple[Move, float]] = []
        for mv, n in zip(legal, norms):
            if n > 0.8:
                weight = (n - 0.8) / 0.2
                filtered.append((mv, weight))

        if not filtered:
            return random.choice(legal)

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
    serial = []
    for agent in parents:
        serial.append({
            "name": agent.name,
            "trainable": False,
            "net": agent.net.to_dict(),
        })
    try:
        _safe_write_pickle(filepath, serial, durable=True)
    except Exception:
        pass

def load_parents(filepath: str) -> List[Agent]:
    data = _safe_read_pickle(filepath)
    if not isinstance(data, list):
        return []

    parents: List[Agent] = []
    for entry in data:
        if not isinstance(entry, dict) or "net" not in entry:
            continue
        net_dict = entry.get("net")
        if not isinstance(net_dict, dict):
            continue
        try:
            net = MLP.from_dict(net_dict)
        except Exception:
            continue
        parents.append(
            Agent(
                name=entry.get("name", "unknown"),
                net=net,
                trainable=False,
            )
        )
    return parents

def save_champion(agent: Agent, filepath: str) -> None:
    data = {
        "name": agent.name,
        "trainable": False,
        "net": agent.net.to_dict(),
    }
    try:
        _safe_write_pickle(filepath, data, durable=True)
    except Exception:
        pass

def load_champion(filepath: str) -> Optional[Agent]:
    data = _safe_read_pickle(filepath)
    if data is None:
        return None

    # Case 1: single champion dict
    if isinstance(data, dict) and "net" in data and isinstance(data["net"], dict):
        try:
            net = MLP.from_dict(data["net"])
        except Exception:
            return None
        return Agent(name=data.get("name", "unknown"), net=net, trainable=False)

    # Case 2: list of parent dicts
    if isinstance(data, list) and data:
        champ_data = data[0]
        if isinstance(champ_data, dict) and "net" in champ_data and isinstance(champ_data["net"], dict):
            try:
                net = MLP.from_dict(champ_data["net"])
            except Exception:
                return None
            return Agent(name=champ_data.get("name", "unknown"), net=net, trainable=False)

    return None

def save_population_state(name: str, population: List[Agent], gen: int, base_dir: str) -> None:
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
        _safe_write_pickle(path, serial, durable=True)
    except Exception:
        pass

def load_population_state(name: str, base_dir: str) -> Optional[Tuple[List[Agent], int]]:
    path = os.path.join(base_dir, f"{name}_0.pkl")
    data = _safe_read_pickle(path)
    if not (isinstance(data, dict) and data.get("kind") == "population"):
        return None

    gen = int(data.get("generation", 0) or 0)
    agents_data = data.get("agents")
    if not isinstance(agents_data, list):
        return None

    population: List[Agent] = []
    for ad in agents_data:
        if not isinstance(ad, dict):
            continue

        net_dict = ad.get("net")
        try:
            net = MLP.from_dict(net_dict) if isinstance(net_dict, dict) else MLP()
        except Exception:
            net = MLP()

        population.append(
            Agent(
                name=str(ad.get("name", "unknown")),
                net=net,
                trainable=bool(ad.get("trainable", True)),
            )
        )

    if not population:
        return None

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
    stop_event=None,
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
        check_stop(stop_event, ga_progress=progress)
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
            check_stop(stop_event, ga_progress=progress)
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
        check_stop(stop_event, ga_progress=progress)
    # Clean newline after the status bar
    status_newline()

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
    check_stop(stop_event, ga_progress=progress)
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
    stop_event=None,
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
        check_stop(stop_event, ga_progress=progress)
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
        check_stop(stop_event, ga_progress=progress)
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
        check_stop(stop_event, ga_progress=progress)
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
        check_stop(stop_event, ga_progress=progress)
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
        check_stop(stop_event, ga_progress=progress)
    # Clean newline after status line
    status_newline()

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
    check_stop(stop_event, ga_progress=progress)
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

def _coerce_to_8_parents(
    parents: List[Agent],
    rnd: np.random.RandomState,
    *,
    fallback_name: str,
) -> List[Agent]:
    """
    Ensure we have exactly 8 parent Agents.

    - If >= 8: truncate.
    - If 1..7: pad by cloning existing parents (chosen via rnd).
    - If 0: synthesize 8 fresh Xavier-initialised parents (rare; indicates corrupt/missing parents file).
    """
    if len(parents) >= 8:
        return parents[:8]

    if len(parents) == 0:
        # Last-resort: make 8 fresh parents so the run doesn't crash.
        # This is intentionally non-deterministic across runs unless rnd is seeded.
        fresh: List[Agent] = []
        for _ in range(8):
            net_seed = int(rnd.randint(0, 2**31 - 1))
            fresh.append(
                Agent(
                    name=fallback_name,
                    net=MLP(input_dim=594, hidden_dim=512, seed=net_seed),
                    trainable=False,
                )
            )
        log(f"[{fallback_name}] WARNING: parent list empty; synthesised 8 fresh parents.")
        return fresh

    out: List[Agent] = list(parents)
    while len(out) < 8:
        k = int(rnd.randint(0, len(out)))
        out.append(out[k].clone())
        out[-1].trainable = False
    return out

def build_children(parents: List[Agent], rnd: np.random.RandomState) -> List[Agent]:
    """
    Generate 256 non-elite children from (up to) 8 parents using a uniform
    crossover grid.

    Robustness: if parents != 8 due to corrupted/legacy files, we pad/truncate
    so training continues instead of crashing.
    """
    fallback_name = parents[0].name if parents else (CURRENT_AGENT_NAME or "unknown")
    parents8 = _coerce_to_8_parents(parents, rnd, fallback_name=fallback_name)

    children: List[Agent] = []
    for i in range(8):
        for j in range(8):
            p1 = parents8[i]
            p2 = parents8[j]
            for _ in range(4):
                child_net = p1.net.crossover(p2.net, rnd)
                child_net.mutate(mutation_rate=0.015625, weight_decay=0.00001526, rnd=rnd)
                children.append(Agent(name=p1.name, net=child_net, trainable=True))

    # Sanity: 256 children
    assert len(children) == 256, f"Expected 256 children, got {len(children)}"
    return children

def build_population_from_parents(
    name: str,
    parents: List[Agent],
    rnd: np.random.RandomState,
) -> List[Agent]:
    """
    Given a parent set, build a 260-net GA population:

      * 256 children produced via crossover/mutation of an 8-parent pool.
      * 4 elite direct clones of the top four parents.

    Robustness: parents are coerced to exactly 8 so legacy/corrupt files
    do not crash the run.
    """
    parents8 = _coerce_to_8_parents(parents, rnd, fallback_name=(name or CURRENT_AGENT_NAME or "unknown"))

    # 256 mutated children from the full 8-parent grid
    children = build_children(parents8, rnd)

    pop: List[Agent] = []
    pop.extend(children)

    # 4 elites as direct clones of the top four parents
    elites: List[Agent] = []
    for i in range(4):
        elite = parents8[i].clone()
        elite.trainable = True
        elites.append(elite)

    pop.extend(elites)
    return pop

def train_population_once(
    pop: List[Agent],
    opponents: List[Agent],
    env: BattledanceEnvironment,
    rnd: np.random.RandomState,
    use_worst_only: bool = False,
    stop_event=None,
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
        stop_event=stop_event,
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
        stop_event=stop_event,
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

    n_opps = len(opponents)

    for local_rank, idx in enumerate(parent_indices):
        row_stats = snapshot_stats_2.get(idx, None)

        # Defensive: missing/short/corrupt stats must fail the gate.
        if not isinstance(row_stats, list) or len(row_stats) != n_opps:
            success = False
            log(
                f"[{CURRENT_AGENT_NAME}] cycle {CURRENT_CYCLE} gen {CURRENT_GEN}: "
                f"WARNING: missing or invalid Stage-2 stats for parent idx={idx} "
                f"(have {0 if not isinstance(row_stats, list) else len(row_stats)}, expected {n_opps}).",
                also_print=False,
            )
            break

        row: List[Tuple[str, int, int]] = []
        for snap_idx in range(n_opps):
            opp_name = opponents[snap_idx].name

            pair = row_stats[snap_idx]
            if not (isinstance(pair, tuple) and len(pair) == 2):
                success = False
                log(
                    f"[{CURRENT_AGENT_NAME}] cycle {CURRENT_CYCLE} gen {CURRENT_GEN}: "
                    f"WARNING: corrupt Stage-2 stat tuple for parent idx={idx} vs {opp_name}.",
                    also_print=False,
                )
                break

            margin, draws = pair
            try:
                margin_i = int(margin)
                draws_i = int(draws)
            except Exception:
                success = False
                log(
                    f"[{CURRENT_AGENT_NAME}] cycle {CURRENT_CYCLE} gen {CURRENT_GEN}: "
                    f"WARNING: non-integer Stage-2 stats for parent idx={idx} vs {opp_name}.",
                    also_print=False,
                )
                break

            row.append((opp_name, margin_i, draws_i))
            if margin_i < 0:
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
    stop_event=None,
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
        status_newline()
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
    check_stop(stop_event)
    # Truncate this agent's log file to the first `next_index` complete games,
    # robustly: scan for header lines and take the following moves line.
    existing: List[str] = []
    if os.path.exists(outfile_path):
        try:
            with open(outfile_path, "r", encoding="utf-8") as f:
                existing = f.read().splitlines()
        except Exception:
            existing = []

    games: List[Tuple[str, str]] = []
    i = 0
    while i + 1 < len(existing):
        line = existing[i]
        if line.startswith("Cycle: "):
            header = existing[i]
            moves = existing[i + 1]
            games.append((header, moves))
            i += 2
        else:
            # Skip stray/extra lines (future-proofing)
            i += 1

    completed_games = min(next_index, len(games))
    trimmed: List[str] = []
    for header, moves in games[:completed_games]:
        trimmed.append(header)
        trimmed.append(moves)

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
        check_stop(stop_event)
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
        check_stop(stop_event)
    # Clean up the status line with a newline at the end of champion matches
    status_newline()

def play_game_with_moves(agent1: Agent, agent2: Agent, env: BattledanceEnvironment) -> Tuple[int, str]:
    """
    result:
      +1 = agent1 (as White) wins
      -1 = agent2 (as Black) wins
       0 = draw

    moves_str:
      fixed-width chunks: "<move><mark><space>"
        mark: ',' normal, '+' gives check, '#' ends game (bishop capture OR mate-by-no-legal-move)
              '=' draw
    """
    board = env.initial_board.copy()
    players = {'w': agent1, 'b': agent2}

    moves_list: List[str] = []
    marks_after: List[str] = []  # one char per move: ',', '+', '#', '='

    while True:
        color = board.turn
        enemy = 'b' if color == 'w' else 'w'
        agent = players[color]

        mv = agent.choose_move(board)
        if mv is None:
            # Side to move loses in your rules.
            loser = color
            result = 1 if loser == 'b' else -1

            # Retroactively mark the previous move as a terminal '#'
            if marks_after:
                marks_after[-1] = '#'
            break

        # Build notation from the *current* board before applying the move.
        if mv.kind == 'move':
            fr, fc = mv.from_pos
            tr, tc = mv.to_pos
            piece = board.board[fr][fc]
            target = board.board[tr][tc]

            sep = 'x' if target is not None else '-'
            move_str = (
                f"{piece.kind}"
                f"{chr(97 + fc)}{8 - fr}"
                f"{sep}"
                f"{chr(97 + tc)}{8 - tr}"
            )
        else:
            tr, tc = mv.to_pos
            move_str = f"{mv.drop_type}-@-{chr(97 + tc)}{8 - tr}"

        winner = board.apply_move(mv)

        # Compute draw exactly once.
        draw = False if winner is not None else board.check_draw()

        # Default mark logic (must be 1 char; never lengthen move_str)
        if winner is not None:
            mark = '#'
        elif draw:
            mark = '='
        else:
            mark = '+' if board.is_in_check(enemy) else ','

        moves_list.append(move_str)
        marks_after.append(mark)

        if winner is not None:
            result = 1 if winner == 'w' else -1
            break
        if draw:
            result = 0
            break

    moves_str = "".join(f"{m}{k} " for m, k in zip(moves_list, marks_after))
    return result, moves_str

###############################################################################
#  Per-agent training helper (for single- and multi-thread modes)
###############################################################################

def train_single_agent(
    name: str,
    opponent_lists: Dict[str, List[str]],
    base_dir: str,
    cycle: int,
    seed: int,
    env: Optional[BattledanceEnvironment] = None,
    stop_event=None,
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

    if env is None:
        env = BattledanceEnvironment()
    check_stop(stop_event)


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
                net_seed = int(rnd.randint(0, 2**31 - 1))
                a = Agent(
                    name=name,
                    net=MLP(input_dim=594, hidden_dim=512, seed=net_seed),
                    trainable=True,
                )
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
    # If the spec says we should have opponents but none loaded, treat as fatal.
    if opponent_lists.get(name) and len(opponents) == 0:
        raise RuntimeError(
            f"[{name}] cycle {cycle}: expected opponent snapshots but loaded none. "
            f"Check that {base_dir} contains <Opp>_1.pkl/<Opp>_2.pkl/<Opp>_3.pkl for listed opponents."
        )

    # Train population until success
    gen = gen_start
    unsuccessful = 0
    success = False
    parents = []

    while not success:
        check_stop(stop_event)
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
            stop_event=stop_event,
        )

        # Persist the full population to `_0` after each generation so we
        # can resume GA without rebuilding it.
        save_population_state(name, population, gen, base_dir)
        check_stop(stop_event)

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
    status_queue: Optional[object] = None,
    log_queue: Optional[object] = None,
    stop_event=None,
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
    # Redirect status + logging to the main process before any output.
    init_ipc(status_queue, log_queue)

    # Single environment reused for champion matches in this process
    env = BattledanceEnvironment()


    try:
        for name in group_names:
            check_stop(stop_event)
    
            seed = name_to_seed.get(name, 0)
    
            # Train this agent until GA success; saves parents to Name_0.pkl
            train_single_agent(name, opponent_lists, base_dir, cycle, seed, env=env, stop_event=stop_event)
    
            check_stop(stop_event)
    
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
                stop_event=stop_event,
            )
    except GracefulStop:
        # Exit cleanly (no worker failure) so the main process can leave the cycle incomplete.
        log("[global] Graceful stop: worker exiting cleanly.")
        status_newline()
        return
    
    
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


    # ------------------------------------------------------------------
    # Graceful stop: main-process listener + shared stop flag for workers.
    # ------------------------------------------------------------------
    stop_event = threading.Event() if thread_mode == "1" else multiprocessing.Event()

    if thread_mode == "1":
        # Single-process mode: run groups inline, no multiprocessing.
        start_stop_listener(stop_event)
        install_sigint_as_graceful(stop_event)
        for group in grouped_agents:
            group_seed_map = {name: name_to_seed[name] for name in group}
            train_group_agent_sequence(
                group,
                opponent_lists,
                base_dir,
                cycle,
                group_seed_map,
                stop_event=stop_event,
            )

        # If a graceful stop was requested, leave this cycle incomplete.
        if stop_event.is_set():
            log(f"[global] cycle {cycle}: graceful stop requested; leaving cycle incomplete (no rotation).")
            save_cycle_progress(base_dir, progress)
            return False


    else:
        status_queue = multiprocessing.Queue()
        log_queue = multiprocessing.Queue()

        # Route main-process status() and log() through the queues too.
        global STATUS_QUEUE, LOG_QUEUE
        STATUS_QUEUE = status_queue
        LOG_QUEUE = log_queue

        stdout_lock = threading.Lock()
        status_text = {"line": "", "len": 0}  # shared state between consumer threads

        def _render_status_locked(line: str) -> None:
            line = (line or "")[:200]
            sys.stdout.write("\r" + line)
            if status_text["len"] > len(line):
                sys.stdout.write(" " * (status_text["len"] - len(line)))
                sys.stdout.write("\r" + line)
            sys.stdout.flush()
            status_text["line"] = line
            status_text["len"] = len(line)

        def _erase_status_locked() -> None:
            """Erase the currently rendered status from the terminal (do not forget it)."""
            if status_text["len"] > 0:
                sys.stdout.write("\r" + (" " * status_text["len"]) + "\r")
                sys.stdout.flush()

        def _consume_status_queue() -> None:
            while True:
                try:
                    pid, msg = status_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                except Exception:
                    continue

                if msg == "__STOP__":
                    break

                with stdout_lock:
                    if msg is None:
                        # newline request
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        status_text["line"] = ""
                        status_text["len"] = 0
                    else:
                        _render_status_locked(msg)

        def _consume_log_queue() -> None:
            while True:
                try:
                    pid, line, also_print = log_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                except Exception:
                    continue

                if line == "__STOP__":
                    break

                # File I/O always happens here (single writer).
                if LOG_PATH is not None:
                    try:
                        with open(LOG_PATH, "a", encoding="utf-8") as f:
                            f.write(line + "\n")
                    except Exception:
                        pass

                # Console printing: clear status, print log line, re-render status.
                if also_print:
                    with stdout_lock:
                        saved = status_text["line"]
                        _erase_status_locked()
                        sys.stdout.write(line + "\n")
                        # If the status bar is active and this is a global log line,
                        # leave a blank line so subsequent '\r' updates can't visually
                        # appear to overwrite the log.
                        if saved and "[global]" in line:
                            sys.stdout.write("\n")
                        sys.stdout.flush()
                        if saved:
                            _render_status_locked(saved)

        consumer_thread = threading.Thread(target=_consume_status_queue, daemon=True)
        log_consumer_thread = threading.Thread(target=_consume_log_queue, daemon=True)

        processes: List[multiprocessing.Process] = []

        try:
            consumer_thread.start()
            log_consumer_thread.start()
            start_stop_listener(stop_event)
            install_sigint_as_graceful(stop_event)


            for idx, group in enumerate(grouped_agents):
                group_seed_map = {name: name_to_seed[name] for name in group}
                p = multiprocessing.Process(
                    target=train_group_agent_sequence,
                    args=(group, opponent_lists, base_dir, cycle, group_seed_map, status_queue, log_queue, stop_event),
                    name=f"BDGroup-{idx+1}",
                )
                processes.append(p)

            for p in processes:
                p.start()

            for p in processes:
                p.join()

        finally:
            # Stop consumers and restore globals no matter what.
            try:
                # Ensure we end on a clean line.
                status_queue.put_nowait((os.getpid(), None))
            except Exception:
                pass

            try:
                status_queue.put_nowait((os.getpid(), "__STOP__"))
            except Exception:
                pass

            try:
                log_queue.put_nowait((os.getpid(), "__STOP__", False))
            except Exception:
                pass

            try:
                consumer_thread.join(timeout=2.0)
            except Exception:
                pass

            try:
                log_consumer_thread.join(timeout=2.0)
            except Exception:
                pass

            # Revert to direct logging (important: do this before any further log() calls).
            STATUS_QUEUE = None
            LOG_QUEUE = None

            try:
                status_queue.close()
                status_queue.join_thread()
            except Exception:
                pass

            try:
                log_queue.close()
                log_queue.join_thread()
            except Exception:
                pass

        failed = [p.name for p in processes if p.exitcode not in (0, None)]
        if failed:
            log(
                f"[global] cycle {cycle}: WARNING: worker failures in {', '.join(failed)}; "
                f"leaving agents in non-'done' states for retry. Snapshots will NOT be rotated.",
            )
            save_cycle_progress(base_dir, progress)
            return False


    # If a graceful stop was requested, do NOT mark agents done; leave the cycle incomplete.
    if stop_event.is_set():
        log(f"[global] cycle {cycle}: graceful stop requested; leaving cycle incomplete (no rotation).")
        save_cycle_progress(base_dir, progress)
        return False

    # Mark all trained agents as "done" in cycle_progress, since champion
    # matches have already been run per-agent inside the workers.
    for name in agents_to_train:
        entry = agents_progress.get(name, {"state": "pending"})
        entry["state"] = "done"
        entry["last_cycle"] = cycle

        done_state = load_ga_done_state(base_dir, name)
        if isinstance(done_state, dict) and done_state.get("cycle") == cycle:
            entry["last_gen"] = int(done_state.get("last_gen", -1) or -1)
        else:
            entry["last_gen"] = int(entry.get("last_gen", -1) or -1)

        agents_progress[name] = entry

    progress["agents"] = agents_progress
    save_cycle_progress(base_dir, progress)

    return True

def rotate_snapshots(agent_names: List[str], base_dir: str) -> None:
    """
    Perform snapshot rotation after all agents have completed their cycle
    training and champion matches.

    This version is defensive: filesystem hiccups must not crash training.
    """
    for name in agent_names:
        path0 = os.path.join(base_dir, f"{name}_0.pkl")
        path1 = os.path.join(base_dir, f"{name}_1.pkl")
        path2 = os.path.join(base_dir, f"{name}_2.pkl")
        path3 = os.path.join(base_dir, f"{name}_3.pkl")

        # 1) Discard old _3
        try:
            if os.path.exists(path3):
                os.remove(path3)
                log(f"[{name}] rotation: removed old {path3}", also_print=False)
        except Exception as e:
            log(f"[{name}] rotation: WARNING: could not remove {path3}: {e}", also_print=False)

        # 2) Move _2 champion to _3 (copy semantics)
        try:
            champ_from_2 = load_champion(path2)
            if champ_from_2 is not None:
                save_champion(champ_from_2, path3)
                log(f"[{name}] rotation: moved champion from _2 to _3.", also_print=False)
        except Exception as e:
            log(f"[{name}] rotation: WARNING: _2 -> _3 failed: {e}", also_print=False)

        # 3) Extract champion from _1 into _2
        try:
            parents_prev = load_parents(path1)
            if parents_prev:
                save_champion(parents_prev[0], path2)
                log(f"[{name}] rotation: extracted champion from _1 into _2.", also_print=False)
        except Exception as e:
            log(f"[{name}] rotation: WARNING: _1 -> _2 failed: {e}", also_print=False)

        # 4) Copy _0 parents into _1
        try:
            parents_this = load_parents(path0)
            if parents_this:
                save_parents(parents_this, path1)
                log(f"[{name}] rotation: copied parents from _0 into _1.", also_print=False)
        except Exception as e:
            log(f"[{name}] rotation: WARNING: _0 -> _1 failed: {e}", also_print=False)


def update_cycle_counter(base_dir: str) -> int:
    state_path = os.path.join(base_dir, "training_state.json")
    data = _safe_read_json(state_path)
    if not isinstance(data, dict):
        data = {"cycle": 0}

    data["cycle"] = int(data.get("cycle", 0) or 0) + 1

    try:
        _safe_write_json(state_path, data, indent=2, durable=True)
    except Exception:
        pass

    return int(data["cycle"])

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
            net_seed = int(rnd.randint(0, 2**31 - 1))
            a = Agent(
                name=name,
                net=MLP(input_dim=594, hidden_dim=512, seed=net_seed),
                trainable=True,
            )
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

    # Write initial state (atomic)
    try:
        _safe_write_json(state_path, {"cycle": 0}, indent=2, durable=True)
    except Exception:
        pass



def main() -> None:
    # Define agent names and opponent lists
    agent_names = [
        "Red", "Grn", "Blu", "Cyn", "Mag", "Yel", "NoN",
        "deR", "nrG", "ulB", "nyC", "gaM", "leY", "XyZ", "ZyX",
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
                "Red", "Grn", "Blu", "Cyn", "Mag", "Yel", "NoN",
                "deR", "nrG", "ulB", "nyC", "gaM", "leY", "XyZ",
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

    # Read current cycle from state, robust against partial/corrupt JSON (+ .bak fallback).
    state_path = os.path.join(base_dir, "training_state.json")
    state_obj = _safe_read_json(state_path)
    if isinstance(state_obj, dict):
        cycle = int(state_obj.get("cycle", 0) or 0)
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
