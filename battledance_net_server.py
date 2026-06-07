#!/usr/bin/env python3
"""
Local Battledance Chess net-play server.

Place this file beside:
  - battledance_training.py
  - battledance_chess_vs_net.html
  - models/*.bdpop

Then run:
  python battledance_net_server.py

The browser GUI is served at http://127.0.0.1:<port>/ and can ask this
server to load the selected .bdpop from ./models and choose a move using
parent/net index 0 by default.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
HTML_NAME = "battledance_chess_vs_net.html"
MODELS_DIR = SCRIPT_DIR / "models"
TRAINING_CONFIG = SCRIPT_DIR / "training_config.ini"

# Ensure imports resolve to the colocated battledance_training.py.
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import battledance_training as bt  # type: ignore
except Exception as exc:  # pragma: no cover - user-facing startup failure
    print("Could not import battledance_training.py from this folder.", file=sys.stderr)
    print(f"Folder: {SCRIPT_DIR}", file=sys.stderr)
    print(f"Error: {exc!r}", file=sys.stderr)
    raise

# Respect a colocated training_config.ini, especially hidden_layers and move_choice.
try:
    if TRAINING_CONFIG.exists() and hasattr(bt, "_load_training_config"):
        bt._load_training_config(str(TRAINING_CONFIG))  # type: ignore[attr-defined]
except Exception as exc:
    print(f"Warning: could not apply training_config.ini: {exc!r}", file=sys.stderr)

START_FEN = "rglbblgr/pfnkknfp/8/8/8/8/PFNKKNFP/RGLBBLGR w - - 0 1"
MOVE_RE = re.compile(r"^([KFNLPGRB])([a-h][1-8])([x-])([a-h][1-8])([+#=]?)$", re.I)
DROP_RE = re.compile(r"^([KFNLPGRBkfnlpgr])-@-([a-h][1-8])([+#=]?)$")

_MODEL_CACHE: Dict[Tuple[str, int, int], Tuple[float, Any]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def coord_to_pos(coord: str) -> Tuple[int, int]:
    """Convert algebraic coordinate like e2 to engine (row, col)."""
    file_ch = coord[0].lower()
    rank = int(coord[1])
    col = ord(file_ch) - ord("a")
    row = 8 - rank
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError(f"Invalid coordinate: {coord!r}")
    return row, col


def pos_to_coord(pos: Tuple[int, int]) -> str:
    row, col = pos
    return f"{chr(ord('a') + col)}{8 - row}"


def clean_token(token: str) -> str:
    token = str(token or "").strip()
    # Raw champion logs may have quiet-move commas. GUI records normally do not.
    if token.endswith(","):
        token = token[:-1]
    return token.strip()


def move_base_notation(board: Any, mv: Any) -> str:
    if mv.kind == "drop":
        tr, tc = mv.to_pos
        return f"{mv.drop_type}-@-{pos_to_coord((tr, tc))}"

    fr, fc = mv.from_pos
    tr, tc = mv.to_pos
    piece = board.board[fr][fc]
    target = board.board[tr][tc]
    sep = "x" if target is not None else "-"
    return f"{piece.kind}{pos_to_coord((fr, fc))}{sep}{pos_to_coord((tr, tc))}"


def legal_move_from_notation(board: Any, token: str) -> Any:
    token = clean_token(token)
    # Remove result/check/draw suffix; legality and scoring are recomputed by the engine.
    token_core = token.rstrip(",+#=")

    m = DROP_RE.match(token_core)
    if m:
        piece = m.group(1).upper()
        to_pos = coord_to_pos(m.group(2))
        for mv in board.generate_legal_moves(board.turn):
            if mv.kind != "drop":
                continue
            if mv.to_pos == to_pos and str(mv.drop_type).upper() == piece:
                return mv
        raise ValueError(f"No legal drop matches {token!r} for side {board.turn!r}.")

    m = MOVE_RE.match(token_core)
    if m:
        piece = m.group(1).upper()
        from_pos = coord_to_pos(m.group(2))
        to_pos = coord_to_pos(m.group(4))
        for mv in board.generate_legal_moves(board.turn):
            if mv.kind != "move":
                continue
            if mv.from_pos != from_pos or mv.to_pos != to_pos:
                continue
            p = board.board[from_pos[0]][from_pos[1]]
            if p is not None and p.kind.upper() == piece:
                return mv
        raise ValueError(f"No legal move matches {token!r} for side {board.turn!r}.")

    raise ValueError(f"Could not parse move token: {token!r}")


def apply_notation_moves(moves: List[str]) -> Tuple[Any, Optional[str], Optional[str]]:
    """
    Rebuild the game from the start and apply the given GUI notation moves.

    Returns (board, result, terminal_reason). result is None if non-terminal.
    """
    board = bt.BattledanceBoard(START_FEN)
    result: Optional[str] = None
    reason: Optional[str] = None

    for raw in moves:
        token = clean_token(raw)
        if not token:
            continue
        if result is not None:
            raise ValueError(f"Move {token!r} appears after terminal result {result}.")
        mv = legal_move_from_notation(board, token)
        mover = board.turn
        winner = board.apply_move(mv)
        if winner is not None:
            result = "1-0" if winner == "w" else "0-1"
            reason = "bishop capture terminal"
        elif not board.generate_legal_moves(board.turn):
            result = "1-0" if mover == "w" else "0-1"
            reason = "mate / no legal move" if board.is_in_check(board.turn) else "stalemate loss / no legal move"
        elif board.check_draw():
            result = "1/2-1/2"
            reason = "draw rule"
        # If non-terminal, board.turn has already advanced.
        _ = mover

    return board, result, reason


def score_and_apply_net_move(board: Any, agent: Any) -> Dict[str, Any]:
    """Ask the agent for a move, apply it, and return GUI notation/status."""
    if board.check_draw():
        return {"game_over": True, "result": "1/2-1/2", "reason": "draw rule"}

    legal = board.generate_legal_moves(board.turn)
    if not legal:
        loser = board.turn
        result = "0-1" if loser == "w" else "1-0"
        return {"game_over": True, "result": result, "reason": "side to move has no legal moves"}

    mv = agent.choose_move(board)
    if mv is None:
        loser = board.turn
        result = "0-1" if loser == "w" else "1-0"
        return {"game_over": True, "result": result, "reason": "net returned no move"}

    base = move_base_notation(board, mv)
    mover = board.turn
    winner = board.apply_move(mv)

    result = None
    suffix = ","
    reason = ""
    if winner is not None:
        suffix = "#"
        result = "1-0" if winner == "w" else "0-1"
        reason = "bishop capture terminal"
    else:
        check = board.is_in_check(board.turn)
        if not board.generate_legal_moves(board.turn):
            suffix = "#"
            result = "1-0" if mover == "w" else "0-1"
            reason = "mate / no legal move" if check else "stalemate loss / no legal move"
        elif board.check_draw():
            suffix = "="
            result = "1/2-1/2"
            reason = "draw rule"
        elif check:
            suffix = "+"
            reason = "check"

    return {
        "move": base + suffix,
        "mover": "W" if mover == "w" else "B",
        "result": result,
        "reason": reason,
        "game_over": result is not None,
    }


def safe_model_path(model_name: str) -> Path:
    if not model_name:
        raise ValueError("No model was specified.")
    raw = str(model_name).replace("\\", "/")
    candidate = (MODELS_DIR / raw).resolve()
    models_root = MODELS_DIR.resolve()
    try:
        candidate.relative_to(models_root)
    except ValueError:
        raise ValueError("Model path must stay inside the local models/ folder.")
    if candidate.suffix.lower() != ".bdpop":
        raise ValueError("Model must be a .bdpop file.")
    if not candidate.exists():
        raise FileNotFoundError(f"Model file not found: {model_name}")
    return candidate


def load_agent_from_model(model_name: str, net_index: int = 0) -> Any:
    path = safe_model_path(model_name)
    stat = path.stat()
    key = (str(path), int(stat.st_mtime_ns), int(stat.st_size))

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            parents = cached[1]
        else:
            parents = bt.load_parents(str(path))
            if not parents and hasattr(bt, "load_champion"):
                champ = bt.load_champion(str(path))  # type: ignore[attr-defined]
                if champ is not None:
                    parents = [champ]
            if not parents:
                raise ValueError(f"Could not load any parent/champion net from {model_name!r}.")
            _MODEL_CACHE.clear()
            _MODEL_CACHE[key] = (time.time(), parents)

    if not (0 <= int(net_index) < len(parents)):
        raise IndexError(f"net_index {net_index} is out of range for {model_name!r}; loaded {len(parents)} net(s).")
    return parents[int(net_index)]


def list_models() -> List[str]:
    if not MODELS_DIR.exists():
        return []
    out: List[str] = []
    for path in MODELS_DIR.rglob("*.bdpop"):
        if path.is_file():
            out.append(path.relative_to(MODELS_DIR).as_posix())
    return sorted(out, key=str.lower)


class Handler(SimpleHTTPRequestHandler):
    server_version = "BattledanceNetServer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} - " + fmt % args)

    def send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/models"):
            self.send_json({"ok": True, "models": list_models(), "models_dir": str(MODELS_DIR)})
            return
        if self.path in ("/", ""):
            self.path = "/" + HTML_NAME
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path.startswith("/api/net-move"):
                payload = self.read_json()
                model = str(payload.get("model", ""))
                net_index = int(payload.get("net_index", 0) or 0)
                moves = payload.get("moves", [])
                if not isinstance(moves, list):
                    raise ValueError("moves must be a JSON array of notation strings.")
                board, result, reason = apply_notation_moves([str(x) for x in moves])
                if result is not None:
                    self.send_json({"ok": True, "game_over": True, "result": result, "reason": reason})
                    return
                agent = load_agent_from_model(model, net_index)
                answer = score_and_apply_net_move(board, agent)
                answer["ok"] = True
                self.send_json(answer)
                return
            self.send_json({"ok": False, "error": "Unknown endpoint."}, status=404)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)


def find_free_port(preferred: int) -> int:
    import socket

    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("Could not find a free localhost port.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Battledance Chess GUI-vs-net server.")
    parser.add_argument("--port", type=int, default=8765, help="Preferred localhost port. Default: 8765.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    args = parser.parse_args()

    if not (SCRIPT_DIR / HTML_NAME).exists():
        raise SystemExit(f"Missing {HTML_NAME!r} beside {Path(__file__).name}.")

    os.chdir(SCRIPT_DIR)
    port = find_free_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    print("Battledance net-play server")
    print(f"Folder : {SCRIPT_DIR}")
    print(f"Models : {MODELS_DIR}")
    print(f"URL    : {url}")
    print("Close this window or press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    # Match training behavior: stochastic choice should not repeat from process forks.
    random.seed()
    main()
