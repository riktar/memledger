"""Command-line interface for MemLedger."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from memledger.api import Ledger
from memledger.prompts import find_project_root


def _default_db_path() -> str:
    return "memory.db"


def _copy_defaults(target_dir: Path) -> None:
    root = find_project_root()
    policy_target = target_dir / "memory.policy.yaml"
    prompts_target = target_dir / "prompts"
    if not policy_target.exists():
        shutil.copy2(root / "memory.policy.yaml", policy_target)
    prompts_target.mkdir(exist_ok=True)
    for prompt_file in (root / "prompts").iterdir():
        if prompt_file.is_file() and not (prompts_target / prompt_file.name).exists():
            shutil.copy2(prompt_file, prompts_target / prompt_file.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memledger")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("path", nargs="?", default=_default_db_path())

    log_parser = subparsers.add_parser("log")
    log_parser.add_argument("--db", default=_default_db_path())
    log_parser.add_argument("--type")
    log_parser.add_argument("--session")
    log_parser.add_argument("--since")

    why_parser = subparsers.add_parser("why")
    why_parser.add_argument("id")
    why_parser.add_argument("--db", default=_default_db_path())

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--db", default=_default_db_path())

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--db", default=_default_db_path())
    replay_parser.add_argument("--at")
    replay_parser.add_argument("--cached", action="store_true")

    rebuild_parser = subparsers.add_parser("rebuild")
    rebuild_parser.add_argument("--db", default=_default_db_path())

    regen_parser = subparsers.add_parser("regenerate")
    regen_parser.add_argument("--db", default=_default_db_path())
    regen_parser.add_argument("--model")
    regen_parser.add_argument("--prompt", default="extract@v1")

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("id")
    delete_parser.add_argument("--db", default=_default_db_path())
    delete_parser.add_argument("--cascade", action="store_true")
    delete_parser.add_argument("--reason", default="manual")

    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("--db", default=_default_db_path())
    return parser


def _open_ledger(path: str) -> Ledger:
    return Ledger(path=path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command == "init":
        target_path = Path(args.path)
        _copy_defaults(target_path.parent if target_path.parent != Path("") else Path.cwd())
        ledger = Ledger(path=str(target_path))
        ledger.close()
        print(target_path)
        return 0

    ledger = _open_ledger(args.db)
    try:
        if command == "log":
            events = ledger.store.iter_events(session=args.session, type=args.type, since=args.since)
            for event in events:
                print(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True))
            return 0

        if command == "why":
            print(json.dumps(ledger.why(args.id), ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        if command == "review":
            proposals = [event for event in ledger.store.iter_events(type="promotion_proposed")]
            print(
                json.dumps(
                    [event.to_dict() for event in proposals],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if command == "replay":
            ledger.replay(at=args.at, cached=args.cached)
            print("ok")
            return 0

        if command == "rebuild":
            ok = ledger.rebuild()
            print("ok" if ok else "mismatch")
            return 0 if ok else 1

        if command == "regenerate":
            count = ledger.regenerate(model=args.model, prompt=args.prompt)
            print(count)
            return 0

        if command == "delete":
            ledger.delete(args.id, cascade=args.cascade, reason=args.reason)
            print(args.id)
            return 0

        if command == "stats":
            print(json.dumps(ledger.stats(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    finally:
        ledger.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
