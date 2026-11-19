"""Command line: python -m aot {propose,review,execute,log}"""

from __future__ import annotations

import argparse
import json
import sys

from . import agent, approval, tracing
from .store import DEFAULT_DB_PATH, Store


def main() -> int:
    ap = argparse.ArgumentParser(prog="aot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose", help="run the agent once; writes a proposal, executes nothing")
    p.add_argument("--allow-untraced", action="store_true", help="run even if Langfuse is not configured")
    r = sub.add_parser("review", help="human decision on a proposal (interactive terminal only)")
    r.add_argument("id")
    e = sub.add_parser("execute", help="dry-run an approved action")
    e.add_argument("id")
    sub.add_parser("log", help="print the action log")
    args = ap.parse_args()

    store = Store(DEFAULT_DB_PATH)
    try:
        if args.cmd == "propose":
            store.load_snapshot()
            try:
                result = agent.propose(store, tracing.from_env(), allow_untraced=args.allow_untraced)
            except tracing.UntracedRunError as err:
                print(f"refused: {err}", file=sys.stderr)
                return 2
            print(json.dumps(result, indent=2))
            if result["status"] == "proposed":
                print(f"\nNothing was executed. A human must run: python -m aot review {result['id']}")
        elif args.cmd == "review":
            if not sys.stdin.isatty():
                print("refused: review needs an interactive terminal", file=sys.stderr)
                return 2
            proposal = store.proposal(args.id)
            if proposal is None:
                print(f"no proposal with id {args.id}", file=sys.stderr)
                return 1
            print(json.dumps(proposal, indent=2))
            operator = input("Operator name: ")
            typed = input(f"Type '{approval.confirmation_phrase(args.id)}' to approve, anything else rejects: ")
            print(approval.decide(store, args.id, operator, typed))
        elif args.cmd == "execute":
            try:
                print(approval.execute(store, args.id))
            except approval.ApprovalError as err:
                print(f"refused: {err}", file=sys.stderr)
                return 2
        elif args.cmd == "log":
            ev = store.events()
            print(ev.to_string(index=False) if not ev.empty else "(empty)")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
