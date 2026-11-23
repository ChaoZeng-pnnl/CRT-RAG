"""CLI: ask the agent a question and show its trace."""

from __future__ import annotations

import argparse
import json
import sys

from .graph import ask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the perioperative nursing agent.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--no-web", action="store_true", help="Disable web escalation.")
    parser.add_argument("--no-verify", action="store_true", help="Disable citation verification.")
    parser.add_argument("--no-safety", action="store_true", help="Disable the safety gate.")
    parser.add_argument("--no-routing", action="store_true", help="Always use the semantic branch.")
    parser.add_argument("--json", action="store_true", help="Print the full state as JSON.")
    args = parser.parse_args(argv)

    question = " ".join(args.question)
    state = ask(question,
                enable_web=not args.no_web,
                enable_verification=not args.no_verify,
                enable_safety=not args.no_safety,
                enable_routing=not args.no_routing)

    if args.json:
        print(json.dumps(state, indent=2, default=str))
        return 0

    print("=" * 70)
    print(f"Q: {question}")
    print("=" * 70)
    print(f"\nbranch: {state.get('query_type')} — {state.get('classification_reason')}")
    print(f"retrieval: {', '.join(state.get('retrieval_branches', []))}")
    print(f"evidence: {len(state.get('ranked') or state.get('contexts', []))} passages | "
          f"sufficient={state.get('sufficient')} "
          f"({state.get('sufficiency_reason','')})")
    if state.get("escalated"):
        print("escalated to web search")

    print(f"\n{state.get('answer','')}\n")

    print("citations:")
    for c in state.get("citations", []):
        page = f" p{c['page']}" if c.get("page") else ""
        print(f"  [{c['marker']}] T{c['tier']} {c['source']} — {(c.get('title') or '')[:52]}{page}")
    if not state.get("citations"):
        print("  (none)")

    if state.get("unsupported_claims"):
        print(f"\nunsupported claims flagged: {len(state['unsupported_claims'])}")
        for claim in state["unsupported_claims"][:3]:
            print(f"  - {claim[:100]}")
    if state.get("safety_flags"):
        print(f"safety flags: {state['safety_flags']} -> {state.get('safety_action')}")

    total = sum(step.get("ms", 0) for step in state.get("trace", []))
    print(f"\ntrace ({total} ms total):")
    for step in state.get("trace", []):
        extra = {k: v for k, v in step.items() if k not in ("node", "ms")}
        print(f"  {step['node']:22s} {step['ms']:6d} ms  {extra if extra else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
