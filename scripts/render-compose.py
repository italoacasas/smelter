#!/usr/bin/env python3
"""Render the generated compose file for Smelter workloads."""

from __future__ import annotations

import sys

from smelter_config import ConfigError, load_state, render_compose


def main() -> int:
    try:
        state = load_state(require_active=False)
        path = render_compose(state)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
