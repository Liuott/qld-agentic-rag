"""Backwards-compatible alias of :mod:`src.eval.eval_ragas`.

``eval_newv.py`` used to be a 660-line copy of ``eval_ragas.py`` with
hand-formatted code and the self-evaluation loop bolted on.  The two files had
already started to drift (the copy lost ``_export_per_row_ragas`` and changed
the ``is_bad_answer`` rule), so keeping them in sync by hand was a bug farm.

This module now only flips the loop on by default, which is the behaviour the
old script had.  Prefer::

    python -m src.eval.eval_ragas --loop --judge --limit 30
"""

from __future__ import annotations

from collections.abc import Sequence

from src.eval.eval_ragas import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shared evaluator with the self-evaluation loop enabled."""
    return _main(argv, default_loop=True)


if __name__ == "__main__":
    raise SystemExit(main())
