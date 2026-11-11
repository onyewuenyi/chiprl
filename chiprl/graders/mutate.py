"""Syntactic RTL mutants. Used for (a) grader golden tests and (b) the Phase 2 DV env (kill rate).

Operators: arithmetic/relational/logical flips, constant off-by-one, stuck-at on assignments,
and swapped if/else branches would come later. Each mutant changes exactly one site.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

_OPS = [
    (r"(?<![<>=!+])\+(?![+=])", "-"),
    (r"(?<![<>=!-])-(?![-=>])", "+"),
    (r"==", "!="),
    (r"!=", "=="),
    (r"(?<![<=])<(?![<=])", "<="),
    (r"<=(?=\s*\d)", "<"),
    (r"(?<![>=])>(?![>=])", ">="),
    (r"&&", "||"),
    (r"\|\|", "&&"),
    (r"(?<![&])&(?![&])", "|"),
    (r"(?<![|])\|(?![|])", "&"),
    (r"\^", "&"),
    (r"~(?=\w)", ""),
    (r"!(?=[\w(])", ""),
]
_NUM = re.compile(r"(?<![\w'.])(\d+)(?![\w'.:])|(\d+)'([hdb])([0-9a-fA-F_]+)")


def _body_span(src: str) -> tuple[int, int]:
    """Mutate only after the port list so we never break the interface."""
    start = src.find(");") + 2
    return start, len(src)


def _strip(src: str) -> str:
    """Blank out comments (same length) so mutants never land inside them."""
    src = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group()), src, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group()), src)


def mutants(src: str) -> Iterator[tuple[str, str]]:
    lo, hi = _body_span(src)
    code = _strip(src)
    seen = set()
    for pat, rep in _OPS:
        for m in re.finditer(pat, code[lo:hi]):
            a, b = lo + m.start(), lo + m.end()
            # skip nonblocking assignment '<=' when mutating '<'
            if rep in ("<", "<=") and re.match(
                r"\s*\w+(\[[^\]]*\])?\s*$", code[code.rfind("\n", 0, a) + 1 : a]
            ):
                continue
            out = src[:a] + rep + src[b:]
            if out not in seen:
                seen.add(out)
                yield f"op:{m.group()}->{rep}@{a}", out
    for m in _NUM.finditer(code[lo:hi]):
        a, b = lo + m.start(), lo + m.end()
        if m.group(1):
            v = int(m.group(1))
            for nv in (v + 1, max(0, v - 1)):
                if nv != v:
                    out = src[:a] + str(nv) + src[b:]
                    if out not in seen:
                        seen.add(out)
                        yield f"const:{v}->{nv}@{a}", out
        else:
            width, base, digits = m.group(2), m.group(3), m.group(4).replace("_", "")
            v = int(digits, {"h": 16, "d": 10, "b": 2}[base])
            nv = v ^ 1
            fmt = {"h": "x", "d": "d", "b": "b"}[base]
            out = src[:a] + f"{width}'{base}{nv:{fmt}}" + src[b:]
            if out not in seen:
                seen.add(out)
                yield f"const:{width}'{base}{digits}->{nv:{fmt}}@{a}", out
