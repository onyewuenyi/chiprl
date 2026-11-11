"""Task dataclass and the VerilogEval v2 (spec-to-rtl) loader."""

from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Task:
    task_id: str
    family: str  # arithmetic | fsm | datapath | counter | comb ... used for splits
    spec: str  # what the policy sees
    ports: tuple[tuple[str, str, int], ...]  # (dir, name, width)
    ref_rtl: str  # hidden. module RefModule
    testbench: str  # hidden. instantiates RefModule + TopModule
    top: str = "TopModule"
    source: str = "verilogeval_v2"

    @property
    def clock(self) -> str | None:
        names = {p[1] for p in self.ports}
        return "clk" if "clk" in names else None

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for s in (self.spec, self.ref_rtl, self.testbench):
            h.update(s.encode())
        return h.hexdigest()[:16]


_PORT_RE = re.compile(r"^\s*(input|output)\s+(?:logic\s+|reg\s+|wire\s+)?(\[[^\]]+\])?\s*(\w+)", re.M)


def parse_ports(ref_rtl: str) -> tuple[tuple[str, str, int], ...]:
    header = ref_rtl.split(");", 1)[0]
    ports = []
    for d, rng, name in _PORT_RE.findall(header):
        w = 1
        if rng:
            hi, lo = (int(x) for x in rng.strip("[]").split(":"))
            w = abs(hi - lo) + 1
        ports.append((d, name, w))
    return tuple(ports)


_FAMILY_HINTS = [
    ("fsm", r"fsm|state|moore|mealy"),
    ("counter", r"count"),
    ("arith", r"add|sub|mul|popcount|sum"),
    ("shift", r"shift|lfsr|rotate"),
    ("mux", r"mux"),
    ("gates", r"gate|xor|and|or|not|wire"),
]


def guess_family(task_id: str, spec: str) -> str:
    key = task_id.lower()
    for fam, pat in _FAMILY_HINTS:
        if re.search(pat, key):
            return fam
    return "other"


def load_verilogeval_dir(d: Path) -> Task:
    """Fixture layout: <dir>/{prompt.txt, ref.sv, test.sv}."""
    ref = (d / "ref.sv").read_text()
    spec = (d / "prompt.txt").read_text()
    return Task(
        task_id=d.name,
        family=guess_family(d.name, spec),
        spec=spec,
        ports=parse_ports(ref),
        ref_rtl=ref,
        testbench=(d / "test.sv").read_text(),
    )


def load_verilogeval_dataset(dataset_dir: Path, limit: int | None = None) -> list[Task]:
    """Upstream layout: dataset_spec-to-rtl/ProbNNN_name_{prompt.txt,ref.sv,test.sv}."""
    tasks = []
    for prompt in sorted(dataset_dir.glob("*_prompt.txt")):
        stem = prompt.name[: -len("_prompt.txt")]
        ref = (dataset_dir / f"{stem}_ref.sv").read_text()
        spec = prompt.read_text()
        tasks.append(
            Task(
                task_id=stem,
                family=guess_family(stem, spec),
                spec=spec,
                ports=parse_ports(ref),
                ref_rtl=ref,
                testbench=(dataset_dir / f"{stem}_test.sv").read_text(),
            )
        )
        if limit and len(tasks) >= limit:
            break
    return tasks
