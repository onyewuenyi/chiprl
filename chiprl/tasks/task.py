"""Task dataclass and the VerilogEval v2 (spec-to-rtl) loader."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Task:
    task_id: str
    family: str  # chiprl/tasks/families.py FAMILIES, or "unlabeled"
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


_MANIFEST = Path(__file__).with_name("manifests") / "verilogeval_v2.json"
_FAMILY_CACHE: dict[str, str] | None = None


def family_of(task_id: str) -> str:
    """Structural family from the committed split manifest (chiprl/tasks/splits.py), else "unlabeled".

    Families come from the synthesized reference, not from the task name, which is usually an
    exam id. See chiprl/tasks/families.py.
    """
    global _FAMILY_CACHE
    if _FAMILY_CACHE is None:
        try:
            rows = json.loads(_MANIFEST.read_text())["tasks"]
            _FAMILY_CACHE = {tid: r["family"] for tid, r in rows.items()}
        except (OSError, ValueError, KeyError):
            _FAMILY_CACHE = {}
    return _FAMILY_CACHE.get(task_id, "unlabeled")


def load_verilogeval_dir(d: Path) -> Task:
    """Fixture layout: <dir>/{prompt.txt, ref.sv, test.sv}."""
    ref = (d / "ref.sv").read_text()
    spec = (d / "prompt.txt").read_text()
    return Task(
        task_id=d.name,
        family=family_of(d.name),
        spec=spec,
        ports=parse_ports(ref),
        ref_rtl=ref,
        testbench=(d / "test.sv").read_text(),
    )


# Upstream tasks that cannot score a submission meaningfully. Excluded from every count.
#   Prob099: found by grading all 156 references (2026-11-18); the reference fails to compile.
#   Prob053: found by differential simulation (2026-11-18). The input is x until the stimulus first
#     drives it, so the reference computes out <= x ^ out at the first edge and is x from then on;
#     VerilogEval counts an x reference as matching anything, so 99 of 100 samples accept any
#     value. An all-zeros module passes, and so does out <= in & out. A free reward for a policy.
QUARANTINE: dict[str, str] = {
    "Prob099_m2014_q6c": "testbench connects ports Y2 and Y4, which the reference module does not have",
    "Prob053_m2014_q4d": "reference output is x after the first edge, so the testbench accepts any design",
}


def load_verilogeval_dataset(
    dataset_dir: Path, limit: int | None = None, include_quarantined: bool = False
) -> list[Task]:
    """Upstream layout: dataset_spec-to-rtl/ProbNNN_name_{prompt.txt,ref.sv,test.sv}."""
    tasks = []
    for prompt in sorted(dataset_dir.glob("*_prompt.txt")):
        stem = prompt.name[: -len("_prompt.txt")]
        if stem in QUARANTINE and not include_quarantined:
            continue
        ref = (dataset_dir / f"{stem}_ref.sv").read_text()
        spec = prompt.read_text()
        tasks.append(
            Task(
                task_id=stem,
                family=family_of(stem),
                spec=spec,
                ports=parse_ports(ref),
                ref_rtl=ref,
                testbench=(dataset_dir / f"{stem}_test.sv").read_text(),
            )
        )
        if limit and len(tasks) >= limit:
            break
    return tasks
