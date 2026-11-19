"""Design families from the synthesized reference, not from task names.

VerilogEval task names are mostly exam ids (`m2014_q4i`, `ece241_2013_q7`), so a family read off the
name is noise, and a held-out-family split built on it means nothing. The family here comes from
what the reference turns into after `proc; flatten; opt`, plus one spec signal:

  fsm           the spec describes a state machine (state machine, FSM, Moore, Mealy, state
                diagram, state transition, one-hot). Checked first, whatever the structure:
                VerilogEval includes FSM next-state logic written as a combinational module.
  comb_logic    no flops, no arithmetic, no muxes or comparators (gates, vectors, wires, K-maps)
  comb_mux      no flops, muxes or comparators (case statements, priority logic, muxes)
  comb_arith    no flops, arithmetic cells (adders, popcount)
  seq_register  flops, no arithmetic (flip-flops, shift registers, edge detectors, LFSRs, automata)
  seq_counter   flops and arithmetic (counters, timers, predictors)

"next state" alone is not an FSM signal: cellular automata and LFSRs are described that way.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from ..tools._run import run, which, yosys_read_sv

FAMILIES = ("comb_logic", "comb_mux", "comb_arith", "seq_register", "seq_counter", "fsm")

_FSM_SPEC = re.compile(
    r"state machine|\bfsm\b|moore|mealy|state diagram|state transition|state-assigned|one-hot", re.I
)
_ARITH = {"$add", "$sub", "$mul", "$alu", "$neg", "$macc", "$div", "$mod"}
_MUX = {"$mux", "$pmux", "$shiftx", "$bmux", "$demux", "$shift", "$shl", "$shr", "$sshl", "$sshr"}
_CMP = {"$eq", "$ne", "$lt", "$le", "$gt", "$ge", "$eqx", "$nex"}


def structural_features(ref_rtl: str, top: str) -> dict[str, int] | None:
    """Cell counts of the reference after proc/flatten/opt. None if Yosys is missing or fails."""
    exe = which("yosys")
    if not exe:
        return None
    with tempfile.TemporaryDirectory(prefix="chiprl_feat_") as d:
        wd = Path(d)
        (wd / "g.sv").write_text(ref_rtl.replace("RefModule", top))
        (wd / "f.ys").write_text(
            f"{yosys_read_sv('g.sv', top)}\nhierarchy -top {top}\nproc; flatten; opt; wreduce; opt_clean\n"
            "write_json f.json\n"
        )
        rc, _, _ = run([exe, "-q", "-s", "f.ys"], wd, "f.log", timeout_s=60)
        if rc != 0 or not (wd / "f.json").exists():
            return None
        mods = json.loads((wd / "f.json").read_text())["modules"]
    types: Counter[str] = Counter()
    ff_bits = 0
    for m in mods.values():
        for cell in m.get("cells", {}).values():
            types[cell["type"]] += 1
            if re.search(r"ff|dlatch", cell["type"]):
                ff_bits += len(cell["connections"].get("Q", []))
    return {
        "ff_bits": ff_bits,
        "arith": sum(v for k, v in types.items() if k in _ARITH),
        "mux": sum(v for k, v in types.items() if k in _MUX),
        "cmp": sum(v for k, v in types.items() if k in _CMP),
        "cells": sum(types.values()),
    }


def classify(features: dict[str, Any], spec: str) -> str:
    if _FSM_SPEC.search(spec):
        return "fsm"
    if features["ff_bits"] == 0:
        if features["arith"]:
            return "comb_arith"
        return "comb_mux" if features["mux"] + features["cmp"] else "comb_logic"
    return "seq_counter" if features["arith"] else "seq_register"
