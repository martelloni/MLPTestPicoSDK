#!/usr/bin/env python3
"""Validate RP2350 core-0/core-1 memory placement invariants on a built ELF.

Checks the four guarantees Step 1 of plan-mlpCoreLocalPrompt.md establishes:
  1. Core-1 sections (.time_critical.core1.code, .core1_bank, .core1_stack)
     fit inside the reserved core-1 SRAM window [0x20040000, 0x20080000).
  2. The core-0 bank (.core0_bank) fits inside the default SRAM window
     [0x20000000, 0x20040000).
  3. The flash-resident dataset section (.flash) has no SRAM execution
     address.
  4. Every mlp/ hot-path symbol (SMLP_CODE_ATTR-tagged, "smlp::"-namespaced)
     lands in the bank the selected MEML_MLP_RUNS_ON_CORE value implies, and
     none of it leaks into the other core's bank. This is the check that
     would have caught a hot-path function silently missing its placement
     hook, or a project TU including mlp/*.h before binding MemoryDefs.hpp.

Usage:
    validate_memory_placement.py <elf> --core "" --readelf <path> --nm <path>
"""

import argparse
import re
import subprocess
import sys

RAM0_BASE, RAM0_END = 0x20000000, 0x20040000
RAM1_BASE, RAM1_END = 0x20040000, 0x20080000

CORE0_SECTIONS = (".core0_bank",)
CORE1_SECTIONS = (".time_critical.core1.code", ".core1_bank", ".core1_stack")


def run(tool, args):
    result = subprocess.run([tool, *args], capture_output=True, text=True, check=True)
    return result.stdout


def parse_sections(readelf_output):
    """Return {name: (addr, size)} from `readelf -SW` output."""
    sections = {}
    # Columns: [Nr] Name Type Addr Off Size ES Flg Lk Inf Al
    pattern = re.compile(
        r"^\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)"
    )
    for line in readelf_output.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        name, addr, size = m.groups()
        sections[name] = (int(addr, 16), int(size, 16))
    return sections


def parse_symbols(nm_output):
    """Return a list of (address, mangled_name) from `nm` output."""
    symbols = []
    for line in nm_output.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        addr_str, _kind, name = parts[0], parts[1], parts[2]
        try:
            addr = int(addr_str, 16)
        except ValueError:
            continue
        symbols.append((addr, name))
    return symbols


def fail(errors, message):
    errors.append(message)


def check_bank_bounds(sections, name, lo, hi, errors):
    if name not in sections:
        fail(errors, f"expected section '{name}' is missing from the ELF")
        return
    addr, size = sections[name]
    if size == 0:
        return  # an empty (unselected) bank is fine
    end = addr + size
    if addr < lo or end > hi:
        fail(
            errors,
            f"section '{name}' [{addr:#x}, {end:#x}) is not within "
            f"the expected window [{lo:#x}, {hi:#x})",
        )


def check_flash_outside_ram(sections, errors):
    if ".flash" not in sections:
        fail(errors, "expected section '.flash' is missing from the ELF")
        return
    addr, size = sections[".flash"]
    end = addr + size
    if not (end <= RAM0_BASE or addr >= RAM1_END):
        fail(errors, f"section '.flash' [{addr:#x}, {end:#x}) overlaps an SRAM window")


_CTOR_DTOR_MARKER = re.compile(r"(C[123]E|D[012]E)")


def _is_ctor_or_dtor(mangled_name):
    """Itanium ABI marks constructors C1/C2/C3 and destructors D0/D1/D2 right
    before the parameter encoding. Constructors/destructors for static-storage
    objects always run once, during the single-core boot sequence that runs
    every .init_array entry on core 0 before core 1 is ever launched -- they
    were never part of the plan's hot-path list and don't need SMLP_CODE_ATTR.
    """
    return bool(_CTOR_DTOR_MARKER.search(mangled_name))


def check_mlp_symbol_placement(symbols, core, errors):
    """Every symbol actually DEFINED in namespace smlp must sit in the selector's
    bank. Matching must be a prefix check (_ZN4smlp...), not a substring check:
    a symbol like std::get<0>(tuple<smlp::StaticLayer<...>>) mentions "smlp" in
    its mangled template arguments but is a libstdc++ symbol we have no
    attribute hook into -- it is not something SMLP_CODE_ATTR could ever tag,
    so it must not be flagged as a coverage regression.
    """
    if core == "1":
        expected_lo, expected_hi = RAM1_BASE, RAM1_END
        forbidden_lo, forbidden_hi = RAM0_BASE, RAM0_END
    else:
        expected_lo, expected_hi = RAM0_BASE, RAM0_END
        forbidden_lo, forbidden_hi = RAM1_BASE, RAM1_END

    mlp_symbols = [
        (addr, name) for addr, name in symbols
        if name.startswith("_ZN4smlp") and not _is_ctor_or_dtor(name)
    ]
    if not mlp_symbols:
        fail(errors, "no smlp:: symbols found in the ELF -- did the build actually link mlp/?")
        return

    for addr, name in mlp_symbols:
        if forbidden_lo <= addr < forbidden_hi:
            fail(
                errors,
                f"mlp hot-path symbol '{name}' at {addr:#x} lands in the "
                f"OTHER core's bank [{forbidden_lo:#x}, {forbidden_hi:#x}) -- "
                "SMLP_CODE_ATTR coverage or include order regression",
            )
        elif not (expected_lo <= addr < expected_hi):
            fail(
                errors,
                f"mlp hot-path symbol '{name}' at {addr:#x} is outside both "
                "known SRAM banks",
            )


def check_placement_probe(symbols, core, errors):
    expected_lo, expected_hi = (RAM1_BASE, RAM1_END) if core == "1" else (RAM0_BASE, RAM0_END)
    # MLPOpticalRecognitionTest.cpp's g_experiment is the real experiment's
    # State placement probe, repurposed from Step 1.5's retired MLPPlacementTest.
    # Only linked into "tests" builds (RUN_TESTS_OR_BENCHMARKS=tests); a
    # "benchmarks" build never compiles this translation unit, so the caller
    # skips this check entirely for that mode rather than treating its
    # absence as a regression.
    wanted = ("g_experiment",)
    found = [(addr, name) for addr, name in symbols if any(w in name for w in wanted)]
    if not found:
        fail(errors, "MLPOpticalRecognitionTest symbols not found in the ELF -- was it linked in?")
        return
    for addr, name in found:
        if not (expected_lo <= addr < expected_hi):
            fail(
                errors,
                f"placement-probe symbol '{name}' at {addr:#x} is outside the "
                f"selected core's bank [{expected_lo:#x}, {expected_hi:#x})",
            )


def check_benchmark_placement(symbols, core, errors):
    """"benchmarks" mode only: TestRAMIndependence.hpp's build-owned
    mlp_experiment_ instance must land in the selected core's bank (like
    check_placement_probe's g_experiment for "tests" mode), and its
    ram_flooder_ instance must land in the OTHER core's bank -- the flooder
    and the MLP experiment are always on opposite cores, driven by the same
    MEML_MLP_RUNS_ON_CORE selector, per Step 6 of plan-mlpCoreLocalPrompt.md.
    """
    mlp_lo, mlp_hi = (RAM1_BASE, RAM1_END) if core == "1" else (RAM0_BASE, RAM0_END)
    other_lo, other_hi = (RAM0_BASE, RAM0_END) if core == "1" else (RAM1_BASE, RAM1_END)

    # Exclude compiler-generated dynamic-initialization guard variables
    # (Itanium ABI "_ZGV" prefix): GCC never applies a data symbol's own
    # section attribute to its guard flag, so the guard lands wherever the
    # linker's default rules put ordinary .bss -- it is bookkeeping, not
    # the actual placed object, and checking it would be a false positive.
    mlp_symbols = [(addr, name) for addr, name in symbols if "mlp_experiment_" in name and "_ZGV" not in name]
    flooder_symbols = [(addr, name) for addr, name in symbols if "ram_flooder_" in name and "_ZGV" not in name]

    if not mlp_symbols:
        fail(errors, "TestRAMIndependence's mlp_experiment_ symbol not found in the ELF -- was it linked in?")
    for addr, name in mlp_symbols:
        if not (mlp_lo <= addr < mlp_hi):
            fail(
                errors,
                f"benchmark placement-probe symbol '{name}' at {addr:#x} is outside the "
                f"selected core's bank [{mlp_lo:#x}, {mlp_hi:#x})",
            )

    if not flooder_symbols:
        fail(errors, "TestRAMIndependence's ram_flooder_ symbol not found in the ELF -- was it linked in?")
    for addr, name in flooder_symbols:
        if not (other_lo <= addr < other_hi):
            fail(
                errors,
                f"benchmark RAM flooder symbol '{name}' at {addr:#x} is outside the "
                f"OTHER core's bank [{other_lo:#x}, {other_hi:#x}) -- it must stay opposite "
                "the selected MEML_MLP_RUNS_ON_CORE bank",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", help="Path to the built ELF")
    parser.add_argument("--core", default="", help="MEML_MLP_RUNS_ON_CORE value: '', '0', or '1'")
    parser.add_argument("--readelf", default="arm-none-eabi-readelf")
    parser.add_argument("--nm", default="arm-none-eabi-nm")
    parser.add_argument(
        "--mode", default="tests", choices=("tests", "benchmarks"),
        help="RUN_TESTS_OR_BENCHMARKS value: 'tests' links MLPOpticalRecognitionTest.cpp's "
             "g_experiment placement probe; 'benchmarks' does not, so that check is skipped.",
    )
    args = parser.parse_args()

    sections = parse_sections(run(args.readelf, ["-SW", args.elf]))
    symbols = parse_symbols(run(args.nm, [args.elf]))

    errors = []
    check_bank_bounds(sections, ".core0_bank", RAM0_BASE, RAM0_END, errors)
    for name in CORE1_SECTIONS:
        check_bank_bounds(sections, name, RAM1_BASE, RAM1_END, errors)
    check_flash_outside_ram(sections, errors)
    check_mlp_symbol_placement(symbols, args.core, errors)
    if args.mode == "tests":
        check_placement_probe(symbols, args.core, errors)
    else:
        check_benchmark_placement(symbols, args.core, errors)

    if errors:
        print(f"validate_memory_placement: FAIL ({len(errors)} issue(s)) "
              f"for MEML_MLP_RUNS_ON_CORE='{args.core}'", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"validate_memory_placement: PASS for MEML_MLP_RUNS_ON_CORE='{args.core}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
