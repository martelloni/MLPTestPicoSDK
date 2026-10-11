#!/usr/bin/env python3
"""Build, flash, run, and record the RP2350 RAM-independence benchmarks.

Coordinates the three build configurations from Step 7 of
plan-mlpCoreLocalPrompt.md:

  * build-core0  -- MEML_MLP_RUNS_ON_CORE=0 (MLP pinned to core 0)
  * build-core1  -- MEML_MLP_RUNS_ON_CORE=1 (MLP pinned to core 1)
  * build-naive  -- MEML_MLP_RUNS_ON_CORE unset (no per-core placement scheme)

For each configuration this script:
  1. Configures + builds with CMake/Ninja (CMAKE_BUILD_TYPE=Release,
     RUN_TESTS_OR_BENCHMARKS=benchmarks).
  2. Flashes the resulting .elf with `picotool load -f`.
  3. Opens a serial connection to the board, answers the firmware's two
     "press any key" prompts, and waits for the benchmark to finish
     (~15 minutes per configuration).
  4. Saves the raw UART transcript to benchmark_results/logs/ and parses it
     for the two RAMIndependence conditions (RAM-flooding other core vs.
     dormant other core), appending one CSV row per condition to
     benchmark_results/ram_independence_results.csv.
  5. Builds the same configuration again with RUN_TESTS_OR_BENCHMARKS=profile
     (tests/ProfileForwardBackward.cpp's standalone per-layer forward/backward
     DWT cycle-count profiler), flashes and runs it, saves its own UART
     transcript to benchmark_results/logs/, and appends one CSV row per
     profiled stage to benchmark_results/profiling_results.csv.

Each condition's per-session (per training-repeat) loss and accuracy are
recorded verbatim (session_losses / session_accuracies, ';'-joined) as the
"golden numerics" of that commit: sessions use fixed seeds, so a change that
claims to be bit-exact must reproduce them exactly (plan-mlpBetterOptimisation.md
Step 0). Run-level metrics main.cpp prints (MLP core timing, other-core
flood/dormant stats, correctness/timing-independence verdicts) are kept too.

Profile runs additionally record the placement (address and address % 16) of
every hot kernel and of each layer's weight/gradient/input buffers in
benchmark_results/placement_results.csv, so cycle counts can be correlated
with SRAM bank phase.

Usage:
    source .venv/bin/activate
    python RunAndSaveBenchmarks.py                     # all 3 configs
    python RunAndSaveBenchmarks.py --configs core0      # just one
    python RunAndSaveBenchmarks.py --engine baseline    # old mlp/MLP.h engine (benchmark only)
    python RunAndSaveBenchmarks.py --engine both        # static + baseline configs
    python RunAndSaveBenchmarks.py --self-test          # parser smoke test, no hardware

See --help for the rest of the flags (serial port, picotool path, timeouts).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import serial  # pyserial
except ImportError:
    serial = None

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_SERIAL_PORT = "/dev/tty.usbmodem2101"
DEFAULT_BAUDRATE = 115200
DEFAULT_PICOTOOL = Path.home() / ".pico-sdk" / "picotool" / "2.3.0" / "picotool" / "picotool"
DEFAULT_RUN_TIMEOUT_S = 25 * 60  # each run takes ~15 min; leave headroom
DEFAULT_PROFILE_TIMEOUT_S = 10 * 60  # per-layer passes + 3 production epochs + 3 counter epochs, ~2 min; headroom on top
PORT_REAPPEAR_TIMEOUT_S = 30.0
HEARTBEAT_INTERVAL_S = 30.0
KEY_RESEND_INTERVAL_S = 2.0

RESULTS_DIR = REPO_ROOT / "benchmark_results"
LOG_DIR = RESULTS_DIR / "logs"
CSV_PATH = RESULTS_DIR / "ram_independence_results.csv"
PROFILE_CSV_PATH = RESULTS_DIR / "profiling_results.csv"
PLACEMENT_CSV_PATH = RESULTS_DIR / "placement_results.csv"
CONTENTION_CSV_PATH = RESULTS_DIR / "contention_results.csv"

# Mirrors main.cpp's kEpochsPerSession, which is not itself printed over UART.
EXPECTED_EPOCHS_PER_SESSION = 10

BENCHMARK_STARTED_MARKER = "Starting RAM independence test..."
END_MARKER = "Test completed."

# Profile builds (RUN_TESTS_OR_BENCHMARKS=profile) only ever print one
# "Forward/backward cycle profile (...)" report and then sit forever in a
# bare tight_loop_contents() with no further output and no "Test completed."
# line -- so the report's own header doubles as the gate-cleared marker, and
# its closing line doubles as the end-of-transcript marker.
PROFILE_STARTED_MARKER = "Forward/backward cycle profile"
# Printed last by RunForwardBackwardProfile(), after the per-batch stage and
# placement blocks that follow the "full pass" stat line.
PROFILE_END_MARKER = "Profile complete."

# Bold cyan for this script's own status messages, so they stand out from
# passed-through cmake/ninja/picotool output and raw UART lines, which are
# printed as-is. Disabled automatically when stdout isn't a real terminal
# (e.g. redirected to a file) so logs don't fill up with escape codes.
_USE_COLOR = sys.stdout.isatty()
_BOLD_CYAN = "\033[1;36m"
_RESET = "\033[0m"


def log(message: str) -> None:
    """Print one of the script's own status lines, styled to stand out."""
    if _USE_COLOR:
        print(f"{_BOLD_CYAN}{message}{_RESET}", flush=True)
    else:
        print(message, flush=True)


CSV_FIELDS = [
    "timestamp",
    "git_commit",
    "build_config",
    "condition",
    "clock_freq_mhz",
    "training_repeats",
    "epochs_per_session",
    "mlp_avg_time_us",
    "mlp_max_time_us",
    "other_core_flood_iterations_consumed",
    "other_core_fill_time_us_mean",
    "other_core_fill_time_us_max",
    "other_core_iterations",
    "other_core_iteration_cap",
    "correctness_identical",
    "session_losses",
    "session_accuracies",
    "timing_avg_diff_pct",
    "timing_max_diff_pct",
    "timing_tolerance_pct",
    "timing_verdict",
    "log_file",
]

RE_CLOCK = re.compile(r"Running RAM independence test with clock frequency:\s*(\d+)\s*MHz")
RE_SESSION = re.compile(r"session \d+: loss=")
RE_SESSION_VALUES = re.compile(r"session \d+: loss=(\S+) accuracy=(\S+)")
RE_MLP = re.compile(r"MLP core: avg_time=([0-9.]+) us, max_time=([0-9.]+) us")
RE_FLOOD_ITERS = re.compile(r"other core: flood iterations consumed until done=([0-9.eE+-]+)")
RE_FLOOD_FILL = re.compile(r"other core: fill_time avg=([0-9.]+) us, max=([0-9.]+) us")
RE_DORMANT_ITERS = re.compile(r"other core: iterations=(\d+) \(cap=(\d+), expected 1\)")
RE_CORRECTNESS = re.compile(r"Correctness independence \(bit-identical loss/accuracy\):\s*(PASS|FAIL)")
RE_TIMING_PINNED = re.compile(
    r"Timing independence \(avg_diff=([0-9.]+)%, max_diff=([0-9.]+)%, tolerance=([0-9.]+)%\):\s*(PASS|FAIL)"
)
RE_TIMING_NAIVE = re.compile(
    r"Timing interference observed without per-core placement \(expected\): "
    r"avg_diff=([0-9.]+)%, max_diff=([0-9.]+)%, tolerance=([0-9.]+)%"
)
RE_FLOOD_BLOCK = re.compile(
    r"Flooding run \(other core flooding RAM\):(.*?)Dormant run \(other core fully idle in WFE\):", re.S
)
RE_DORMANT_BLOCK = re.compile(
    r"Dormant run \(other core fully idle in WFE\):(.*?)Correctness independence", re.S
)

PROFILE_CSV_FIELDS = [
    "timestamp",
    "git_commit",
    "build_config",
    "iterations",
    "stage",
    "avg_cycles",
    "max_cycles",
    "log_file",
]

RE_PROFILE_HEADER = re.compile(r"Forward/backward cycle profile \((\d+) iterations")
# Matches tests/ProfileForwardBackward.cpp's PrintStage() output:
#   "  %-28s avg=%9.1f cycles   max=%8lu cycles"
RE_PROFILE_STAGE = re.compile(r"^\s*(\S(?:.*\S)?)\s+avg=\s*([0-9.]+) cycles\s+max=\s*(\d+) cycles\s*$")


@dataclass
class BuildConfig:
    name: str
    core_value: str  # "", "0", or "1" -- passed verbatim as -DMEML_MLP_RUNS_ON_CORE=
    build_dir_name: str
    run_mode: str = "benchmarks"  # RUN_TESTS_OR_BENCHMARKS value: "benchmarks" or "profile"
    engine: str = "static"  # MEML_MLP_ENGINE value: "static" (smlp::StaticMLP) or "baseline" (old mlp/MLP.h)

    @property
    def build_dir(self) -> Path:
        return REPO_ROOT / self.build_dir_name

    @property
    def elf_path(self) -> Path:
        return self.build_dir / "MLPTestPicoSDK.elf"


BUILD_CONFIGS = [
    BuildConfig("core0", "0", "build-core0"),
    BuildConfig("core1", "1", "build-core1"),
    BuildConfig("naive", "", "build-naive"),
]

# Same placements, but running the experiment on the old heap-based mlp/MLP.h
# engine (MEML_MLP_ENGINE=baseline) via MLPOpticalRecognitionBaseline. Benchmark
# only: the DWT profiler is written against the static engine's layer types, so
# these have no profile counterpart. Opt in with --engine baseline|both.
BASELINE_BUILD_CONFIGS = [
    BuildConfig("core0-baseline", "0", "build-core0-baseline", engine="baseline"),
    BuildConfig("core1-baseline", "1", "build-core1-baseline", engine="baseline"),
    BuildConfig("naive-baseline", "", "build-naive-baseline", engine="baseline"),
]

# Same three MEML_MLP_RUNS_ON_CORE selections, built with
# RUN_TESTS_OR_BENCHMARKS=profile instead, in their own build directories so
# the benchmark and profile binaries for a given config don't clobber each
# other's CMake cache.
PROFILE_BUILD_CONFIGS = [
    BuildConfig("core0", "0", "build-core0-profile", run_mode="profile"),
    BuildConfig("core1", "1", "build-core1-profile", run_mode="profile"),
    BuildConfig("naive", "", "build-naive-profile", run_mode="profile"),
]


class BenchmarkError(RuntimeError):
    pass


def get_git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def ensure_git_clean() -> None:
    """Refuse to run unless the working tree exactly matches HEAD.

    Results are tagged with the current commit hash so runs can be traced
    back to the exact source that produced them; that traceability is worthless
    if uncommitted changes (tracked or untracked, anything `git status` would
    show) could have affected the build.
    """
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    if result.stdout.strip():
        raise BenchmarkError(
            "Git working tree has uncommitted changes and/or untracked files "
            "(commit, stash, or remove them first so results can be tied to an exact commit):\n"
            f"{result.stdout}"
        )


def run_cmd(cmd: list) -> None:
    log(f"$ {' '.join(str(c) for c in cmd)}")
    # No capture_output: the child inherits our stdout/stderr directly, so its
    # own output (cmake/ninja build progress) streams live regardless of
    # Python's buffering.
    subprocess.run([str(c) for c in cmd], check=True)


def configure(cfg: BuildConfig) -> None:
    cfg.build_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "cmake",
        "-S", str(REPO_ROOT),
        "-B", str(cfg.build_dir),
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DRUN_TESTS_OR_BENCHMARKS={cfg.run_mode}",
        f"-DMEML_MLP_RUNS_ON_CORE={cfg.core_value}",
        f"-DMEML_MLP_ENGINE={cfg.engine}",
    ])


def build(cfg: BuildConfig) -> None:
    run_cmd(["cmake", "--build", str(cfg.build_dir)])


def flash(picotool: str, elf_file: Path) -> None:
    run_cmd([picotool, "load", "-f", str(elf_file)])


def normalize_serial_port(port: str) -> str:
    """Prefer the /dev/cu.* twin of a /dev/tty.* USB-CDC serial device on macOS.

    macOS's tty.* nodes are meant for dial-in modems and wait on carrier
    detect; for a Pico's USB CDC port this means opening tty.* can appear to
    succeed (DTR reads back True) while never actually delivering any bytes.
    The cu.* ("call-up") node for the exact same device does not have this
    problem. Swap transparently so a tty.* path passed on the command line
    still works.
    """
    if sys.platform == "darwin" and "/tty." in port:
        candidate = port.replace("/tty.", "/cu.")
        if Path(candidate).exists():
            log(f"Using {candidate} instead of {port} (macOS tty.* can silently drop USB CDC data)")
            return candidate
    return port


def wait_for_port(port: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(port).exists():
            return
        time.sleep(0.5)
    raise BenchmarkError(f"Serial port {port} did not reappear within {timeout:.0f}s after flashing")


def capture_run(
    port: str,
    baudrate: int,
    overall_timeout_s: float,
    log_path: Path,
    start_marker: str = BENCHMARK_STARTED_MARKER,
    end_marker: str = END_MARKER,
) -> str:
    """Answer the firmware's prompts and capture the full UART transcript.

    main.cpp gates on two separate getchar_timeout_us() calls ("press any
    key to continue", then "press any key to start benchmarks"), each only
    reachable once stdio_usb_connected() (the USB CDC line state's DTR bit)
    is true. Waiting to *see* each prompt's text before replying loses the
    race after a real flash+reboot cycle: the firmware can print a prompt
    (and TinyUSB can drop it -- no host was reading the endpoint yet) before
    our freshly-opened port has actually started polling it, so the reply
    never gets sent and the board hangs forever waiting for a keystroke.
    Instead, send a keystroke unconditionally as soon as the port is open,
    then keep resending on an interval until real benchmark output shows up
    -- whichever gate the firmware is actually sitting at when a keystroke
    arrives, it unblocks; a keystroke arriving between gates or after both
    are cleared is simply ignored by the firmware (getchar_timeout_us is
    never called again once training starts).
    """
    port = normalize_serial_port(port)
    wait_for_port(port, PORT_REAPPEAR_TIMEOUT_S)
    time.sleep(1.0)  # let the freshly re-enumerated port settle

    ser = serial.Serial(port, baudrate=baudrate, timeout=1.0)
    ser.dtr = True
    ser.rts = True

    lines: list[str] = []
    gates_cleared = False
    start_time = time.monotonic()
    deadline = start_time + overall_timeout_s
    last_output_time = start_time
    last_heartbeat_time = start_time
    last_key_sent_time = 0.0  # 0 forces an immediate first send
    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            while time.monotonic() < deadline:
                now = time.monotonic()
                if not gates_cleared and now - last_key_sent_time >= KEY_RESEND_INTERVAL_S:
                    ser.write(b"a")
                    last_key_sent_time = now

                raw = ser.readline()
                now = time.monotonic()
                if not raw:
                    # No line within the 1s read timeout: print a periodic
                    # heartbeat so a long silent stretch (e.g. mid-training)
                    # is visibly "still waiting", not indistinguishable from hung.
                    if now - last_heartbeat_time >= HEARTBEAT_INTERVAL_S:
                        elapsed = now - start_time
                        silent_for = now - last_output_time
                        log(
                            f"  ...still waiting (elapsed {elapsed:.0f}s, "
                            f"no UART output for {silent_for:.0f}s)"
                        )
                        last_heartbeat_time = now
                    continue

                last_output_time = now
                last_heartbeat_time = now
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                print(f"  UART| {line}", flush=True)
                log_f.write(line + "\n")
                log_f.flush()
                lines.append(line)

                if not gates_cleared and start_marker in line:
                    gates_cleared = True
                elif end_marker in line:
                    return "\n".join(lines)
    finally:
        ser.close()
    raise BenchmarkError(f"Timed out after {overall_timeout_s:.0f}s waiting for '{end_marker}'")


def parse_transcript(text: str, build_config: str, timestamp: str, git_commit: str, log_file: str) -> list[dict]:
    clock_match = RE_CLOCK.search(text)
    clock_mhz = clock_match.group(1) if clock_match else ""

    flood_match = RE_FLOOD_BLOCK.search(text)
    dormant_match = RE_DORMANT_BLOCK.search(text)
    if not flood_match or not dormant_match:
        raise BenchmarkError("Could not locate flooding/dormant run sections in UART transcript")
    flood_text = flood_match.group(1)
    dormant_text = dormant_match.group(1)

    correctness_match = RE_CORRECTNESS.search(text)
    if not correctness_match:
        raise BenchmarkError("Could not locate correctness-independence verdict in UART transcript")
    correctness = correctness_match.group(1)

    timing_pinned = RE_TIMING_PINNED.search(text)
    timing_naive = RE_TIMING_NAIVE.search(text)
    if timing_pinned:
        avg_diff, max_diff, tolerance, verdict = timing_pinned.groups()
    elif timing_naive:
        avg_diff, max_diff, tolerance = timing_naive.groups()
        verdict = "EXPECTED-DEVIATION"
    else:
        raise BenchmarkError("Could not locate timing independence/interference line in UART transcript")

    def mlp_metrics(block: str) -> tuple:
        m = RE_MLP.search(block)
        if not m:
            raise BenchmarkError("Missing 'MLP core: avg_time=...' line in a run block")
        return m.group(1), m.group(2)

    flood_avg, flood_max = mlp_metrics(flood_text)
    dormant_avg, dormant_max = mlp_metrics(dormant_text)

    flood_iters_match = RE_FLOOD_ITERS.search(flood_text)
    flood_fill_match = RE_FLOOD_FILL.search(flood_text)
    dormant_iters_match = RE_DORMANT_ITERS.search(dormant_text)

    common = dict(
        timestamp=timestamp,
        git_commit=git_commit,
        build_config=build_config,
        clock_freq_mhz=clock_mhz,
        epochs_per_session=EXPECTED_EPOCHS_PER_SESSION,
        correctness_identical=correctness,
        timing_avg_diff_pct=avg_diff,
        timing_max_diff_pct=max_diff,
        timing_tolerance_pct=tolerance,
        timing_verdict=verdict,
        log_file=log_file,
    )

    def session_values(block: str) -> dict:
        found = RE_SESSION_VALUES.findall(block)
        return dict(
            session_losses=";".join(loss for loss, _ in found),
            session_accuracies=";".join(acc for _, acc in found),
        )

    flood_row = dict(common)
    flood_row.update(
        condition="flooding",
        **session_values(flood_text),
        training_repeats=len(RE_SESSION.findall(flood_text)),
        mlp_avg_time_us=flood_avg,
        mlp_max_time_us=flood_max,
        other_core_flood_iterations_consumed=flood_iters_match.group(1) if flood_iters_match else "",
        other_core_fill_time_us_mean=flood_fill_match.group(1) if flood_fill_match else "",
        other_core_fill_time_us_max=flood_fill_match.group(2) if flood_fill_match else "",
        other_core_iterations="",
        other_core_iteration_cap="",
    )

    dormant_row = dict(common)
    dormant_row.update(
        condition="dormant",
        **session_values(dormant_text),
        training_repeats=len(RE_SESSION.findall(dormant_text)),
        mlp_avg_time_us=dormant_avg,
        mlp_max_time_us=dormant_max,
        other_core_flood_iterations_consumed="",
        other_core_fill_time_us_mean="",
        other_core_fill_time_us_max="",
        other_core_iterations=dormant_iters_match.group(1) if dormant_iters_match else "",
        other_core_iteration_cap=dormant_iters_match.group(2) if dormant_iters_match else "",
    )

    return [flood_row, dormant_row]


def migrate_csv_header(path: Path, fields: list) -> None:
    """If `path` exists with a different header than `fields`, rewrite it with
    the new header, keeping every existing row (new columns left blank), so
    appending rows with the new schema never mis-aligns against an old header.
    """
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == fields:
            return
        old_rows = list(reader)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(old_rows)
    log(f"Migrated {path.name} to the current column set ({len(old_rows)} existing row(s) kept)")


def append_rows_to_csv(rows: list) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    migrate_csv_header(CSV_PATH, CSV_FIELDS)
    csv_is_new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        if csv_is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_profile_transcript(text: str, build_config: str, timestamp: str, git_commit: str, log_file: str) -> list[dict]:
    header_match = RE_PROFILE_HEADER.search(text)
    if not header_match:
        raise BenchmarkError("Could not locate 'Forward/backward cycle profile (...)' header in UART transcript")
    iterations = header_match.group(1)

    rows = []
    for line in text.splitlines():
        stage_match = RE_PROFILE_STAGE.match(line)
        if not stage_match:
            continue
        stage, avg_cycles, max_cycles = stage_match.groups()
        rows.append(dict(
            timestamp=timestamp,
            git_commit=git_commit,
            build_config=build_config,
            iterations=iterations,
            stage=stage,
            avg_cycles=avg_cycles,
            max_cycles=max_cycles,
            log_file=log_file,
        ))

    if not rows:
        raise BenchmarkError("Could not locate any per-stage 'avg=... max=...' lines in UART transcript")
    return rows


PLACEMENT_CSV_FIELDS = [
    "timestamp",
    "git_commit",
    "build_config",
    "kind",
    "name",
    "address",
    "addr_mod16",
    "bank",
    "size_bytes",
    "log_file",
]

# tests/ProfileForwardBackward.cpp's PrintPlacement():
#   "  placement data <name> addr=0x20001234 mod16=4"
RE_PLACEMENT_DATA = re.compile(r"^\s*placement data (\S+)\s+addr=0x([0-9a-fA-F]+) mod16=(\d+)\s*$")
# Hot code whose address the firmware cannot report reliably (the compiler may
# emit out-of-line `.constprop` clones, or inline the kernel into the caller):
# taken from the ELF symbol table instead. Demangled-name match.
RE_HOT_CODE_SYMBOL = re.compile(
    r"StaticLayer<.*>::(forward|AccumulateGradients|ApplyAccumulatedGradients|"
    r"GetGradSumSquared|InitGradientAccumulators)\(|"
    r"MLPOpticalRecognition::(Train|Initialise)\(|for_each_layer<|"
    r"test::profile::.*(RunLayerPass|RunBatchStages|ProbeBody|MeasureTrain|RunTrainOnce)\("
)


def bank_of(address: int) -> str:
    if 0x20000000 <= address < 0x20040000:
        return "core0-bank"
    if 0x20040000 <= address < 0x20080000:
        return "core1-bank"
    if 0x20080000 <= address < 0x20082000:
        return "scratch"
    return "other"


def parse_placement_transcript(text: str, build_config: str, timestamp: str, git_commit: str, log_file: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        m = RE_PLACEMENT_DATA.match(line)
        if not m:
            continue
        name, addr, mod16 = m.groups()
        address = int(addr, 16)
        rows.append(dict(
            timestamp=timestamp, git_commit=git_commit, build_config=build_config,
            kind="data", name=name, address=f"0x{address:08x}", addr_mod16=mod16,
            bank=bank_of(address), size_bytes="", log_file=log_file,
        ))
    return rows


def elf_code_placement_rows(elf: Path, nm: str, build_config: str, timestamp: str, git_commit: str, log_file: str,
                            objdump: str | None = None) -> list[dict]:
    """Address / address%16 / size of every hot-kernel code symbol in the ELF
    (kind=code), plus, when `objdump` is given, the head address, length and
    addr%16 of every inner loop inside those symbols (kind=loop)."""
    out = subprocess.run([nm, "-C", "-S", str(elf)], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:  # symbols without a size column (-S) are not code we care about
            continue
        addr_s, size_s, kind, name = parts
        if kind not in ("T", "t", "W", "w"):
            continue
        if not RE_HOT_CODE_SYMBOL.search(name):
            continue
        address = int(addr_s, 16)
        rows.append(dict(
            timestamp=timestamp, git_commit=git_commit, build_config=build_config,
            kind="code", name=name, address=f"0x{address:08x}", addr_mod16=str(address % 16),
            bank=bank_of(address), size_bytes=str(int(size_s, 16)), log_file=log_file,
        ))
    if objdump:
        dis = subprocess.run([objdump, "-d", "-C", str(elf)], capture_output=True, text=True, check=True).stdout
        for name, head, length in find_loops(dis, RE_HOT_CODE_SYMBOL):
            rows.append(dict(
                timestamp=timestamp, git_commit=git_commit, build_config=build_config,
                kind="loop", name=f"{name} loop@0x{head:08x}", address=f"0x{head:08x}",
                addr_mod16=str(head % 16), bank=bank_of(head), size_bytes=str(length), log_file=log_file,
            ))
    return rows


RE_DIS_FUNC = re.compile(r"^([0-9a-fA-F]+) <(.+)>:$")
RE_DIS_INSN = re.compile(r"^\s*([0-9a-fA-F]+):\t([0-9a-fA-F ]+)\t(\S+)(?:\s+(.*))?$")
RE_DIS_BRANCH = re.compile(r"^(?:b|cbz|cbnz)(?:eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le)?(?:\.[nw])?$")


def find_loops(disassembly: str, wanted) -> list[tuple[str, int, int]]:
    """(symbol, head address, length in bytes) for every backward branch whose
    target lies inside the same symbol, for symbols whose demangled name
    matches `wanted`. A loop head targeted by several backward branches is
    reported once, with the longest extent. `bl` calls are not loops."""
    loops: dict[tuple[str, int], int] = {}
    func = None
    func_start = 0
    for line in disassembly.splitlines():
        h = RE_DIS_FUNC.match(line)
        if h:
            func_start = int(h.group(1), 16)
            func = h.group(2) if wanted.search(h.group(2)) else None
            continue
        if func is None:
            continue
        m = RE_DIS_INSN.match(line)
        if not m or not RE_DIS_BRANCH.match(m.group(3)):
            continue
        addr = int(m.group(1), 16)
        size = len(m.group(2).replace(" ", "")) // 2
        t = re.match(r"^([0-9a-fA-F]+) <", (m.group(4) or "").strip())
        if not t:
            continue
        target = int(t.group(1), 16)
        if func_start <= target <= addr:
            key = (func, target)
            loops[key] = max(loops.get(key, 0), addr + size - target)
    return [(f, head, length) for (f, head), length in sorted(loops.items(), key=lambda kv: kv[0][1])]


def find_tool(build_dir: Path, cache_var: str, fallback: str) -> str:
    cache = build_dir / "CMakeCache.txt"
    if cache.exists():
        m = re.search(rf"^{cache_var}:FILEPATH=(.+)$", cache.read_text(), re.M)
        if m and Path(m.group(1)).exists():
            return m.group(1)
    return shutil.which(fallback) or fallback


def find_nm(build_dir: Path) -> str:
    return find_tool(build_dir, "CMAKE_NM", "arm-none-eabi-nm")


CONTENTION_CSV_FIELDS = [
    "timestamp",
    "git_commit",
    "build_config",
    "stage",
    "sram_bank",
    "contested_avg",
    "access_avg",
    "valid",
    "log_file",
]

# tests/ProfileForwardBackward.cpp's PrintContention():
#   "  contention stage=<s> bank=SRAM<n> contested=<v|-> access=<v|->"
RE_CONTENTION = re.compile(
    r"^\s*contention stage=(\S+) bank=SRAM(\d+) contested=([0-9.]+|-) access=([0-9.]+|-)\s*$"
)
# "  contention pass=N full_pass_cycles=X reference=Y valid=yes|NO" -- a pass whose
# full-pass cycles drift from the counter-free reference by >0.1% means the
# counter reads perturbed the measurement, so that pass's rows are not usable.
RE_CONTENTION_PASS = re.compile(r"^\s*contention pass=(\d+) .* valid=(yes|NO)\s*$")
# Which pass covers which bank in the per-layer harness (see kPassSel).
_PASS_OF_BANK = {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3}
PER_LAYER_STAGES = {
    "layer0.forward", "layer1.forward", "layer2.forward", "compute_loss",
    "layer2.AccumulateGradients", "layer1.AccumulateGradients", "layer0.AccumulateGradients",
}


def parse_contention_transcript(text: str, build_config: str, timestamp: str, git_commit: str, log_file: str) -> list[dict]:
    pass_valid = {}
    merged: dict[tuple[str, int], dict] = {}
    for line in text.splitlines():
        pm = RE_CONTENTION_PASS.match(line)
        if pm:
            pass_valid[int(pm.group(1))] = pm.group(2) == "yes"
            continue
        m = RE_CONTENTION.match(line)
        if not m:
            continue
        stage, bank, contested, access = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        row = merged.setdefault((stage, bank), dict(
            timestamp=timestamp, git_commit=git_commit, build_config=build_config,
            stage=stage, sram_bank=str(bank), contested_avg="", access_avg="", valid="", log_file=log_file))
        if contested != "-":
            row["contested_avg"] = contested
        if access != "-":
            row["access_avg"] = access
    rows = list(merged.values())
    for row in rows:
        if row["stage"] in PER_LAYER_STAGES:
            ok = pass_valid.get(_PASS_OF_BANK[int(row["sram_bank"])], True)
            row["valid"] = "yes" if ok else "NO"
    return rows


def append_contention_rows_to_csv(rows: list) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_is_new = not CONTENTION_CSV_PATH.exists()
    with open(CONTENTION_CSV_PATH, "a", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CONTENTION_CSV_FIELDS)
        if csv_is_new:
            writer.writeheader()
        writer.writerows(rows)


def append_placement_rows_to_csv(rows: list) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    migrate_csv_header(PLACEMENT_CSV_PATH, PLACEMENT_CSV_FIELDS)
    csv_is_new = not PLACEMENT_CSV_PATH.exists()
    with open(PLACEMENT_CSV_PATH, "a", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=PLACEMENT_CSV_FIELDS)
        if csv_is_new:
            writer.writeheader()
        writer.writerows(rows)


def append_profile_rows_to_csv(rows: list) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_is_new = not PROFILE_CSV_PATH.exists()
    with open(PROFILE_CSV_PATH, "a", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=PROFILE_CSV_FIELDS)
        if csv_is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


SAMPLE_TRANSCRIPT_PINNED = """\
Press any key to continue...
Press any key to start benchmarks...

Starting RAM independence test...
Running RAM independence test with clock frequency: 150 MHz

Flooding run (other core flooding RAM):
  session 0: loss=0.123456 accuracy=0.900000
  session 1: loss=0.111111 accuracy=0.910000
  MLP core: avg_time=1234.56 us, max_time=1300.00 us
  other core: flood iterations consumed until done=1.234500e+06
  other core: fill_time avg=12.34 us, max=15.67 us

Dormant run (other core fully idle in WFE):
  session 0: loss=0.123456 accuracy=0.900000
  session 1: loss=0.111111 accuracy=0.910000
  MLP core: avg_time=1230.00 us, max_time=1290.00 us
  other core: iterations=1 (cap=1, expected 1)

Correctness independence (bit-identical loss/accuracy): PASS
Timing independence (avg_diff=0.37%, max_diff=0.78%, tolerance=1%): PASS

Test completed.
"""

SAMPLE_TRANSCRIPT_NAIVE = SAMPLE_TRANSCRIPT_PINNED.replace(
    "Timing independence (avg_diff=0.37%, max_diff=0.78%, tolerance=1%): PASS",
    "Timing interference observed without per-core placement (expected): "
    "avg_diff=12.30%, max_diff=18.20%, tolerance=1%",
)

SAMPLE_PROFILE_TRANSCRIPT = """\
Press any key to continue...
Press any key to start profiling...

Forward/backward cycle profile (100 iterations, network 64-64-32-10):
Forward:
  layer0.forward               avg=   1234.5 cycles   max=    1300 cycles
  layer1.forward               avg=    567.8 cycles   max=     600 cycles
  layer2.forward               avg=    123.4 cycles   max=     150 cycles
  forward total                avg=   1925.7 cycles   max=    2050 cycles
Loss:
  compute_loss                 avg=     45.6 cycles   max=      60 cycles
Backward:
  layer2.AccumulateGradients   avg=    234.5 cycles   max=     260 cycles
  layer1.AccumulateGradients   avg=    678.9 cycles   max=     700 cycles
  layer0.AccumulateGradients   avg=   1345.6 cycles   max=    1400 cycles
  backward total               avg=   2259.0 cycles   max=    2360 cycles
Full forward+loss+backward total:
  full pass                    avg=   4230.3 cycles   max=    4470 cycles
Per-batch stages (once per 128 samples; 20 iterations):
  batch.zero_grads             avg=    900.0 cycles   max=     900 cycles
  batch.grad_norm              avg=   1500.5 cycles   max=    1510 cycles
  batch.rmsprop.layer0         avg=  90000.0 cycles   max=   90010 cycles
  batch.rmsprop.layer1         avg=  45000.0 cycles   max=   45010 cycles
  batch.rmsprop.layer2         avg=   7000.0 cycles   max=    7010 cycles
  batch.rmsprop total          avg= 142000.0 cycles   max=  142030 cycles
  batch total                  avg= 144400.5 cycles   max=  144450 cycles
  batch total per sample       avg=    1128.0 cycles   max=    1128 cycles
Bus contention per sample (per-layer harness; counters outside the timed bracket):
  contention pass=1 full_pass_cycles=4230.0 reference=4230.3 valid=yes
  contention stage=layer1.forward bank=SRAM0 contested=0.0 access=-
  contention stage=layer1.forward bank=SRAM3 contested=12.5 access=-
  contention pass=2 full_pass_cycles=4300.0 reference=4230.3 valid=NO
  contention stage=layer1.forward bank=SRAM4 contested=25.0 access=-
  contention pass=3 full_pass_cycles=4230.1 reference=4230.3 valid=yes
  contention stage=layer1.forward bank=SRAM0 contested=- access=77.0
  contention stage=layer1.forward bank=SRAM8 contested=0.0 access=-
Production path (MLPOpticalRecognition::Train on this core, min of 3):
  train.eval_pass              avg=154881639.0 cycles   max=154917299 cycles
  train.epoch                  avg=757294578.0 cycles   max=757294848 cycles
  train.epoch_minus_eval_per_sample avg= 157575.0 cycles   max=  157575 cycles
Bus contention, production path (totals over Initialise + Train(1); polled from the other core):
  contention stage=train.epoch bank=SRAM4 contested=12466738.0 access=-
Placement (net and kernel code addresses; code addresses are added by
RunAndSaveBenchmarks.py from the ELF symbol table):
  placement data net                          addr=0x20008a30 mod16=0
  placement data layer0.m_weights             addr=0x20008a34 mod16=4
Profile complete.

"""


def run_self_test() -> None:
    pinned_rows = parse_transcript(SAMPLE_TRANSCRIPT_PINNED, "core0", "20260101_000000", "deadbeef", "sample.log")
    assert len(pinned_rows) == 2
    flood, dormant = pinned_rows
    assert flood["condition"] == "flooding"
    assert flood["training_repeats"] == 2
    assert flood["mlp_avg_time_us"] == "1234.56"
    assert flood["other_core_flood_iterations_consumed"] == "1.234500e+06"
    assert flood["other_core_fill_time_us_mean"] == "12.34"
    assert dormant["condition"] == "dormant"
    assert dormant["mlp_avg_time_us"] == "1230.00"
    assert dormant["other_core_iterations"] == "1"
    assert dormant["other_core_iteration_cap"] == "1"
    assert flood["correctness_identical"] == "PASS"
    assert flood["session_losses"] == "0.123456;0.111111"
    assert flood["session_accuracies"] == "0.900000;0.910000"
    assert flood["timing_verdict"] == "PASS"
    assert flood["timing_avg_diff_pct"] == "0.37"

    naive_rows = parse_transcript(SAMPLE_TRANSCRIPT_NAIVE, "naive", "20260101_000000", "deadbeef", "sample.log")
    assert naive_rows[0]["timing_verdict"] == "EXPECTED-DEVIATION"
    assert naive_rows[0]["timing_avg_diff_pct"] == "12.30"

    profile_rows = parse_profile_transcript(
        SAMPLE_PROFILE_TRANSCRIPT, "core0", "20260101_000000", "deadbeef", "sample_profile.log"
    )
    assert len(profile_rows) == 21
    assert profile_rows[0]["stage"] == "layer0.forward"
    assert profile_rows[0]["avg_cycles"] == "1234.5"
    assert profile_rows[0]["max_cycles"] == "1300"
    assert profile_rows[0]["iterations"] == "100"
    assert profile_rows[9]["stage"] == "full pass"
    assert profile_rows[9]["max_cycles"] == "4470"
    assert profile_rows[17]["stage"] == "batch total per sample"
    assert profile_rows[15]["stage"] == "batch.rmsprop total"
    assert profile_rows[-1]["stage"] == "train.epoch_minus_eval_per_sample"
    assert profile_rows[-3]["stage"] == "train.eval_pass"
    assert profile_rows[-2]["avg_cycles"] == "757294578.0"

    contention_rows = parse_contention_transcript(
        SAMPLE_PROFILE_TRANSCRIPT, "core0", "20260101_000000", "deadbeef", "sample_profile.log"
    )
    by_key = {(r["stage"], r["sram_bank"]): r for r in contention_rows}
    assert by_key[("layer1.forward", "3")]["contested_avg"] == "12.5"
    assert by_key[("layer1.forward", "3")]["valid"] == "yes"
    assert by_key[("layer1.forward", "4")]["valid"] == "NO"  # pass 2 drifted
    assert by_key[("layer1.forward", "0")]["contested_avg"] == "0.0"  # pass 1 contested ...
    assert by_key[("layer1.forward", "0")]["access_avg"] == "77.0"    # ... merged with pass 3 access
    assert by_key[("train.epoch", "4")]["contested_avg"] == "12466738.0"
    assert by_key[("train.epoch", "4")]["valid"] == ""

    sample_dis = """\
20000000 <test::profile::(anonymous namespace)::RunLayerPass(int, LayerPass&)>:
20000000:\tb510      \tpush\t{r4, lr}
20000002:\tbf00      \tnop
20000004:\t3001      \tadds\tr0, #1
20000006:\t2864      \tcmp\tr0, #100
20000008:\td1fc      \tbne.n\t20000004 <test::profile::(anonymous namespace)::RunLayerPass(int, LayerPass&)+0x4>
2000000a:\tf000 f805 \tbl\t20000018 <helper>
2000000e:\te7f8      \tb.n\t20000002 <test::profile::(anonymous namespace)::RunLayerPass(int, LayerPass&)+0x2>
20000018 <helper>:
20000018:\t4770      \tbx\tlr
"""
    loops = find_loops(sample_dis, RE_HOT_CODE_SYMBOL)
    assert [(h, n) for _, h, n in loops] == [(0x20000002, 14), (0x20000004, 6)], loops

    placement_rows = parse_placement_transcript(
        SAMPLE_PROFILE_TRANSCRIPT, "core0", "20260101_000000", "deadbeef", "sample_profile.log"
    )
    assert [r["name"] for r in placement_rows] == ["net", "layer0.m_weights"]
    assert placement_rows[1]["addr_mod16"] == "4"
    assert placement_rows[1]["address"] == "0x20008a34"
    assert placement_rows[1]["bank"] == "core0-bank"

    log("Self-test OK: parser correctly extracts pinned, naive, profile, placement, contention and loop data.")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--configs", nargs="+", choices=[c.name for c in BUILD_CONFIGS + BASELINE_BUILD_CONFIGS],
        default=None,
        help="Which build configurations to run (default: the three configs of the --engine selection)",
    )
    parser.add_argument(
        "--engine", choices=("static", "baseline", "both"), default="static",
        help="Which MLP engine's configs to run when --configs is not given: 'static' "
             "(smlp::StaticMLP, default), 'baseline' (old mlp/MLP.h, benchmark only -- no profile "
             "run), or 'both'",
    )
    parser.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT, help="Serial device for the board's USB CDC UART")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--picotool", default=str(DEFAULT_PICOTOOL), help="Path to picotool if not on PATH")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_RUN_TIMEOUT_S,
        help="Max seconds to wait for one flashed benchmark run to print 'Test completed.'",
    )
    parser.add_argument(
        "--profile-timeout", type=float, default=DEFAULT_PROFILE_TIMEOUT_S,
        help="Max seconds to wait for one flashed profile run to print its 'full pass' stats line",
    )
    parser.add_argument(
        "--skip-profile", action="store_true",
        help="Skip the RUN_TESTS_OR_BENCHMARKS=profile build/flash/run entirely, for each selected config",
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true",
        help="Skip the RUN_TESTS_OR_BENCHMARKS=benchmarks build/flash/run entirely, for each selected "
             "config (useful since the benchmark run takes ~15 min/config vs. seconds for the profile run)",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Reuse each configuration's existing build directory (benchmark and profile alike) "
             "instead of reconfiguring+rebuilding",
    )
    parser.add_argument(
        "--skip-flash", action="store_true",
        help="Skip picotool flashing (assume the board is already running the right firmware); "
             "only meaningful with a single --configs entry and one of --skip-profile/--skip-benchmark",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run the UART transcript parser against bundled sample transcripts and exit (no hardware needed)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.configs is None:
        args.configs = (
            ([c.name for c in BUILD_CONFIGS] if args.engine in ("static", "both") else [])
            + ([c.name for c in BASELINE_BUILD_CONFIGS] if args.engine in ("baseline", "both") else [])
        )

    if args.self_test:
        run_self_test()
        return 0

    if args.skip_benchmark and args.skip_profile:
        sys.exit("--skip-benchmark and --skip-profile together would skip everything; drop one")

    if serial is None:
        sys.exit(
            "pyserial is not installed. Create/activate the project venv first:\n"
            "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        )

    picotool = shutil.which("picotool") or args.picotool
    if not Path(picotool).exists() and shutil.which(picotool) is None:
        sys.exit(f"picotool not found at '{picotool}' and not on PATH")

    try:
        ensure_git_clean()
    except BenchmarkError as e:
        sys.exit(str(e))
    git_commit = get_git_commit_hash()
    log(f"Git commit: {git_commit}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    configs_by_name = {c.name: c for c in BUILD_CONFIGS + BASELINE_BUILD_CONFIGS}
    profile_configs_by_name = {c.name: c for c in PROFILE_BUILD_CONFIGS}
    for name in args.configs:
        if not args.skip_benchmark:
            cfg = configs_by_name[name]
            log(f"\n=== {cfg.name} (MEML_MLP_RUNS_ON_CORE={cfg.core_value!r}) ===")

            if not args.skip_build:
                configure(cfg)
                build(cfg)

            if not cfg.elf_path.exists():
                sys.exit(f"Expected built ELF not found: {cfg.elf_path}")

            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = LOG_DIR / f"{cfg.name}_{timestamp}.log"

            if not args.skip_flash:
                flash(picotool, cfg.elf_path)

            log(f"Waiting for board on {args.serial_port} (timeout {args.timeout:.0f}s)...")
            transcript = capture_run(args.serial_port, args.baudrate, args.timeout, log_path)

            rows = parse_transcript(transcript, cfg.name, timestamp, git_commit, str(log_path.relative_to(REPO_ROOT)))
            append_rows_to_csv(rows)
            log(f"Saved {len(rows)} row(s) for {cfg.name} to {CSV_PATH}; raw log at {log_path}")

        if args.skip_profile:
            continue
        if name not in profile_configs_by_name:
            log(f"{name}: no profile build for the baseline engine; skipping profile step")
            continue

        profile_cfg = profile_configs_by_name[name]
        log(f"\n=== {profile_cfg.name} profile (MEML_MLP_RUNS_ON_CORE={profile_cfg.core_value!r}) ===")

        if not args.skip_build:
            configure(profile_cfg)
            build(profile_cfg)

        if not profile_cfg.elf_path.exists():
            sys.exit(f"Expected built ELF not found: {profile_cfg.elf_path}")

        profile_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_log_path = LOG_DIR / f"{profile_cfg.name}_profile_{profile_timestamp}.log"

        if not args.skip_flash:
            flash(picotool, profile_cfg.elf_path)

        log(f"Waiting for board on {args.serial_port} (timeout {args.profile_timeout:.0f}s)...")
        profile_transcript = capture_run(
            args.serial_port, args.baudrate, args.profile_timeout, profile_log_path,
            start_marker=PROFILE_STARTED_MARKER, end_marker=PROFILE_END_MARKER,
        )

        profile_rows = parse_profile_transcript(
            profile_transcript, profile_cfg.name, profile_timestamp, git_commit,
            str(profile_log_path.relative_to(REPO_ROOT)),
        )
        append_profile_rows_to_csv(profile_rows)
        profile_log_rel = str(profile_log_path.relative_to(REPO_ROOT))
        placement_rows = parse_placement_transcript(
            profile_transcript, profile_cfg.name, profile_timestamp, git_commit, profile_log_rel)
        placement_rows += elf_code_placement_rows(
            profile_cfg.elf_path, find_nm(profile_cfg.build_dir), profile_cfg.name,
            profile_timestamp, git_commit, profile_log_rel, objdump=find_tool(profile_cfg.build_dir, "CMAKE_OBJDUMP", "arm-none-eabi-objdump"))
        append_placement_rows_to_csv(placement_rows)
        contention_rows = parse_contention_transcript(
            profile_transcript, profile_cfg.name, profile_timestamp, git_commit, profile_log_rel)
        append_contention_rows_to_csv(contention_rows)
        invalid = [r for r in contention_rows if r["valid"] == "NO"]
        if invalid:
            log(f"WARNING: {len(invalid)} per-layer contention row(s) came from a pass whose cycle count drifted "
                f">0.1% from the counter-free reference; discard them and re-run")
        log(f"Saved {len(contention_rows)} contention row(s) to {CONTENTION_CSV_PATH}")
        log(f"Saved {len(placement_rows)} placement row(s) for {profile_cfg.name} to {PLACEMENT_CSV_PATH}")
        log(f"Saved {len(profile_rows)} row(s) for {profile_cfg.name} to {PROFILE_CSV_PATH}; raw log at {profile_log_path}")

    result_paths = []
    if not args.skip_benchmark:
        result_paths.append(str(CSV_PATH))
    if not args.skip_profile:
        result_paths.append(str(PROFILE_CSV_PATH))
        result_paths.append(str(PLACEMENT_CSV_PATH))
        result_paths.append(str(CONTENTION_CSV_PATH))
    log(f"\nAll done. Results in {' and '.join(result_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
