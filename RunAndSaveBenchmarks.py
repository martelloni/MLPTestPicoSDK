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

Per-session (per training-repeat) loss/accuracy is intentionally NOT
extracted -- only the run-level metrics main.cpp prints (MLP core timing,
other-core flood/dormant stats, correctness/timing-independence verdicts).

Usage:
    source .venv/bin/activate
    python RunAndSaveBenchmarks.py                     # all 3 configs
    python RunAndSaveBenchmarks.py --configs core0      # just one
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

DEFAULT_SERIAL_PORT = "/dev/cu.usbmodem21401"
DEFAULT_BAUDRATE = 115200
DEFAULT_PICOTOOL = Path.home() / ".pico-sdk" / "picotool" / "2.3.0" / "picotool" / "picotool"
DEFAULT_RUN_TIMEOUT_S = 25 * 60  # each run takes ~15 min; leave headroom
PORT_REAPPEAR_TIMEOUT_S = 30.0
HEARTBEAT_INTERVAL_S = 30.0
KEY_RESEND_INTERVAL_S = 2.0

RESULTS_DIR = REPO_ROOT / "benchmark_results"
LOG_DIR = RESULTS_DIR / "logs"
CSV_PATH = RESULTS_DIR / "ram_independence_results.csv"

# Mirrors main.cpp's kEpochsPerSession, which is not itself printed over UART.
EXPECTED_EPOCHS_PER_SESSION = 10

BENCHMARK_STARTED_MARKER = "Starting RAM independence test..."
END_MARKER = "Test completed."

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
    "timing_avg_diff_pct",
    "timing_max_diff_pct",
    "timing_tolerance_pct",
    "timing_verdict",
    "log_file",
]

RE_CLOCK = re.compile(r"Running RAM independence test with clock frequency:\s*(\d+)\s*MHz")
RE_SESSION = re.compile(r"session \d+: loss=")
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


@dataclass
class BuildConfig:
    name: str
    core_value: str  # "", "0", or "1" -- passed verbatim as -DMEML_MLP_RUNS_ON_CORE=
    build_dir_name: str

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
        "-DRUN_TESTS_OR_BENCHMARKS=benchmarks",
        f"-DMEML_MLP_RUNS_ON_CORE={cfg.core_value}",
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


def capture_run(port: str, baudrate: int, overall_timeout_s: float, log_path: Path) -> str:
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

                if not gates_cleared and BENCHMARK_STARTED_MARKER in line:
                    gates_cleared = True
                elif END_MARKER in line:
                    return "\n".join(lines)
    finally:
        ser.close()
    raise BenchmarkError(f"Timed out after {overall_timeout_s:.0f}s waiting for '{END_MARKER}'")


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

    flood_row = dict(common)
    flood_row.update(
        condition="flooding",
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


def append_rows_to_csv(rows: list) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_is_new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
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
    assert flood["timing_verdict"] == "PASS"
    assert flood["timing_avg_diff_pct"] == "0.37"

    naive_rows = parse_transcript(SAMPLE_TRANSCRIPT_NAIVE, "naive", "20260101_000000", "deadbeef", "sample.log")
    assert naive_rows[0]["timing_verdict"] == "EXPECTED-DEVIATION"
    assert naive_rows[0]["timing_avg_diff_pct"] == "12.30"

    log("Self-test OK: parser correctly extracts both pinned and naive transcripts.")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--configs", nargs="+", choices=[c.name for c in BUILD_CONFIGS],
        default=[c.name for c in BUILD_CONFIGS],
        help="Which build configurations to run (default: all three)",
    )
    parser.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT, help="Serial device for the board's USB CDC UART")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--picotool", default=str(DEFAULT_PICOTOOL), help="Path to picotool if not on PATH")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_RUN_TIMEOUT_S,
        help="Max seconds to wait for one flashed run to print 'Test completed.'",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Reuse each configuration's existing build directory instead of reconfiguring+rebuilding",
    )
    parser.add_argument(
        "--skip-flash", action="store_true",
        help="Skip picotool flashing (assume the board is already running the right firmware); "
             "only meaningful with a single --configs entry",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run the UART transcript parser against bundled sample transcripts and exit (no hardware needed)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

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

    configs_by_name = {c.name: c for c in BUILD_CONFIGS}
    for name in args.configs:
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

    log(f"\nAll done. Results in {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
