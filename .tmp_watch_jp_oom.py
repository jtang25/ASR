#!/usr/bin/env python3
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

PID = 769591
POLL_SEC = 10
OOM_PATTERNS = [
    "out of memory",
    "cuda out of memory",
    "killed process",
    "oom-killer",
    "oom kill",
]


def pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def read_cmdline(pid: int):
    data = Path(f"/proc/{pid}/cmdline").read_bytes()
    parts = [p.decode("utf-8", errors="ignore") for p in data.split(b"\0") if p]
    return parts


def get_arg(parts, name, default=None):
    if name in parts:
        i = parts.index(name)
        if i + 1 < len(parts):
            return parts[i + 1]
    return default


def set_arg(parts, name, value):
    if name in parts:
        i = parts.index(name)
        if i + 1 < len(parts):
            parts[i + 1] = str(value)
        else:
            parts.append(str(value))
    else:
        parts.extend([name, str(value)])


def contains_oom_text(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in OOM_PATTERNS)


def scan_kernel_since(since_iso: str) -> str:
    # journalctl may be unavailable/restricted; return empty on failure
    try:
        out = subprocess.check_output(
            ["journalctl", "-k", "--since", since_iso, "--no-pager"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out
    except Exception:
        return ""


def scan_file(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(errors="ignore")
    except Exception:
        pass
    return ""


def main():
    start_wall = datetime.now()
    start_iso = start_wall.strftime("%Y-%m-%d %H:%M:%S")

    watch_log = Path("checkpoints_jp_only_ctc_no_vn/watchdog.log")
    watch_log.parent.mkdir(parents=True, exist_ok=True)

    with watch_log.open("a", encoding="utf-8") as wf:
        wf.write(f"[{datetime.now().isoformat()}] Watchdog start for PID={PID}\n")

    if not pid_alive(PID):
        with watch_log.open("a", encoding="utf-8") as wf:
            wf.write(f"[{datetime.now().isoformat()}] PID {PID} not alive at startup. Exiting.\n")
        return

    parts = read_cmdline(PID)
    cmd_snapshot = " ".join(shlex.quote(x) for x in parts)

    with watch_log.open("a", encoding="utf-8") as wf:
        wf.write(f"[{datetime.now().isoformat()}] cmd={cmd_snapshot}\n")

    while pid_alive(PID):
        time.sleep(POLL_SEC)

    # Process exited: determine whether OOM happened.
    ckpt_dir = get_arg(parts, "--ckpt-dir", "checkpoints_jp_only_ctc_no_vn")
    ckpt = Path(ckpt_dir)
    run_log = ckpt / "run.log"

    kernel_text = scan_kernel_since(start_iso)
    log_text = scan_file(run_log)

    oom = contains_oom_text(kernel_text) or contains_oom_text(log_text)

    with watch_log.open("a", encoding="utf-8") as wf:
        wf.write(f"[{datetime.now().isoformat()}] PID {PID} exited. OOM_detected={oom}\n")

    if not oom:
        with watch_log.open("a", encoding="utf-8") as wf:
            wf.write(f"[{datetime.now().isoformat()}] No OOM signature found; not restarting.\n")
        return

    # Restart with lower batch size (half, floor 32)
    bs = int(get_arg(parts, "--batch-size", "256"))
    new_bs = max(32, bs // 2)

    new_parts = list(parts)
    set_arg(new_parts, "--batch-size", str(new_bs))

    last_pt = ckpt / "last.pt"
    if last_pt.exists():
        set_arg(new_parts, "--resume-path", str(last_pt))

    # Ensure no stale pid arg in command; we only run argv
    restart_log = ckpt / f"run_autorestart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    out_f = restart_log.open("w", encoding="utf-8")
    proc = subprocess.Popen(new_parts, stdout=out_f, stderr=subprocess.STDOUT)

    (ckpt / "watchdog.restarted.pid").write_text(str(proc.pid), encoding="utf-8")

    with watch_log.open("a", encoding="utf-8") as wf:
        wf.write(
            f"[{datetime.now().isoformat()}] Restarted with batch_size={new_bs} pid={proc.pid} log={restart_log}\n"
        )


if __name__ == "__main__":
    main()
