"""Job supervisor that runs inside the Colab kernel.

start() launches the job as a subprocess and returns at once. A daemon thread in
the kernel then writes run_dir/status.json every `poll` seconds with job state,
GPU load and the last log line. It kills a job whose log and GPU have both been
quiet for `stall_min` minutes, and releases the VM when a finished job is not
fetched within `guard_min` minutes, so a lost local watcher cannot burn units.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time


def _gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        util, used, total = (float(x) for x in out[0].split(","))
        return {"gpu_util": util, "gpu_mem_mb": used, "gpu_mem_total_mb": total}
    except Exception:
        return {"gpu_util": None}


def _last_line(path, nbytes=4000):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - nbytes))
            tail = handle.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""
    lines = [line for line in tail.replace("\r", "\n").splitlines() if line.strip()]
    return lines[-1][-300:] if lines else ""


def _write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(data, handle)
    os.replace(tmp, path)


def _others_running(run_dir):
    """True if another supervised run on this VM reported running in the last 2 minutes."""
    import glob
    for path in glob.glob("/content/run*/status.json"):
        if os.path.dirname(path) == os.path.abspath(run_dir):
            continue
        try:
            with open(path) as handle:
                st = json.load(handle)
        except Exception:
            continue
        if st.get("state") == "running" and time.time() - st.get("now", 0) < 120:
            return True
    return False


def _unassign(reason, run_dir):
    if _others_running(run_dir):
        return False
    _write(os.path.join(run_dir, "unassign.json"), {"reason": reason, "time": time.time()})
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as exc:  # keep the reason on disk for the watcher
        _write(os.path.join(run_dir, "unassign_error.json"), {"error": repr(exc)})
    return True


def _kill(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        time.sleep(10)


def _current(run_dir, token):
    try:
        with open(os.path.join(run_dir, "generation")) as handle:
            return handle.read().strip() == token
    except FileNotFoundError:
        return False


def _loop(proc, run_dir, log_path, guard_min, stall_min, poll, token, keep_min):
    started = time.time()
    last_size, last_change = -1, started
    state, rc, done_at = "running", None, None
    status_path = os.path.join(run_dir, "status.json")
    while True:
        if not _current(run_dir, token):
            return  # a newer job owns this run_dir now
        now = time.time()
        size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        if size != last_size:
            last_size, last_change = size, now
        gpu = _gpu()
        if state == "running":
            rc = proc.poll()
            if rc is not None:
                state, done_at = ("done" if rc == 0 else "failed"), now
            elif (now - last_change) / 60 >= stall_min and (gpu.get("gpu_util") or 0) < 5:
                _kill(proc)
                state, rc, done_at = "stalled", proc.poll(), time.time()
        fetched = os.path.exists(os.path.join(run_dir, "FETCHED"))
        keep = os.path.exists(os.path.join(run_dir, "KEEP"))
        status = {
            "state": state, "rc": rc, "pid": proc.pid, "started": started, "now": now,
            "elapsed_min": round((now - started) / 60, 2), "log_bytes": size,
            "log_quiet_min": round((now - last_change) / 60, 2),
            "last_line": _last_line(log_path), "fetched": fetched, **gpu,
        }
        if done_at is not None:
            status["since_done_min"] = round((now - done_at) / 60, 2)
        _write(status_path, status)
        if state != "running":
            if keep:
                if now - done_at > keep_min * 60 and _unassign("keep_expired", run_dir):
                    return
            elif fetched and now - done_at > 120:
                if _unassign("fetched", run_dir):
                    return
            elif now - done_at > guard_min * 60:
                if _unassign("guard", run_dir):
                    return
        time.sleep(poll)


def start(cmd, run_dir="/content/run", guard_min=60, stall_min=45, poll=20, env=None, keep_min=40):
    """Launch `cmd` (list of args) in the background and supervise it."""
    os.makedirs(run_dir, exist_ok=True)
    token = "%d-%d" % (time.time() * 1000, os.getpid())
    with open(os.path.join(run_dir, "generation"), "w") as handle:
        handle.write(token)
    for stale in ("status.json", "FETCHED", "KEEP", "unassign.json", "unassign_error.json"):
        path = os.path.join(run_dir, stale)
        if os.path.exists(path):
            os.remove(path)
    log_path = os.path.join(run_dir, "job.log")
    log_file = open(log_path, "ab")
    job_env = dict(os.environ)
    job_env.update(env or {})
    job_env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable if part == "python" else part for part in cmd]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=run_dir,
                            start_new_session=True, env=job_env)
    thread = threading.Thread(target=_loop, args=(proc, run_dir, log_path, guard_min, stall_min, poll, token, keep_min),
                              daemon=True, name="paiec-supervisor")
    thread.start()
    return proc.pid
