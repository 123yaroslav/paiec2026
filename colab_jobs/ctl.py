#!/usr/bin/env python3
"""Local control of Colab jobs with a hard compute-unit budget.

new      create a session only while the balance stays above the floor
start    upload files and launch a supervised background job
status   one status line (job state, GPU load, last log line, units)
watch    poll until the job ends, then download outputs and stop the VM
stop     stop a session

The floor is the balance at campaign start minus the units the user allowed.
Every session start and stop is appended to ledger.jsonl next to this file.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "ledger.jsonl")
BUDGET_FILE = os.path.join(HERE, "budget.json")


def now_str():
    return datetime.datetime.now().strftime("%m-%d %H:%M:%S")


def colab(*args, stdin=None, timeout=300):
    try:
        res = subprocess.run(["colab", *args], input=stdin, capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "local timeout"


def ledger(event, **data):
    with open(LEDGER, "a") as handle:
        handle.write(json.dumps({"time": datetime.datetime.now().isoformat(timespec="seconds"),
                                 "event": event, **data}) + "\n")


def usage():
    rc, out, err = colab("usage", timeout=120)
    text = out + err
    bal = re.search(r"Current balance:\s*([0-9.]+)", text)
    rate = re.search(r"Usage rate:\s*([0-9.]+)", text)
    return (float(bal.group(1)) if bal else None, float(rate.group(1)) if rate else None)


# Conservative hourly rates. L4 was measured at 10.82 units per hour on 2026-11-04.
RATES = {"L4": 10.82, "T4": 4.0, "A100": 16.0, "H100": 25.0, "G4": 16.0, None: 1.0}


def own_spent():
    """Units spent by sessions this tool started, from ledger times and rates.

    The account balance is shared with sessions started elsewhere, so it is not
    a measure of this campaign's spend.
    """
    if not os.path.exists(LEDGER):
        return 0.0, {}
    opened, total = {}, 0.0
    for line in open(LEDGER):
        event = json.loads(line)
        when = datetime.datetime.fromisoformat(event["time"])
        name = event.get("session")
        if event["event"] == "new" and event.get("rc") == 0:
            opened[name] = (when, RATES.get(event.get("gpu"), 25.0))
        elif event["event"] in ("stop", "gone") and name in opened:
            start, rate = opened.pop(name)
            total += (when - start).total_seconds() / 3600 * rate
    now = datetime.datetime.now()
    for start, rate in opened.values():
        total += (now - start).total_seconds() / 3600 * rate
    return total, opened


def budget():
    with open(BUDGET_FILE) as handle:
        cfg = json.load(handle)
    return cfg["allowed_units"], cfg


def sessions():
    rc, out, err = colab("sessions", timeout=120)
    return out + err


def cmd_new(args):
    allowed, cfg = budget()
    spent, active = own_spent()
    bal, rate = usage()
    if spent + args.reserve >= allowed:
        sys.exit(f"own spend {spent:.1f} plus reserve {args.reserve} reaches the allowance {allowed}")
    print(f"own spend so far {spent:.1f} of {allowed} units, account balance {bal}")
    extra = ["--gpu", args.gpu] if args.gpu else []
    rc, out, err = colab("new", "-s", args.session, *extra, timeout=900)
    print(out[-2000:], err[-2000:])
    ledger("new", session=args.session, gpu=args.gpu, rc=rc, balance=bal)
    if rc != 0:
        sys.exit(rc)


def remote(session, code, timeout=120):
    return colab("exec", "-s", session, "--timeout", str(timeout), stdin=code, timeout=timeout + 120)


def cmd_start(args):
    # The Jupyter contents API returns HTTP 500 when the target folder is missing.
    parents = sorted({os.path.dirname(spec.split("=", 1)[1]) for spec in args.upload} | {args.run_dir})
    rc, out, err = remote(args.session, f"import os\nfor d in {json.dumps(parents)}: os.makedirs(d, exist_ok=True)\n")
    if rc != 0:
        sys.exit(f"cannot create remote folders: {err[-500:]}")
    for spec in args.upload:
        local, remote_path = spec.split("=", 1)
        rc, out, err = colab("upload", "-s", args.session, local, remote_path, timeout=3600)
        if rc != 0:
            sys.exit(f"upload failed {local}: {err[-500:]}")
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {json.dumps(os.path.dirname(args.supervisor))})\n"
        "import importlib, supervisor\nimportlib.reload(supervisor)\n"
        f"pid = supervisor.start({json.dumps(args.cmd)}, run_dir={json.dumps(args.run_dir)}, "
        f"guard_min={args.guard_min}, stall_min={args.stall_min}, poll={args.poll}, keep_min={args.keep_min})\n"
        "print('STARTED_PID', pid)\n"
    )
    rc, out, err = remote(args.session, code)
    print(out[-1500:], err[-800:] if rc else "")
    if "STARTED_PID" not in out:
        sys.exit("job did not start")
    ledger("start", session=args.session, cmd=args.cmd)


def read_status(session, run_dir):
    code = (
        "import json, os, time\n"
        f"p = {json.dumps(run_dir + '/status.json')}\n"
        "s = json.load(open(p)) if os.path.exists(p) else {'state': 'nostatus'}\n"
        "s['vm_now'] = time.time()\n"
        f"u = {json.dumps(run_dir + '/unassign_error.json')}\n"
        "s['unassign_error'] = open(u).read() if os.path.exists(u) else None\n"
        "print('STATUS_JSON ' + json.dumps(s))\n"
    )
    rc, out, err = remote(session, code, timeout=60)
    for line in out.splitlines():
        if line.startswith("STATUS_JSON "):
            return json.loads(line[len("STATUS_JSON "):]), None
    return None, (err or out)[-400:]


def fmt(st):
    if st.get("state") == "nostatus":
        return "no status file yet"
    gpu = st.get("gpu_util")
    stale = st["vm_now"] - st["now"]
    return (f"{st['state']:<8} {st['elapsed_min']:>7.1f}m gpu={gpu if gpu is not None else '-'}% "
            f"mem={st.get('gpu_mem_mb', '-')}MB quiet={st['log_quiet_min']}m "
            f"stale={stale:.0f}s | {st['last_line'][-140:]}")


def fetch_and_stop(args, st):
    os.makedirs(args.local_dir, exist_ok=True)
    pack = args.run_dir + "/fetch.tar.gz"
    code = (
        "import tarfile, os\n"
        f"paths = {json.dumps(args.fetch + [args.run_dir + '/job.log', args.run_dir + '/status.json'])}\n"
        f"with tarfile.open({json.dumps(pack)}, 'w:gz') as tar:\n"
        "    for p in paths:\n"
        "        if os.path.exists(p): tar.add(p, arcname=os.path.relpath(p, '/content'))\n"
        f"print('PACKED', os.path.getsize({json.dumps(pack)}))\n"
    )
    rc, out, err = remote(args.session, code, timeout=1800)
    print(now_str(), out.strip()[-200:])
    local_pack = os.path.join(args.local_dir, "fetch.tar.gz")
    rc, out, err = colab("download", "-s", args.session, pack, local_pack, timeout=7200)
    ok = rc == 0 and os.path.exists(local_pack)
    print(now_str(), "download", "ok" if ok else f"FAILED {err[-300:]}")
    if ok:
        subprocess.run(["tar", "-xzf", local_pack, "-C", args.local_dir], check=False)
        flag = "/KEEP" if args.keep else "/FETCHED"
        remote(args.session, f"open({json.dumps(args.run_dir + flag)}, 'w').write('1')\n", timeout=60)
    if args.keep:
        print(now_str(), "keeping the VM as requested")
    else:
        rc, out, err = colab("stop", "-s", args.session, timeout=300)
        bal, rate = usage()
        ledger("stop", session=args.session, state=st.get("state"), balance=bal)
        print(now_str(), f"stopped session, balance={bal}")
    return ok


def cmd_watch(args):
    allowed, cfg = budget()
    fails = 0
    while True:
        st, error = read_status(args.session, args.run_dir)
        bal, rate = usage()
        spent, _ = own_spent()
        units = f"own={spent:.1f}/{allowed:.0f}u acct_bal={bal}"
        if st is None:
            fails += 1
            print(now_str(), f"status read failed ({fails}) {error.strip()[-200:]} | {units}", flush=True)
            if args.session not in sessions() and fails >= 2:
                print(now_str(), "SESSION GONE", flush=True)
                ledger("gone", session=args.session, balance=bal)
                sys.exit(2)
            if fails >= 6:
                sys.exit(2)
        else:
            fails = 0
            print(now_str(), fmt(st), "|", units, flush=True)
            if spent >= allowed - 2:
                print(now_str(), "BUDGET FLOOR REACHED, stopping", flush=True)
                fetch_and_stop(args, st)
                sys.exit(4)
            if st.get("state") in ("done", "failed", "stalled"):
                fetch_and_stop(args, st)
                sys.exit({"done": 0, "failed": 1, "stalled": 3}[st["state"]])
            if st.get("state") != "nostatus" and st["vm_now"] - st["now"] > 900:
                print(now_str(), "SUPERVISOR SILENT 15 min, fetching log and stopping", flush=True)
                fetch_and_stop(args, st)
                sys.exit(5)
        time.sleep(args.every)


def cmd_status(args):
    st, error = read_status(args.session, args.run_dir)
    print(fmt(st) if st else f"no status: {error}")
    print("units", usage())


CHUNK = 64 * 1024 * 1024


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cmd_push(args):
    """Upload a large file in 64 MB parts, join on the VM and verify SHA256."""
    import tempfile
    size = os.path.getsize(args.local)
    digest = sha256_file(args.local)
    remote_dir = os.path.dirname(args.remote)
    remote(args.session, f"import os\nos.makedirs({json.dumps(remote_dir)}, exist_ok=True)\n")
    parts = []
    with open(args.local, "rb") as handle, tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(args.local))) as tmp:
        index = 0
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            part_local = os.path.join(tmp, f"part{index:04d}")
            open(part_local, "wb").write(block)
            part_remote = f"{args.remote}.part{index:04d}"
            for attempt in range(4):
                rc, out, err = colab("upload", "-s", args.session, part_local, part_remote, timeout=1800)
                if rc == 0:
                    break
                print(now_str(), f"part {index} retry {attempt}: {err.strip()[-200:]}", flush=True)
            else:
                sys.exit(f"upload of part {index} failed")
            parts.append(part_remote)
            index += 1
            print(now_str(), f"pushed part {index} ({min(index * CHUNK, size) / 1e6:.0f} of {size / 1e6:.0f} MB)", flush=True)
    code = (
        "import hashlib, os\n"
        f"parts = {json.dumps(parts)}\n"
        f"with open({json.dumps(args.remote)}, 'wb') as out:\n"
        "    for p in parts:\n"
        "        out.write(open(p, 'rb').read()); os.remove(p)\n"
        "h = hashlib.sha256()\n"
        f"with open({json.dumps(args.remote)}, 'rb') as f:\n"
        "    for b in iter(lambda: f.read(1 << 20), b''): h.update(b)\n"
        "print('REMOTE_SHA', h.hexdigest())\n"
    )
    rc, out, err = remote(args.session, code, timeout=900)
    ok = f"REMOTE_SHA {digest}" in out
    print(now_str(), "push verified" if ok else f"SHA MISMATCH {out[-300:]} {err[-300:]}")
    if not ok:
        sys.exit(1)


def cmd_pull(args):
    """Split a remote file into 64 MB parts, download them, join and verify SHA256."""
    code = (
        "import hashlib, os\n"
        f"src = {json.dumps(args.remote)}\n"
        "h = hashlib.sha256(); parts = []\n"
        "with open(src, 'rb') as f:\n"
        "    i = 0\n"
        f"    for b in iter(lambda: f.read({CHUNK}), b''):\n"
        "        h.update(b); p = f'{src}.part{i:04d}'; open(p, 'wb').write(b); parts.append(p); i += 1\n"
        "print('PARTS', len(parts)); print('REMOTE_SHA', h.hexdigest())\n"
    )
    rc, out, err = remote(args.session, code, timeout=900)
    nparts = int(re.search(r"PARTS (\d+)", out).group(1))
    remote_sha = re.search(r"REMOTE_SHA ([0-9a-f]{64})", out).group(1)
    os.makedirs(os.path.dirname(os.path.abspath(args.local)), exist_ok=True)
    with open(args.local + ".tmp", "wb") as joined:
        for index in range(nparts):
            part_remote = f"{args.remote}.part{index:04d}"
            part_local = f"{args.local}.part{index:04d}"
            for attempt in range(4):
                rc, out, err = colab("download", "-s", args.session, part_remote, part_local, timeout=1800)
                if rc == 0 and os.path.exists(part_local):
                    break
                print(now_str(), f"part {index} retry {attempt}: {err.strip()[-200:]}", flush=True)
            else:
                sys.exit(f"download of part {index} failed")
            joined.write(open(part_local, "rb").read())
            os.remove(part_local)
            print(now_str(), f"pulled part {index + 1} of {nparts}", flush=True)
    remote(args.session, f"import glob, os\nfor p in glob.glob({json.dumps(args.remote + '.part*')}): os.remove(p)\n")
    if sha256_file(args.local + ".tmp") != remote_sha:
        sys.exit("SHA MISMATCH after pull")
    os.replace(args.local + ".tmp", args.local)
    print(now_str(), "pull verified", args.local)


def cmd_stop(args):
    rc, out, err = colab("stop", "-s", args.session)
    bal, rate = usage()
    ledger("stop", session=args.session, balance=bal)
    print(out, err, "balance", bal)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("new")
    p.add_argument("session")
    p.add_argument("--gpu")
    p.add_argument("--reserve", type=float, default=10)
    p.set_defaults(func=cmd_new)
    p = sub.add_parser("start")
    p.add_argument("session")
    p.add_argument("--upload", action="append", default=[], help="local=remote")
    p.add_argument("--supervisor", default="/content/run/supervisor.py")
    p.add_argument("--run-dir", default="/content/run")
    p.add_argument("--guard-min", type=float, default=60)
    p.add_argument("--stall-min", type=float, default=45)
    p.add_argument("--poll", type=float, default=20)
    p.add_argument("--keep-min", type=float, default=40)
    p.add_argument("cmd", nargs="+")
    p.set_defaults(func=cmd_start)
    for name, func in (("watch", cmd_watch), ("status", cmd_status)):
        p = sub.add_parser(name)
        p.add_argument("session")
        p.add_argument("--run-dir", default="/content/run")
        p.add_argument("--every", type=float, default=300)
        p.add_argument("--fetch", action="append", default=[])
        p.add_argument("--local-dir", default=os.path.join(HERE, "results"))
        p.add_argument("--keep", action="store_true")
        p.set_defaults(func=func)
    p = sub.add_parser("stop")
    p.add_argument("session")
    p.set_defaults(func=cmd_stop)
    for name, func in (("push", cmd_push), ("pull", cmd_pull)):
        p = sub.add_parser(name)
        p.add_argument("session")
        p.add_argument("local" if name == "push" else "remote")
        p.add_argument("remote" if name == "push" else "local")
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
