#!/usr/bin/env python3
"""Private local Codabench login, competition access checks and reviewed uploads.

Run login yourself in a terminal. Passwords are prompted without echo and never
saved. The API token is stored outside the repository with owner-only access.
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import uuid
import urllib.error
import urllib.parse
import urllib.request
import zipfile

BASE = "https://www.codabench.org/api/"
TOKEN_FILE = Path.home() / ".config" / "paiec" / "codabench-token"
PHASE_ID = 29785


class SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "https" or target.hostname != "www.codabench.org" or target.port not in (None, 443):
            raise RuntimeError("Refused an authentication redirect outside Codabench")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


class PublicLogRedirect(urllib.request.HTTPRedirectHandler):
    """Follow only HTTPS log links, without any account credentials."""
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "https" or not target.hostname or target.username or target.password:
            raise RuntimeError("Refused an unsafe scoring log redirect")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def api(path, *, token=None, payload=None, method=None):
    headers = {"Accept": "application/json", "User-Agent": "paiec-local-client/1"}
    if token:
        headers["Authorization"] = "Token " + token
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    request = urllib.request.Request(BASE + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.build_opener(SameOriginRedirect()).open(request, timeout=30) as response:
            content = response.read()
            return json.loads(content) if content else None
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Codabench returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        raise RuntimeError("Codabench connection failed") from None


def save_token(token):
    if not isinstance(token, str) or not token.strip():
        raise ValueError("Codabench returned no token")
    TOKEN_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if stat.S_IMODE(TOKEN_FILE.parent.stat().st_mode) & 0o077:
        raise RuntimeError("Token directory must have permissions 700")
    descriptor, temporary = tempfile.mkstemp(prefix=".codabench-token-", dir=TOKEN_FILE.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(token.strip() + "\n")
        os.replace(temporary, TOKEN_FILE)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_token():
    if not TOKEN_FILE.is_file():
        raise RuntimeError("Run the login command in your own terminal first")
    if stat.S_IMODE(TOKEN_FILE.stat().st_mode) & 0o077:
        raise RuntimeError("Token file must have permissions 600")
    return TOKEN_FILE.read_text().strip()


def submit(archive, token, receipt_dir=None):
    archive = Path(archive).resolve()
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        raise ValueError("Submission must be a ZIP file")
    content = archive.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    manifest_path = archive.with_suffix(".manifest.json")
    if not manifest_path.is_file() or json.loads(manifest_path.read_text()).get("sha256") != digest:
        raise ValueError("Submission must match its reviewed build manifest")
    receipt_dir = Path(receipt_dir) if receipt_dir is not None else Path(__file__).resolve().parents[1] / "private" / "submissions"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / (digest + ".json")
    if receipt_path.exists():
        raise ValueError("An upload receipt exists for this archive. Inspect it before another submission")
    competition = api("competitions/17828/", token=token)
    eligibility = api(f"can_make_submission/{PHASE_ID}/", token=token)
    if not eligibility.get("can"):
        raise ValueError("Codabench does not permit a submission at present")
    dataset = api("datasets/", token=token, payload={
        "name": "submission-" + uuid.uuid4().hex, "type": "submission", "competition": 17828,
        "is_public": False, "request_sassy_file_name": archive.name,
        "file_name": archive.name, "file_size": len(content),
    })
    receipt = {"archive": archive.name, "sha256": digest, "bytes": len(content),
               "dataset_key": dataset["key"], "upload_complete": False, "submission_id": None}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    upload_url = dataset["sassy_url"]
    parsed = urllib.parse.urlsplit(upload_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError("Codabench returned an invalid storage upload URL")
    # This separate request never carries the account token to object storage.
    request = urllib.request.Request(upload_url, data=content, method="PUT", headers={"Content-Type": "application/zip"})
    try:
        # Bundled frozen encoders can be hundreds of MB. Keep the storage
        # request separate from account authentication and allow its upload.
        with urllib.request.urlopen(request, timeout=1800) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError("Storage upload failed")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError):
        raise RuntimeError("Storage upload failed. Inspect the saved receipt before retrying") from None
    api(f"datasets/completed/{dataset['key']}/", token=token, method="PUT")
    receipt["upload_complete"] = True
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    result = api("submissions/", token=token, payload={
        "phase": PHASE_ID, "data": dataset["key"], "queue": (competition.get("queue") or {}).get("id"),
    })
    receipt["submission_id"] = result["id"]
    receipt["status"] = result.get("status")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"submission_id": result["id"], "status": result.get("status"), "archive_sha256": digest}))


def feedback(submission_id, token, output=None):
    """Fetch only the owner-visible Scoring output, never hidden result files."""
    details = api(f"submissions/{submission_id}/get_details/", token=token)
    logs = details.get("logs", [])
    matches = [log for log in logs if isinstance(log, dict) and log.get("name") == "scoring_stdout"]
    if len(matches) != 1 or not matches[0].get("data_file"):
        raise ValueError("Scoring output is not available yet")
    url = matches[0]["data_file"]
    if not isinstance(url, str):
        raise ValueError("Scoring output link is invalid")
    target = urllib.parse.urlsplit(url)
    if target.scheme != "https" or not target.hostname or target.username or target.password:
        raise ValueError("Scoring output link must use HTTPS")
    # The normal web UI fetches this same log link without API authentication.
    # Do not reuse the authenticated API opener, cookies or account headers.
    request = urllib.request.Request(url, headers={"User-Agent": "paiec-local-client/1"})
    limit = 8 * 1024 * 1024
    try:
        with urllib.request.build_opener(PublicLogRedirect()).open(request, timeout=60) as response:
            content = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Scoring output returned HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Scoring output connection failed ({type(exc.reason).__name__})") from None
    except (TimeoutError, ConnectionError):
        raise RuntimeError("Scoring output download timed out or disconnected") from None
    if len(content) > limit:
        raise ValueError("Scoring output exceeds the bounded download limit")
    output = Path(output) if output is not None else Path(__file__).resolve().parents[1] / "private/feedback" / f"{submission_id}_scoring_stdout.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    metadata = {"submission_id": submission_id, "kind": "owner_visible_scoring_stdout",
                "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                "output": str(output.resolve()), "hidden_results_requested": False}
    output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("login", "access", "list", "submit", "status", "feedback"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--submission-id", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "login":
            username = input("Codabench username ").strip()
            password = getpass.getpass("Codabench password ")
            result = api("api-token-auth/", payload={"username": username, "password": password})
            del password
            token = result.get("token")
            api("my_profile/", token=token)
            save_token(token)
            print("LOGIN_OK")
            return
        token = load_token()
        if args.action == "submit":
            if not args.archive:
                parser.error("submit requires --archive")
            submit(args.archive, token)
        elif args.action == "feedback":
            if not args.submission_id:
                parser.error("feedback requires --submission-id")
            feedback(args.submission_id, token, args.output)
        elif args.action == "status":
            if not args.submission_id:
                parser.error("status requires --submission-id")
            result = api(f"submissions/{args.submission_id}/", token=token)
            snapshot = {"submission_id": result.get("id"), "status": result.get("status"),
                        "scores": result.get("scores", [])}
            if args.output:
                previous = json.loads(args.output.read_text()) if args.output.exists() else {}
                if previous.get("submission_id", args.submission_id) != args.submission_id:
                    raise ValueError("Output file belongs to a different submission")
                previous.update(snapshot)
                previous["checked_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(previous, indent=2) + "\n")
            print(json.dumps(snapshot, ensure_ascii=False))
        elif args.action == "access":
            profile = api("my_profile/", token=token)
            eligibility = api(f"can_make_submission/{PHASE_ID}/", token=token)
            print(json.dumps({"username": profile.get("username"), "can_submit": eligibility.get("can"),
                              "reason": eligibility.get("reason")}, ensure_ascii=False))
        else:
            entries = api(f"submissions/?phase={PHASE_ID}&page_size=20", token=token)
            print(json.dumps([{"id": row.get("id"), "status": row.get("status")}
                              for row in entries.get("results", [])], ensure_ascii=False))
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
