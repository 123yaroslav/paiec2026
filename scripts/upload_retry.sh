#!/bin/sh
# Upload an archive to Codabench with retries, then add extra runs by dataset key.
# usage: upload_retry.sh ARCHIVE_PATH EXTRA_RUNS LABEL
# An incomplete receipt from a failed attempt is kept under a new name, never deleted.
set -u
ARCHIVE="$1"; EXTRA="$2"; LABEL="$3"
ROOT="${PAIEC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT" || exit 1
SHA=$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)
REC="private/submissions/$SHA.json"
for attempt in 1 2 3 4 5 6; do
  OUT=$(/usr/bin/python3 scripts/codabench_client.py submit --archive "$ARCHIVE" 2>&1 | tail -1)
  echo "attempt $attempt: $OUT"
  if echo "$OUT" | grep -q submission_id; then
    (cd scripts && /usr/bin/python3 resubmit.py "$(basename "$ARCHIVE")" "$EXTRA" --label "$LABEL" 2>&1 | grep -o '"submission_id": [0-9]*\|HTTP [0-9]*\|expected .*')
    exit 0
  fi
  if [ -f "$REC" ]; then
    mv "$REC" "${REC%.json}.failed_upload_$(date +%H%M%S)"
  fi
  perl -e 'select(undef,undef,undef,90)'
done
exit 1
