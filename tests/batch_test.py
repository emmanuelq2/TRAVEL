"""
tests/batch_test.py
────────────────────
Batch-test all audio files in a directory against the live server.

Usage
─────
    # Test all files in audio_files/ against /v1/audio/query
    python tests/batch_test.py

    # Custom directory and endpoint
    python tests/batch_test.py --dir audio_files/english --endpoint transcribe

    # Adjust server URL and rate limit
    python tests/batch_test.py --url http://localhost:8001 --delay 13

Output
──────
    • Live progress printed to console
    • Full report saved to audio_files/batch_report_<timestamp>.csv
    • Summary table printed at the end

Rate limits
───────────
    Cohere trial  : 5 req/min  → use --delay 13  (13 s between requests)
    Deepgram      : no hard limit on batch
    Whisper       : ~3 req/min on free tier → use --delay 20
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

# ── CLI args ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Batch audio test runner")
parser.add_argument("--dir", default="audio_files", help="Directory with audio files")
parser.add_argument("--endpoint", default="query", help="'query' or 'transcribe'")
parser.add_argument("--url", default="http://localhost:8001", help="Server base URL")
parser.add_argument(
    "--delay",
    type=float,
    default=13.0,
    help="Seconds between requests (default 13 = 5/min)",
)
parser.add_argument(
    "--language", default=None, help="Language hint: 'en' or 'fr' (optional)"
)
parser.add_argument(
    "--ext",
    default="m4a,wav,mp3,ogg",
    help="Comma-separated audio extensions to include",
)
args = parser.parse_args()

# ── Setup ─────────────────────────────────────────────────────────────────────

audio_dir = Path(args.dir)
extensions = {f".{e.strip().lstrip('.')}" for e in args.ext.split(",")}
audio_files = sorted(
    f for f in audio_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions
)

if not audio_files:
    print(f"No audio files found in '{audio_dir}' with extensions {extensions}")
    exit(1)

endpoint_url = f"{args.url.rstrip('/')}/v1/audio/{args.endpoint}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = audio_dir / f"batch_report_{timestamp}.csv"

print(f"\nBatch test — {len(audio_files)} files")
print(f"Endpoint : {endpoint_url}")
print(f"Delay    : {args.delay}s between requests")
print(f"Report   : {report_path}\n")
print(f"{'#':<4} {'File':<35} {'Status':<8} {'Intent':<18} {'Transcript snippet'}")
print("─" * 100)

# ── Run ───────────────────────────────────────────────────────────────────────

results = []

with httpx.Client(timeout=60.0) as client:
    for i, audio_path in enumerate(audio_files, start=1):
        # Build form data
        form_data = {}
        if args.language:
            form_data["language"] = args.language

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, "audio/octet-stream")}
                resp = client.post(endpoint_url, files=files, data=form_data)

            status = resp.status_code

            if status == 200:
                body = resp.json()
                transcript = body.get("transcript", "")
                intent = body.get("intent", "—")
                missing = body.get("missing_slots", [])
                reply = body.get("reply", "")
                error = ""
            elif status == 422:
                body = resp.json()
                transcript = ""
                intent = "NO_SPEECH"
                missing = []
                reply = ""
                error = body.get("detail", "422")
            else:
                body = resp.text
                transcript = ""
                intent = "ERROR"
                missing = []
                reply = ""
                error = body[:120]

        except Exception as exc:
            status = 0
            transcript = ""
            intent = "EXCEPTION"
            missing = []
            reply = ""
            error = str(exc)[:120]

        snippet = (transcript[:55] + "…") if len(transcript) > 55 else transcript
        status_str = str(status) if status else "ERR"
        print(f"{i:<4} {audio_path.name:<35} {status_str:<8} {intent:<18} {snippet}")

        results.append(
            {
                "file": audio_path.name,
                "status": status,
                "intent": intent,
                "missing_slots": json.dumps(missing),
                "transcript": transcript,
                "reply": reply[:200],
                "error": error,
            }
        )

        # Rate-limit delay (skip after last file)
        if i < len(audio_files):
            time.sleep(args.delay)

# ── Save CSV ──────────────────────────────────────────────────────────────────

with open(report_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# ── Summary ───────────────────────────────────────────────────────────────────

total = len(results)
ok = sum(1 for r in results if r["status"] == 200)
no_speech = sum(1 for r in results if r["intent"] == "NO_SPEECH")
errors = sum(1 for r in results if r["intent"] in ("ERROR", "EXCEPTION"))

intent_counts: dict[str, int] = {}
for r in results:
    intent_counts[r["intent"]] = intent_counts.get(r["intent"], 0) + 1

print("\n" + "═" * 60)
print(f"  SUMMARY — {total} files")
print("═" * 60)
print(f"  {'Success (200)':<25} {ok:>4}  ({ok / total * 100:.0f}%)")
print(f"  {'No speech (422)':<25} {no_speech:>4}")
print(f"  {'Errors':<25} {errors:>4}")
print()
print("  Intents detected:")
for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
    print(f"    {intent:<22} {count:>4}")
print(f"\n  Full report saved to: {report_path}")
