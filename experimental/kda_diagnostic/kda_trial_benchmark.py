#!/usr/bin/env python3
"""Measure saved requests against one already-running GLM condition.

No service management, retry, cancellation, or request rewriting is performed.
A failed or busy engine ends the trial before another completion is attempted.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import signal
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def request_hash(payload):
    # Matches the original fallback runner's hash convention.
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@contextmanager
def request_deadline(seconds):
    """Bound total request time, including a response that trickles bytes."""
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(signum, frame):
        raise TimeoutError(f"Request exceeded its {seconds:g}-second deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        delay, interval = previous_timer
        if delay:
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, delay - (time.monotonic() - started)), interval)


def http_request(base_url, path, timeout, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url + path, data=data, headers={"Content-Type": "application/json"}
    )
    started = time.monotonic()
    result = {"http_status": None, "headers": None, "body": None}
    try:
        with request_deadline(timeout):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    result["http_status"] = response.status
                    result["headers"] = list(response.headers.items())
                    raw = response.read()
            except urllib.error.HTTPError as error:
                result["http_status"] = error.code
                result["headers"] = list(error.headers.items()) if error.headers else []
                try:
                    raw = error.read()
                finally:
                    error.close()
                result["http_error"] = str(error)
    except Exception as error:
        result["transport_error"] = {
            "type": type(error).__name__, "message": str(error), "repr": repr(error)
        }
        result["wall_seconds"] = time.monotonic() - started
        return result
    result["wall_seconds"] = time.monotonic() - started
    result["body_text"] = raw.decode("utf-8", errors="replace")
    try:
        result["body"] = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        result["json_error"] = str(error)
    if result.get("http_error") or result.get("json_error"):
        # Preserve every received byte even for a non-UTF-8 error body.
        result["body_base64"] = base64.b64encode(raw).decode("ascii")
    return result


def response_error(result):
    if result.get("transport_error"):
        return "transport_error"
    if result["http_status"] != 200:
        return "http_error"
    if result.get("json_error") or not isinstance(result["body"], dict):
        return "invalid_json_response"
    if "error" in result["body"]:
        return "api_error"
    return None


def health_error(health, models, cache_tokens=393216):
    # Owner-scoped diagnostic copy only. Preserve request bodies/hashes; the
    # private API accepts glm while reporting its canonical model in health.
    # Mixed model fixtures and unknown names must remain a pre-request refusal.
    canonical_model = None
    if len(models) == 1:
        name = next(iter(models))
        if name in ("glm", "GLM-5.3-Flash"):
            canonical_model = "GLM-5.3-Flash"
    expected = {
        "status": "ok",
        "engine": "ExLlamaV3",
        "cache_tokens": cache_tokens,
        # The API reserves 256 tokens in every tested cache profile.
        "context_length": cache_tokens - 256,
        "speculative_method": "dflash2",
        "draft_num_tokens": 7,
        "busy": False,
    }
    mismatches = {
        key: {"expected": value, "actual": health.get(key)}
        for key, value in expected.items()
        if health.get(key) != value
        or (key == "busy" and health.get(key) is not False)
    }
    if canonical_model is None or health.get("model") != canonical_model:
        mismatches["model"] = {"expected": canonical_model,
                               "fixture_models": sorted(models), "actual": health.get("model")}
    return mismatches


def metrics(body, elapsed):
    usage, timings = body["usage"], body["timings"]
    count, predicted_count = usage["completion_tokens"], timings["predicted_n"]
    decode_ms = timings["predicted_ms"]
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("completion_tokens must be a positive integer")
    if count != predicted_count:
        raise ValueError("completion_tokens and predicted_n disagree")
    if not isinstance(decode_ms, (int, float)) or not math.isfinite(decode_ms) or decode_ms <= 0:
        raise ValueError("predicted_ms must be finite and positive")
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("wall_seconds must be finite and positive")
    drafted, accepted = usage["draft_tokens"], usage["draft_tokens_accepted"]
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (drafted, accepted)):
        raise ValueError("draft-token counts must be nonnegative integers")
    if accepted > drafted or timings["draft_n"] != drafted or timings["draft_n_accepted"] != accepted:
        raise ValueError("draft-token counts disagree")
    return {
        "output_tok_s": count / (decode_ms / 1000),
        "end_to_end_tok_s": count / elapsed,
        "acceptance": accepted / drafted if drafted else None,
    }


def write_record(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def case_summaries(cases, records):
    result = {}
    for case in cases:
        rows = [row for row in records if row["case"] == case["id"] and row["success"]]
        item = {"n": len(rows)}
        if rows:
            rates = [row["output_tok_s"] for row in rows]
            acceptance = [row["acceptance"] for row in rows if row["acceptance"] is not None]
            item.update({
                "median_output_tok_s": statistics.median(rates),
                "min_output_tok_s": min(rates),
                "max_output_tok_s": max(rates),
                "median_end_to_end_tok_s": statistics.median(row["end_to_end_tok_s"] for row in rows),
                "median_acceptance": statistics.median(acceptance) if acceptance else None,
                "completion_tokens": [row["response"]["usage"]["completion_tokens"] for row in rows],
                "cached_tokens": [row["response"]["timings"].get("cached_tokens") for row in rows],
                "finish_reasons": [[choice.get("finish_reason") for choice in row["response"]["choices"]] for row in rows],
            })
        result[case["id"]] = item
    return result


def load_fixture(path):
    raw = path.read_bytes()
    fixture = json.loads(raw)
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture must have a nonempty cases list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError("each fixture case must have a nonempty string id")
        if case["id"] in seen:
            raise ValueError("fixture case ids must be unique")
        seen.add(case["id"])
        payload = case.get("request")
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
            raise ValueError("each fixture case must contain a request with a model")
        if payload.get("stream") is not False:
            raise ValueError("fixture requests must explicitly set stream=false")
    return fixture, cases, hashlib.sha256(raw).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8012")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--expected-cache-tokens", type=int, default=393216)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0, help="completion timeout, at most 180 seconds")
    args = parser.parse_args(argv)
    if args.expected_cache_tokens < 512 or args.expected_cache_tokens % 256:
        parser.error("--expected-cache-tokens must be at least 512 and a multiple of 256")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 180:
        parser.error("--timeout must be greater than 0 and at most 180")
    if not args.base_url.startswith(("http://", "https://")):
        parser.error("--base-url must use http:// or https://")
    if not args.condition.strip():
        parser.error("--condition must not be blank")
    paths = [p.resolve() for p in (args.fixture, args.output, args.summary)]
    if len(set(paths)) != len(paths):
        parser.error("fixture, output, and summary must be different files")
    args.base_url = args.base_url.rstrip("/")
    return args


def run_trial(args):
    fixture, cases, fixture_sha256 = load_fixture(args.fixture)
    models = {case["request"]["model"] for case in cases}
    records = []
    attempted = 0
    failure = None
    health_after = None
    started = utc_now()
    # Reserve both paths before contacting the service and never overwrite receipts.
    with args.output.open("x", encoding="utf-8") as raw:
        with args.summary.open("x", encoding="utf-8") as summary_file:
            active = None
            try:
                for run in range(1, args.repeats + 1):
                    for case in cases:
                        payload = case["request"]
                        active = {
                            "condition": args.condition, "case": case["id"], "run": run,
                            "started_utc": utc_now(), "request": payload,
                            "request_sha256": request_hash(payload), "fixture_sha256": fixture_sha256,
                        }
                        health_result = http_request(args.base_url, "/health", min(args.timeout, 10.0))
                        health = health_result["body"]
                        active["health_before"] = health
                        active["health_check"] = health_result
                        write_record(raw, {"event": "start", **active})
                        print(json.dumps({"event": "start", "condition": args.condition, "case": case["id"], "run": run}), flush=True)
                        reason = response_error(health_result)
                        mismatches = None if reason else health_error(health, models, args.expected_cache_tokens)
                        if reason or mismatches:
                            failure = {"phase": "health_before", "reason": reason or "health_guard_failed", "mismatches": mismatches}
                            record = {"event": "result", **active, "success": False, "http_status": None, "completion_attempted": False, "failure": failure, "finished_utc": utc_now()}
                            write_record(raw, record)
                            records.append(record)
                            break
                        attempted += 1
                        completion = http_request(args.base_url, "/v1/chat/completions", args.timeout, payload)
                        record = {
                            "event": "result", **active, "finished_utc": utc_now(),
                            "http_status": completion["http_status"], "wall_seconds": completion["wall_seconds"],
                            "completion_attempted": True, "completion_http": completion,
                            "response": completion["body"], "success": False,
                        }
                        reason = response_error(completion)
                        if reason:
                            failure = {"phase": "completion", "reason": reason}
                        else:
                            try:
                                record.update(metrics(completion["body"], completion["wall_seconds"]))
                                if completion["body"]["usage"]["completion_tokens"] > payload["max_tokens"]:
                                    raise ValueError("completion_tokens exceeds fixture max_tokens")
                                if not completion["body"]["choices"]:
                                    raise ValueError("completion response has no choices")
                                record["success"] = True
                            except (KeyError, TypeError, ValueError) as error:
                                failure = {"phase": "completion_metrics", "reason": "invalid_completion", "error": {"type": type(error).__name__, "message": str(error)}}
                        if failure:
                            record["failure"] = failure
                        write_record(raw, record)
                        records.append(record)
                        print(json.dumps({"event": "result", "condition": args.condition, "case": case["id"], "run": run, "success": record["success"], "http_status": record["http_status"], "output_tok_s": record.get("output_tok_s"), "failure": failure}), flush=True)
                        if failure:
                            break
                    if failure:
                        break
                if not failure:
                    health_after = http_request(args.base_url, "/health", min(args.timeout, 10.0))
                    reason = response_error(health_after)
                    mismatches = None if reason else health_error(health_after["body"], models, args.expected_cache_tokens)
                    if reason or mismatches:
                        failure = {"phase": "health_after", "reason": reason or "health_guard_failed", "mismatches": mismatches}
            except BaseException as error:
                failure = {"phase": "runner", "reason": "interrupted" if isinstance(error, KeyboardInterrupt) else "runner_error", "error": {"type": type(error).__name__, "message": str(error)}}
                if active:
                    write_record(raw, {"event": "result", **active, "success": False, "http_status": None, "failure": failure, "finished_utc": utc_now()})
            finally:
                summary = {
                    "condition": args.condition, "status": "failed" if failure else "passed",
                    "condition_failed": bool(failure), "failure": failure,
                    "started_utc": started, "finished_utc": utc_now(),
                    "fixture": str(args.fixture), "fixture_sha256": fixture_sha256,
                    "base_url": args.base_url, "repeats": args.repeats,
                    "expected_cache_tokens": args.expected_cache_tokens,
                    "expected_requests": len(cases) * args.repeats,
                    "attempted_requests": attempted,
                    "successful_requests": sum(record["success"] for record in records),
                    "decode_rate_definition": "completion_tokens / (timings.predicted_ms / 1000)",
                    "acceptance_definition": "draft_tokens_accepted / draft_tokens",
                    "comparison_to_historical_baselines": "invalid: complete historical request settings unknown",
                    "scope": "Unchanged fixture requests for one condition; throughput only, no semantic quality or long-context qualification.",
                    "cases": case_summaries(cases, records), "health_after": health_after,
                }
                summary_file.write(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                summary_file.flush()
                os.fsync(summary_file.fileno())
                print(json.dumps({"event": "summary", "condition": args.condition, "status": summary["status"], "successful_requests": summary["successful_requests"], "attempted_requests": attempted, "failure": failure}), flush=True)
    return 1 if failure else 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run_trial(args)
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"event": "setup_error", "condition": args.condition, "error": {"type": type(error).__name__, "message": str(error)}}), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
