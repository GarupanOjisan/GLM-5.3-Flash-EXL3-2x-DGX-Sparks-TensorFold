#!/usr/bin/env python3
"""Copy-draft A/B on a running server: echo, edit, code and prose, at 1 and 4 streams.

Each prompt is built for that request (a nonce plus client.prose), so an n-gram cache from an
earlier run cannot replay the answer. To compare COPY on with COPY off, start the server twice
and run this once against each; do not reuse a prompt file. This process does not change the
server. It reports completion tok/s for the server's drafts and for the same prompts with
"draft": false, and checks the token sha256 of the two replies. The draft:false request is sent
after the drafted one, so its tok/s can include a kept prompt; the drafted tok/s is the fresh request.

MiaAI-Lab issue #208: a zig engine at PARALLEL=8 with dense q4 finished short benches faster than
this recipe's Python path at PARALLEL=2 with dense FP8, and finished question sets slower. Those
settings were not aligned, so the gap is not an engine-only or a copy-draft result. Pass
--parallel and --dense to record the server you started. Compare tok/s only when both match.

--recall also GETs /v1/recall. A 404 that says recall is off is reported and is not a copy-draft
failure. One extra echo then repeats a passage from earlier in this process, which a server
started with --recall can copy from its record. Without --recall every prompt stays unique.

--self-check builds prompts and the sha helper and does not open a connection.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
from client import API_URL, prose, open_url  # noqa: E402

WORKLOADS = ("echo", "edit", "code", "prose")
STREAMS = (1, 4)
_nonce_n = 0


def alignment_note(parallel: str | None, dense: str | None) -> str:
    recorded_p = parallel if parallel else "unset"
    recorded_d = dense if dense else "unset"
    return (
        "MiaAI-Lab #208: compare tok/s only when PARALLEL and DENSE match "
        "(that issue's zig run was PARALLEL=8 dense q4; the Python run was PARALLEL=2 dense FP8). "
        f"recorded PARALLEL={recorded_p} DENSE={recorded_d}"
    )


def nonce() -> str:
    global _nonce_n
    _nonce_n += 1
    return f"{time.time_ns():x}-{_nonce_n:x}"


def passage(tag: str) -> str:
    seed = zlib.crc32(tag.encode()) & 0x7FFFFFFF
    return prose(24, seed) + f" Marker {tag}."


def prompt_for(workload: str, tag: str) -> str:
    body = passage(tag)
    if workload == "echo":
        return "Repeat the next passage exactly, with the same words and the same marker.\n\n" + body
    if workload == "edit":
        return "Edit the passage: replace the word Marker with Edited. Change nothing else.\n\n" + body
    if workload == "code":
        name = "fn_" + tag.replace("-", "_")
        k = zlib.crc32(tag.encode()) % 10000
        return f"Write a Python function named {name} that returns the integer {k}. Reply with the function only."
    if workload == "prose":
        return f"Write two sentences about a bridge numbered {tag}. Do not quote any earlier answer."
    raise ValueError(workload)


def recall_echo(echo_prompt: str) -> str:
    """A later echo that repeats an earlier passage, for a server started with --recall."""
    return "Repeat the passage in the next block one more time, exactly, including its marker.\n\n" + echo_prompt


def plan() -> list[tuple[str, int, int, str]]:
    """(workload, streams, slot, prompt). Stream 1 and stream 4 do not share prompts."""
    rows = []
    for workload in WORKLOADS:
        for streams in STREAMS:
            for slot in range(streams):
                rows.append((workload, streams, slot, prompt_for(workload, nonce())))
    return rows


def token_sha(ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def post(path: str, body: dict, timeout: float, model: str) -> dict:
    req = urllib.request.Request(
        API_URL + path,
        json.dumps({"model": model, **body}).encode(),
        {"Content-Type": "application/json"},
    )
    with open_url(req, timeout=timeout) as response:
        return json.load(response)


def completion_ids(resp: dict) -> list[int] | None:
    choice = resp["choices"][0]
    tf = resp.get("tensorfold") or {}
    ids = tf.get("token_ids") or choice.get("token_ids")
    if ids:
        return [int(i) for i in ids]
    return None


def completion_text(resp: dict) -> str:
    message = resp["choices"][0].get("message") or {}
    return message.get("content") or ""


def completion_tokens(resp: dict, ids: list[int] | None) -> int:
    usage = resp.get("usage") or {}
    n = usage.get("completion_tokens")
    if isinstance(n, int) and n >= 0:
        return n
    if ids is not None:
        return len(ids)
    return 0


def tokenize(text: str, timeout: float, model: str) -> list[int] | None:
    try:
        req = urllib.request.Request(
            API_URL + "/tokenize",
            json.dumps({"model": model, "prompt": text}).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, TimeoutError):
        return None
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        return None
    return [int(i) for i in tokens]


def pair_sha(drafted: dict, serial: dict, timeout: float, model: str) -> tuple[str, str, str]:
    """Sha both replies the same way: generation ids, else /tokenize, else the reply text."""
    d_ids, s_ids = completion_ids(drafted), completion_ids(serial)
    if d_ids is not None and s_ids is not None:
        return token_sha(d_ids), token_sha(s_ids), "ids"
    d_text, s_text = completion_text(drafted), completion_text(serial)
    d_tok, s_tok = tokenize(d_text, timeout, model), tokenize(s_text, timeout, model)
    if d_tok is not None and s_tok is not None:
        return token_sha(d_tok), token_sha(s_tok), "tokenize"
    return text_sha(d_text), text_sha(s_text), "text-sha"


def chat(prompt: str, draft: bool, max_tokens: int, timeout: float, model: str) -> dict:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 0,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if not draft:
        body["draft"] = False
    return post("/v1/chat/completions", body, timeout, model)


def run_drafted(prompts: list[str], max_tokens: int, timeout: float, model: str) -> tuple[list[dict], float]:
    """Drafted replies and the wall from the first submit to the last completion."""

    def one(prompt: str) -> dict:
        return chat(prompt, True, max_tokens, timeout, model)

    started = time.perf_counter()
    if len(prompts) == 1:
        replies = [one(prompts[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            replies = list(pool.map(one, prompts))
    return replies, time.perf_counter() - started


def recall_state(timeout: float) -> str:
    """'on', 'off', or exit on any other answer. A 404 that names the flag is off, not a failure."""
    req = urllib.request.Request(API_URL + "/v1/recall")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read(500)
            code = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read(500)
        code = exc.code
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach the server at {API_URL} ({exc.reason}): is it running? (./start.sh)")
    if code == 200:
        return "on"
    if code == 404 and b"Recall is off" in body:
        return "off"
    sys.exit(f"GET /v1/recall answered {code}: {body.decode(errors='replace')[:300]}")


def report_group(workload: str, streams: int, drafted: list[dict], wall: float,
                 serial: list[tuple[str, str, float]], max_note: str) -> bool:
    """Print one row. Return True when every drafted sha matches draft:false."""
    tokens = 0
    ok = True
    parts = []
    for i, resp in enumerate(drafted):
        draft_sha, serial_sha, source = serial[i]
        match = draft_sha == serial_sha
        ok = ok and match
        tokens += completion_tokens(resp, completion_ids(resp))
        parts.append(f"slot{i} {source} match={'yes' if match else 'NO'} {draft_sha[:12]}")
    rate = tokens / wall if wall > 0 else 0.0
    print(f"{workload} streams={streams} tok/s={rate:.2f} tokens={tokens} wall_s={wall:.3f} {max_note} " + " ".join(parts))
    return ok


def self_check() -> None:
    first = plan()
    second = plan()
    texts_a = [row[3] for row in first]
    texts_b = [row[3] for row in second]
    assert len(texts_a) == len(WORKLOADS) * sum(STREAMS)
    assert len(set(texts_a)) == len(texts_a)
    assert set(texts_a).isdisjoint(texts_b)
    assert [row[1] for row in first].count(1) == len(WORKLOADS)
    assert [row[1] for row in first].count(4) == len(WORKLOADS) * 4
    assert {row[0] for row in first} == set(WORKLOADS)
    echo = next(row[3] for row in first if row[0] == "echo")
    again = recall_echo(echo)
    assert again != echo and echo in again
    assert token_sha([1, 2, 3]) == token_sha([1, 2, 3])
    assert token_sha([1, 2]) != token_sha([12])
    assert token_sha([1, 2]) != token_sha([1, 2, 0])
    note = alignment_note(None, "q4")
    assert "208" in note and "PARALLEL" in note and "DENSE" in note
    assert "recorded PARALLEL=unset DENSE=q4" in note
    print("copy_draft_bench self-check: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true", help="check prompts and the sha helper; no server")
    parser.add_argument("--recall", action="store_true", help="GET /v1/recall and repeat one earlier echo passage")
    parser.add_argument("--parallel", default=None, help="PARALLEL of the server you started (recorded, not applied)")
    parser.add_argument("--dense", default=None, help="DENSE of the server you started: q4, fp8 or bf16 (recorded, not applied)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.max_tokens < 1:
        sys.exit("--max-tokens is a token count")
    print(alignment_note(args.parallel, args.dense))
    print(f"server {API_URL}  workloads={','.join(WORKLOADS)}  streams={','.join(str(s) for s in STREAMS)}")
    rows = plan()
    ok = True
    echo_prompt = next(prompt for workload, _streams, _slot, prompt in rows if workload == "echo")
    for streams in STREAMS:
        for workload in WORKLOADS:
            prompts = [prompt for w, s, _slot, prompt in rows if w == workload and s == streams]
            drafted, wall = run_drafted(prompts, args.max_tokens, args.timeout, args.model)
            compared = []
            serial_tokens = 0
            serial_wall = 0.0
            for resp, prompt in zip(drafted, prompts):
                started = time.perf_counter()
                serial = chat(prompt, False, args.max_tokens, args.timeout, args.model)
                serial_wall += time.perf_counter() - started
                serial_tokens += completion_tokens(serial, completion_ids(serial))
                compared.append(pair_sha(resp, serial, args.timeout, args.model))
            serial_rate = serial_tokens / serial_wall if serial_wall > 0 else 0.0
            if not report_group(workload, streams, drafted, wall, compared, f"serial_tok/s={serial_rate:.2f}"):
                ok = False
    if args.recall:
        state = recall_state(args.timeout)
        print(f"recall: {state}")
        reused = recall_echo(echo_prompt)
        drafted, wall = run_drafted([reused], args.max_tokens, args.timeout, args.model)
        serial = chat(reused, False, args.max_tokens, args.timeout, args.model)
        pair = pair_sha(drafted[0], serial, args.timeout, args.model)
        if not report_group("recall-echo", 1, drafted, wall, [pair], "reused-passage"):
            ok = False
    if not ok:
        sys.exit("token sha differed from draft:false")


if __name__ == "__main__":
    main()
