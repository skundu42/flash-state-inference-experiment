#!/usr/bin/env python3
"""A fake completions server, to test the client and report scripts without a GPU. Its numbers mean nothing.

    python3 tests/mock_server.py PORT official|flashstate

It keeps "checkpoints" per prompt prefix: in official mode at the last 1,024-token boundary of each prompt, in
flashstate mode at the exact end of prompt plus output. Time to first token grows with the uncached tokens.
"""
import hashlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE = sys.argv[2]
BLOCK = 1024
checkpoints, lengths, lock = set(), set(), threading.Lock()


def key(ids):
    return hashlib.sha1(json.dumps(ids).encode()).digest()


def cached_len(ids):
    with lock:
        for n in sorted((n for n in lengths if n < len(ids)), reverse=True):
            if key(ids[:n]) in checkpoints:
                return n
    return 0


def remember(ids):
    checkpoints.add(key(ids))
    lengths.add(len(ids))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ids, n_out = body["prompt"], body["max_tokens"]
        cached = cached_len(ids)
        time.sleep((len(ids) - cached) * 2e-6)
        seed = int.from_bytes(key(ids)[:4], "big")
        out = [1000 + (seed + i * 7919) % 90000 for i in range(n_out)]
        with lock:
            if MODE == "flashstate":
                remember(ids + out)
            else:
                remember(ids[:len(ids) // BLOCK * BLOCK])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for tok in out:
            chunk = {"choices": [{"index": 0, "text": f" t{tok}", "token_ids": [tok]}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        usage = {"prompt_tokens": len(ids), "completion_tokens": n_out,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\ndata: [DONE]\n\n".encode())


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
