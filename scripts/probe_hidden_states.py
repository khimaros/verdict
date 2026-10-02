#!/usr/bin/env python3
"""what does it cost to get every position's hidden state out of llama-server?

a joint head such as clef's reads the final hidden state of every prompt
position, and a llama-server pooling `none` returns them as json: thousands of
floats per token. over a network that reply is hundreds of megabytes a record,
so this measures where the time goes, at several prompt lengths, to say whether
such a model can be read over a link or only beside the server.

self-contained and stdlib only, so it can be copied to the serving host and
run there against llama-swap on loopback, which removes the network and leaves
the server's encoding and this side's decoding.

usage:
  scripts/probe_hidden_states.py --base-url http://localhost:7860 \
      --model clef-flash:Q8_0 --tokens 256 1024 4096
"""

import argparse
import http.client
import json
import time
import urllib.parse

SENTENCE = "The invoice was charged twice for the same month and a refund is wanted. "
TIMEOUT = 1800
MB = 1024 * 1024


def post(conn, path, payload):
    """one request: the parsed reply, its size, and where the time went."""
    body = json.dumps(payload).encode()
    t0 = time.monotonic()
    conn.request("POST", path, body=body, headers={"content-type": "application/json"})
    response = conn.getresponse()
    t_headers = time.monotonic()
    raw = response.read()
    t_read = time.monotonic()
    if response.status >= 400:
        raise SystemExit(f"{path} answered {response.status}: {raw[:300]!r}")
    parsed = json.loads(raw)
    t_parsed = time.monotonic()
    return parsed, {"request_bytes": len(body), "reply_bytes": len(raw),
                    "until_headers_s": t_headers - t0, "reading_s": t_read - t_headers,
                    "json_decode_s": t_parsed - t_read, "total_s": t_parsed - t0}


def prompt_of(conn, base, count):
    """`count` token ids of ordinary prose."""
    text = SENTENCE * (count // 8 + 1)
    tokens, _ = post(conn, base + "/tokenize", {"content": text})
    return tokens["tokens"][:count]


def measure(conn, base, count):
    tokens = prompt_of(conn, base, count)
    reply, cost = post(conn, base + "/embedding", {"content": tokens, "embd_normalize": -1})
    rows = reply[0]["embedding"]
    floats = sum(len(row) for row in rows)
    return {"prompt_tokens": len(tokens), "rows": len(rows), "width": len(rows[0]),
            "floats": floats, "bytes_per_float": round(cost["reply_bytes"] / floats, 2),
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in cost.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="llama-swap or llama-server root")
    ap.add_argument("--model", help="model id behind llama-swap; omit for a bare llama-server")
    ap.add_argument("--tokens", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--out", help="write the measurements here as json")
    args = ap.parse_args()

    url = urllib.parse.urlsplit(args.base_url.rstrip("/").removesuffix("/v1"))
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    conn = cls(url.netloc, timeout=TIMEOUT)
    base = url.path + (f"/upstream/{args.model}" if args.model else "")

    results = []
    print(f"{'tokens':>7} {'rows':>6} {'width':>6} {'reply':>9} {'server':>8} {'read':>8} "
          f"{'decode':>8} {'total':>8}")
    for count in args.tokens:
        # the first request at a length also loads the model and fills no cache
        # worth keeping, so each length is measured twice and the second kept
        measure(conn, base, count)
        r = measure(conn, base, count)
        results.append(r)
        print(f"{r['prompt_tokens']:>7} {r['rows']:>6} {r['width']:>6} "
              f"{r['reply_bytes'] / MB:>7.1f}mb {r['until_headers_s']:>7.2f}s "
              f"{r['reading_s']:>7.2f}s {r['json_decode_s']:>7.2f}s {r['total_s']:>7.2f}s",
              flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
