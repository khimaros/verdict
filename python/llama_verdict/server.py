"""a jev-compatible http endpoint, stdlib only.

deliberately a thin adapter over Decider. see DESIGN.md: this is a stopgap
that the rust core is expected to replace rather than something to grow.

usage:
  python -m llama_verdict.server --base-url http://host:port \
      --model qwen3.5-0.8b:Q8_0 --formatter spec/formatters/qwen3.5-0.8b_Q8_0.json

every setting also reads its environment variable, and a .env beside the
working directory (or above it) fills whatever the environment leaves open,
so `make serve` needs no arguments once a .env is in place. flags win over
the environment and the environment wins over the .env; see config.py.
"""

import argparse
import http.client
import json
import sys
import threading
import time
import traceback
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config, derive, jev
from .backend import HttpBackend, weights_from_props
from .decide import Decider
from .types import Formatter

# jev clients treat these as retryable and back off; anything else they raise on
RETRYABLE = (429, 503, 529)
# llama-server's answer to a prompt it will not take as sent, such as one longer
# than its context
BACKEND_REFUSED = 400


def backend_message(error):
    """the reason a backend gave for an http error, or the status line."""
    try:
        body = json.loads(error.read() or b"{}")
    except (ValueError, OSError):
        return str(error)
    detail = body.get("error", body)
    return detail.get("message", str(error)) if isinstance(detail, dict) else str(detail)

# the aliases a jev client asks for by default. TYPESAFE_MODEL=jev-latest is the
# shipped default in browser-use/jev-ultrafast, so serving these is not cosmetic.
ALIASES = ("jev-latest", "jev-preview")


class Handler(BaseHTTPRequestHandler):
    decider = None
    model_name = "verdict"
    api_key = None
    verbose = False

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if self.verbose:
            sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, code, message):
        # a rejection the caller cannot read is a rejection they cannot fix
        sys.stderr.write(f"  !! {code} {message}\n")
        self._send(code, {"error": {"type": code, "message": message}})

    def _authorised(self):
        if not self.api_key:
            return True
        header = self.headers.get("authorization", "")
        return header == f"Bearer {self.api_key}"

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok"})
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            served = [{"id": name, "object": "model", "owned_by": "verdict"}
                      for name in (self.model_name, *ALIASES)]
            return self._send(200, {"object": "list", "data": served})
        return self._error(404, f"no route for {self.path}")

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/systemone", "/systemone", "/v1/decisions"):
            return self._error(404, f"no route for {self.path}")
        if not self._authorised():
            return self._error(401, "missing or invalid api key")

        length = int(self.headers.get("content-length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            state, questions = jev.validate_request(body)
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(422, str(e))

        try:
            result = self.decider.decide(state, questions)
        except ValueError as e:
            # a malformed question is the caller's problem, not an outage
            return self._error(422, str(e))
        except urllib.error.HTTPError as e:
            if e.code != BACKEND_REFUSED:
                return self._backend_unavailable(e)
            # the backend refused this prompt as sent, most often a state longer
            # than its context. sending it again cannot fit it, so the client is
            # told not to retry rather than handed the retryable 529
            return self._error(422, f"the backend cannot take this request: {backend_message(e)}")
        except Exception as e:
            return self._backend_unavailable(e)

        wire = jev.result_to_wire(result, self.model_name)
        if self.verbose:
            self._log_decision(wire, result)
        return self._send(200, wire)

    def _backend_unavailable(self, e):
        # the traceback names absolute source paths, and a quiet server's log is
        # kept as evidence beside published results; the one-line error is enough
        if self.verbose:
            traceback.print_exc()
        return self._error(529, f"backend unavailable: {type(e).__name__}: {e}")

    def _log_decision(self, wire, result):
        """option mass is the health check; print it next to every answer so a
        watched demo shows when the prompt stops steering the model."""
        for name, a in result["answers"].items():
            w = wire["answers"][name]
            chosen = w.get("choice", w.get("score", w.get("noul")))
            sys.stderr.write(
                f"  {name:16} -> {str(chosen):24} "
                f"conf={a['confidence']:.3f} mass={a['option_mass']:.4f} "
                f"{'!' + ','.join(a['flags']) if a['flags'] else ''}\n")


def parse_args(argv=None):
    """flags first, then the environment, then the .env. the .env is what a
    fresh checkout fills in, so that bringing the endpoint up is `make serve`
    and not a command line long enough to mistype."""
    env = config.get
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=env("LLAMA_VERDICT_URL"),
                    help="llama-server or llama-swap base url [LLAMA_VERDICT_URL]")
    ap.add_argument("--model", default=env("LLAMA_VERDICT_MODEL"),
                    help="model id behind that url [LLAMA_VERDICT_MODEL]")
    ap.add_argument("--formatter", default=env("VERDICT_FORMATTER"),
                    help="a pinned formatter json. omit it and the formatter is "
                         "derived from the model's own chat template on first "
                         "use, per SPEC 4.3 [VERDICT_FORMATTER]")
    ap.add_argument("--host", default=env("VERDICT_HOST") or "127.0.0.1",
                    help="[VERDICT_HOST]")
    ap.add_argument("--port", type=int, default=int(env("VERDICT_PORT") or 8477),
                    help="[VERDICT_PORT]")
    ap.add_argument("--api-key", default=env("VERDICT_API_KEY"),
                    help="required of callers as authorization: bearer. unset "
                         "leaves the endpoint open [VERDICT_API_KEY]")
    ap.add_argument("--tournament", action="store_true",
                    help="score option lists past the label ceiling in groups. "
                         "the probabilities become an approximation; off by default")
    ap.add_argument("--wide-alphabet", action="store_true",
                    help="label option lists past 52 from the model's own "
                         "vocabulary and score them in one exact pass. costs a "
                         "whole-vocabulary read the first time a long list "
                         "arrives, then nothing. takes precedence over "
                         "--tournament")
    ap.add_argument("--assistant-open", default=None,
                    help="spell out the assistant opening instead of deriving it. "
                         "for formats whose generation prompt ends before content "
                         "does: gpt-oss needs "
                         "'<|start|>assistant<|channel|>final<|message|>' "
                         "[VERDICT_ASSISTANT_OPEN]")
    ap.add_argument("--order-averaging", type=int,
                    default=int(env("VERDICT_ORDER_AVERAGING") or 1),
                    help="score every question under N option orders and average; "
                         "2 is forward and reversed. costs N passes "
                         "[VERDICT_ORDER_AVERAGING]")
    ap.add_argument("--prior-correction", action="store_true",
                    default=bool(env("VERDICT_PRIOR_CORRECTION")),
                    help="divide out what the model answers for an empty state. "
                         "costs one extra pass per distinct question "
                         "[VERDICT_PRIOR_CORRECTION]")
    ap.add_argument("--layout", default=env("VERDICT_LAYOUT"),
                    help="read the model with one of spec/layouts, or `chat`, "
                         "instead of what the model registry advertises on "
                         "llama-swap's /v1/models [VERDICT_LAYOUT]")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    wanted = (("--base-url / LLAMA_VERDICT_URL", args.base_url),
              ("--model / LLAMA_VERDICT_MODEL", args.model))
    missing = [name for name, value in wanted if not value]
    if missing:
        ap.error("requires " + " and ".join(missing) + ". a .env at the repo root "
                 "fills them; see .env.example")
    return args


def main(argv=None):
    args = parse_args(argv)

    backend = HttpBackend(args.base_url, args.model)
    try:
        if args.formatter:
            formatter, source = Formatter.load(args.formatter), "pinned"
        else:
            formatter, hit = derive.build(backend, args.model,
                                          assistant_open=args.assistant_open,
                                          layout=args.layout)
            source = "cached" if hit else "derived from the model's chat template"
            if args.assistant_open:
                source += ", opening overridden"
    except ValueError as e:
        raise SystemExit(f"cannot serve {args.model}: {e}") from e
    except (OSError, http.client.HTTPException) as e:
        # a refused connection, a timeout or a garbled reply. the usual cause is
        # the backend being down or the model not loaded, and make serve is the
        # front door, so say that instead of handing over a traceback.
        # urlopen raises urllib errors and socket errors, both OSError
        raise SystemExit(f"cannot reach the backend at {args.base_url} "
                         f"(model {args.model}): {e}") from e
    Handler.decider = Decider(backend, formatter, tournament=args.tournament,
                              wide_alphabet=args.wide_alphabet,
                              order_averaging=args.order_averaging,
                              prior_correction=args.prior_correction)
    Handler.model_name = args.model
    Handler.api_key = args.api_key
    Handler.verbose = not args.quiet

    mass = formatter.verification.get("mean_option_mass")
    print(f"verdict jev endpoint on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"  model     {args.model}", file=sys.stderr)
    print(f"  formatter {formatter.model} (mean option mass {mass:.4f}, {source})",
          file=sys.stderr)
    print(f"  layout    {formatter.layout}", file=sys.stderr)
    # a result is only comparable to another measured on the same file, and
    # the served name does not say which file that was
    try:
        props = backend.props()
    except (OSError, http.client.HTTPException):
        # a backend that cannot say what it loaded still serves; the results
        # simply carry no weights
        props = {}
    print(f"  weights   {json.dumps(weights_from_props(props))}", file=sys.stderr)
    print(f"  build     {props.get('build_info')}", file=sys.stderr)
    print(f"  debiasing order averaging {args.order_averaging}, prior correction "
          f"{'on' if args.prior_correction else 'off'}", file=sys.stderr)
    print(f"  aliases   {', '.join(ALIASES)}", file=sys.stderr)
    # a server answering every caller because a key was meant to be set is the
    # failure nobody notices, and `make serve` reads the key from a .env
    print("  auth      bearer key required" if args.api_key else "  auth      OPEN, no key",
          file=sys.stderr)
    # say what goes on the wire. a long-lived server silently running code
    # from before the prompt became text cost an hour of reading token arrays
    # in a proxy log and disbelieving the source on disk.
    print(f"  prompts   text, one /completion per scoring pass "
          f"(loaded {time.strftime('%H:%M:%S')})", file=sys.stderr)
    if args.wide_alphabet:
        print("  wide alphabet ON: option lists past 52 are labelled from the "
              "model's vocabulary and read exactly, flagged extended_alphabet",
              file=sys.stderr)
        checked = derive.wide_verification(formatter)
        if checked:
            print(f"  wide alphabet verified: worst option mass "
                  f"{checked['min_option_mass']:.4f} at "
                  f"{checked['option_counts']} labels, "
                  f"{checked['smoke_correct']} correct", file=sys.stderr)
        else:
            print("  wide alphabet not yet resolved; it is verified on the "
                  "first request that needs more than 52 labels",
                  file=sys.stderr)
    elif args.tournament:
        print("  tournament ON: option lists past the label ceiling are scored "
              "in groups and their probabilities are approximate", file=sys.stderr)
    print("  point a jev client at this with TYPESAFE_BASE_URL", file=sys.stderr)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.current_thread().name = "verdict"
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
