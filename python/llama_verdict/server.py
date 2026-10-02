"""a jev-compatible http endpoint, stdlib only.

deliberately a thin adapter over Decider. see DESIGN.md: this is a stopgap
that the rust core is expected to replace rather than something to grow.

usage:
  python -m llama_verdict.server --base-url http://host:port \
      --model qwen3.5-0.8b:Q8_0 --formatter spec/formatters/qwen3.5-0.8b_Q8_0.json

one endpoint serves every model its backend has. a request's `model` names the
backend model that answers it, and a decider for that model is built the first
time one is asked for; `--model` is the one the jev aliases and a request that
names nothing mean.

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
import typing
import urllib.error
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config, derive, jev
from . import head as heads
from .backend import BACKEND_REFUSED, HttpBackend, weights_from_props
from .decide import Decider
from .types import Formatter

# jev clients treat these as retryable and back off; anything else they raise on
RETRYABLE = (429, 503, 529)
# a backend's answer for a model it does not have
BACKEND_MISSING = 404
DECISION_PATHS = ("/v1/systemone", "/systemone", "/v1/decisions")
MODELS_PATHS = ("/v1/models", "/models")
BANNER_PATH = "/v1/banner"
# the flags that say how the bound model is read, and mean nothing without one
BOUND_MODEL_FLAGS = ("--formatter", "--layout", "--head", "--head-rows", "--assistant-open")


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


class Served(typing.NamedTuple):
    """a backend model as this endpoint reads it, and the lines that say how."""
    decider: Decider
    banner: list


def describe(args, model, backend, formatter, head, source):
    """how one model is read, as lines: what a result measured through it has
    to be kept beside, since the served name says none of it."""
    checked = formatter.verification
    joint = bool(head) and head.joint
    health = (f"{'joint ' if joint else ''}head answered {checked['smoke_correct']} of its "
              f"verification" if head else f"mean option mass {checked['mean_option_mass']:.4f}")
    # a result is only comparable to another measured on the same file
    try:
        props = backend.props()
    except (OSError, http.client.HTTPException):
        # a backend that cannot say what it loaded still serves; the results
        # simply carry no weights
        props = {}
    # say what goes on the wire. a long-lived server silently running code
    # from before the prompt became text cost an hour of reading token arrays
    # in a proxy log and disbelieving the source on disk.
    wire = ("token ids, one /embedding per request" if joint else
            f"text, one {'/embedding' if head else '/completion'} per scoring pass")
    lines = [f"  model     {model}",
             f"  formatter {formatter.model} ({health}, {source})",
             f"  layout    {formatter.layout}",
             f"  weights   {json.dumps(weights_from_props(props))}",
             f"  build     {props.get('build_info')}",
             f"  debiasing order averaging {args.order_averaging}, prior correction "
             f"{'on' if args.prior_correction else 'off'}",
             f"  prompts   {wire} (loaded {time.strftime('%H:%M:%S')})"]
    if args.wide_alphabet:
        lines.append("  wide alphabet ON: option lists past 52 are labelled from the "
                     "model's vocabulary and read exactly, flagged extended_alphabet")
        wide = derive.wide_verification(formatter)
        lines.append(
            f"  wide alphabet verified: worst option mass {wide['min_option_mass']:.4f} at "
            f"{wide['option_counts']} labels, {wide['smoke_correct']} correct" if wide else
            "  wide alphabet not yet resolved; it is verified on the first request that "
            "needs more than 52 labels")
    elif args.tournament:
        lines.append("  tournament ON: option lists past the label ceiling are scored "
                     "in groups and their probabilities are approximate")
    return lines


def build(args, model):
    """a decider for one backend model.

    the flags that pin a formatter, a layout, a head or an opening describe the
    bound model and no other. a routed model is read as the model registry
    says: its live word, then verdict's copy of it, then the model's own chat
    template for one nobody recognises.
    """
    bound = model == args.model
    backend = HttpBackend(args.base_url, model)
    if bound and args.formatter:
        formatter, source = Formatter.load(args.formatter), "pinned"
    else:
        named = {"assistant_open": args.assistant_open, "layout": args.layout,
                 "head_path": args.head, "rows_path": args.head_rows} if bound else {}
        formatter, hit = derive.build(backend, model, **named)
        source = "cached" if hit else "derived from the model's chat template"
        if named.get("assistant_open"):
            source += ", opening overridden"
    head = heads.for_layout(formatter.layout, *((args.head, args.head_rows) if bound else ()))
    decider = Decider(backend, formatter, tournament=args.tournament,
                      wide_alphabet=args.wide_alphabet, order_averaging=args.order_averaging,
                      prior_correction=args.prior_correction, head=head)
    return Served(decider, describe(args, model, backend, formatter, head, source))


class Deciders:
    """the decider for each backend model a request has named, built once.

    building one derives and verifies a formatter, which can take a minute the
    first time a model is seen, so it is done under a lock of that model's own:
    two requests for a new model wait on one derivation, and a request for a
    model already built waits on nothing.
    """

    def __init__(self, args):
        self.args = args
        self._built, self._locks, self._guard = {}, {}, threading.Lock()

    def resolve(self, requested):
        """the backend model a request's `model` means: the bound one for an
        alias, for nothing at all, and for everything when routing is off."""
        if not self.args.routing or not requested or requested in ALIASES:
            return self.args.model
        return requested

    def get(self, model):
        if not model:
            raise ValueError(
                "this endpoint is bound to no model, so there is nothing for an alias or "
                "a request without one to mean. name a backend model in the request's "
                "`model`, or start the endpoint with --model")
        with self._guard:
            lock = self._locks.setdefault(model, threading.Lock())
        with lock:
            if model not in self._built:
                self._built[model] = build(self.args, model)
            return self._built[model]

    def served(self):
        """the bound model first, where there is one, then every other one
        built so far."""
        bound = [self.args.model] if self.args.model else []
        return [*bound, *(m for m in self._built if m != self.args.model)]


class Handler(BaseHTTPRequestHandler):
    deciders = None
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

    def _served(self, requested):
        """the decider for a request's model, or None having answered why not.
        a model that cannot be read is the caller's to fix and not an outage."""
        model = self.deciders.resolve(requested)
        try:
            return model, self.deciders.get(model)
        except ValueError as e:
            self._error(422, f"cannot serve {model}: {e}" if model else str(e))
        except urllib.error.HTTPError as e:
            if e.code == BACKEND_MISSING:
                self._error(422, f"cannot serve {model}: the backend has no such model")
            else:
                # a model the backend has and cannot start says why in its reply
                self._error(529, f"cannot serve {model}: the backend answered {e.code}: "
                                 f"{backend_message(e)}")
        except (OSError, http.client.HTTPException) as e:
            self._backend_unavailable(e)
        return model, None

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        path = url.path.rstrip("/")
        if path == "/health":
            return self._send(200, {"status": "ok"})
        if path in MODELS_PATHS:
            aliases = ALIASES if self.deciders.args.model else ()
            served = [{"id": name, "object": "model", "owned_by": "verdict"}
                      for name in (*self.deciders.served(), *aliases)]
            return self._send(200, {"object": "list", "data": served})
        if path == BANNER_PATH:
            if not self._authorised():
                return self._error(401, "missing or invalid api key")
            requested = urllib.parse.parse_qs(url.query).get("model", [None])[0]
            model, served = self._served(requested)
            return served and self._send(200, {"model": model,
                                               "banner": "\n".join(served.banner)})
        return self._error(404, f"no route for {self.path}")

    def do_POST(self):
        if self.path.rstrip("/") not in DECISION_PATHS:
            return self._error(404, f"no route for {self.path}")
        if not self._authorised():
            return self._error(401, "missing or invalid api key")

        length = int(self.headers.get("content-length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            state, questions = jev.validate_request(body)
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(422, str(e))

        model, served = self._served(body.get("model"))
        if not served:
            return None
        try:
            result = served.decider.decide(state, questions)
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

        wire = jev.result_to_wire(result, model)
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
            mass = "head" if a["option_mass"] is None else f"{a['option_mass']:.4f}"
            sys.stderr.write(
                f"  {name:16} -> {str(chosen):24} "
                f"conf={a['confidence']:.3f} mass={mass} "
                f"{'!' + ','.join(a['flags']) if a['flags'] else ''}\n")


def parse_args(argv=None):
    """flags first, then the environment, then the .env. the .env is what a
    fresh checkout fills in, so that bringing the endpoint up is `make serve`
    and not a command line long enough to mistype."""
    env = config.get
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=config.backend_url(),
                    help=f"llama-server or llama-swap base url "
                         f"[{', '.join(config.BACKEND_URLS)}]")
    ap.add_argument("--model", default=config.backend_model(),
                    help=f"the backend model the jev aliases, and a request that names "
                         f"none, are answered by. optional: without one every request "
                         f"has to name its model [{', '.join(config.BACKEND_MODELS)}]")
    ap.add_argument("--no-routing", dest="routing", action="store_false",
                    default=env("VERDICT_ROUTING") not in ("0", "false", "off", "no"),
                    help="answer every request with --model, whatever model it "
                         "names. by default a request's model names the backend "
                         "model that answers it [VERDICT_ROUTING=0]")
    ap.add_argument("--formatter", default=env("VERDICT_FORMATTER"),
                    help="a pinned formatter json for --model. omit it and the "
                         "formatter is derived from the model's own chat template "
                         "on first use, per SPEC 4.3 [VERDICT_FORMATTER]")
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
    ap.add_argument("--assistant-open", default=env("VERDICT_ASSISTANT_OPEN"),
                    help="spell out --model's assistant opening instead of deriving "
                         "it. for formats whose generation prompt ends before "
                         "content does; the ones verdict knows, gpt-oss among them, "
                         "need no flag [VERDICT_ASSISTANT_OPEN]")
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
                    help="read --model with one of spec/layouts, or `chat`, instead "
                         "of what the model registry says of it [VERDICT_LAYOUT]")
    ap.add_argument("--head", default=env("VERDICT_HEAD"),
                    help="a decision head on disk, for a layout read through one. "
                         "omit it and the head the layout pins is fetched once "
                         "and held to its checksum [VERDICT_HEAD]")
    ap.add_argument("--head-rows", default=env("VERDICT_HEAD_ROWS"),
                    help="a safetensors file on disk holding the output embedding "
                         "rows a joint head reads. omit it and the rows are read "
                         "by byte range from the weights the layout pins "
                         "[VERDICT_HEAD_ROWS]")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not args.base_url:
        ap.error("requires --base-url / LLAMA_VERDICT_URL. a .env at the repo root "
                 "fills it; see .env.example")
    if not args.model:
        # a bound model is only what the aliases mean, so routing does without
        if not args.routing:
            ap.error("--no-routing answers everything with one model, so it requires "
                     "--model / LLAMA_VERDICT_MODEL")
        about_it = [flag for flag in BOUND_MODEL_FLAGS
                    if getattr(args, flag.lstrip("-").replace("-", "_"))]
        if about_it:
            ap.error(f"{', '.join(about_it)} describe the bound model, so they require "
                     f"--model / LLAMA_VERDICT_MODEL")
    return args


def main(argv=None):
    args = parse_args(argv)
    deciders = Deciders(args)
    try:
        # the bound model is read now, so a model that cannot be served stops
        # `make serve` at the front door; every other one waits to be asked for
        described = (deciders.get(args.model).banner if args.model else
                     ["  model     none bound: a request names the backend model it wants",
                      f"  backend   {args.base_url}"])
    except ValueError as e:
        raise SystemExit(f"cannot serve {args.model}: {e}") from e
    except (OSError, http.client.HTTPException) as e:
        # a refused connection, a timeout or a garbled reply. the usual cause is
        # the backend being down or the model not loaded, and make serve is the
        # front door, so say that instead of handing over a traceback.
        # urlopen raises urllib errors and socket errors, both OSError
        raise SystemExit(f"cannot reach the backend at {args.base_url} "
                         f"(model {args.model}): {e}") from e
    Handler.deciders = deciders
    Handler.api_key = args.api_key
    Handler.verbose = not args.quiet

    lines = [
        f"verdict jev endpoint on http://{args.host}:{args.port}", *described,
        f"  aliases   {', '.join(ALIASES)}" if args.model else
        "  aliases   none, with no model for them to mean",
        f"  routing   a request's model names the backend model that answers it; "
        f"{BANNER_PATH}?model= says how it was read" if args.routing else
        f"  routing   OFF, every request is answered by {args.model}",
        # a server answering every caller because a key was meant to be set is
        # the failure nobody notices, and `make serve` reads the key from a .env
        "  auth      bearer key required" if args.api_key else "  auth      OPEN, no key",
        "  point a jev client at this with TYPESAFE_BASE_URL"]
    print("\n".join(lines), file=sys.stderr)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.current_thread().name = "verdict"
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
