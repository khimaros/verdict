#!/usr/bin/env python3
"""refresh spec/layouts from the model registry's layout table.

the model registry (aimbot) is canonical for how a model is read, bytes
included. verdict keeps its own committed copy in spec/layouts so that
rendering never depends on a server being up or on what it publishes, and this
is how the copy is refreshed: run it now and then, review the diff, regenerate
the fixtures.

the table is json, `{"version": 1, "layouts": {<name>: {kind?, source, note?,
affixes?, head?, constants}}, "models": {<registry key>: {short, repos,
readout}}}`. every block is checked against the closed knob list before
anything is written, so an import carrying a knob this verdict cannot render
is refused whole. a layout verdict has and the table lacks is reported and
left alone. `models` says which readout each recognised model needs and is
copied to spec/models.json; a table that carries none leaves that copy alone.

usage:
  scripts/import_layouts.py [SOURCE]          # a path or url; then make fixtures
  scripts/import_layouts.py --check [SOURCE]  # write nothing, fail if it would
  scripts/import_layouts.py --export          # the committed copy, in the table's shape
"""

import argparse
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from llama_verdict import config, spec

LAYOUTS = os.path.join(spec.SPEC_DIR, "layouts")
TABLE_VERSION = 1
# the registry's published table. a checkout's docs/layouts.json reads the
# same and needs no network: name it, or set VERDICT_LAYOUTS_SOURCE
REGISTRY_TABLE = "https://raw.githubusercontent.com/khimaros/aimbot/master/docs/layouts.json"
# a block's keys, in the order a layout file carries them
BLOCK_KEYS = ("kind", "source", "note", "affixes", "head", "constants")
# the copy of the registry's recognised models, beside the layouts directory
MODELS_FILE = "models.json"
# the readouts a recognised model can name; plain chat needs no entry
READOUT_KINDS = ("layout", "head")
FETCH_TIMEOUT = 60


def load(source):
    """the table, from a path or a url."""
    if "://" in source:
        with urllib.request.urlopen(source, timeout=FETCH_TIMEOUT) as r:
            return json.load(r)
    with open(source) as f:
        return json.load(f)


def table_from(layouts_dir):
    """the layout files of a directory, in the registry's shape."""
    blocks = {}
    for name in sorted(os.listdir(layouts_dir)):
        with open(os.path.join(layouts_dir, name)) as f:
            body = json.load(f)
        blocks[body["layout"]] = {k: body[k] for k in BLOCK_KEYS if k in body}
    table = {"version": TABLE_VERSION, "layouts": blocks}
    models = os.path.join(os.path.dirname(layouts_dir), MODELS_FILE)
    if os.path.exists(models):
        with open(models) as f:
            table["models"] = json.load(f)["models"]
    return table


def import_models(table, path, write=True):
    """bring the copy of the registry's recognised models in line with the
    table: which readout each needs, by the repositories its weights come from
    and by its short name. says `added`, `changed`, `unchanged`, or `absent`
    for a table that carries none, which leaves the copy alone."""
    models = table.get("models")
    if models is None:
        return "absent"
    for key, model in models.items():
        kind, _, name = str(model.get("readout", "")).partition(":")
        # a bare `head` is a model nothing here can apply, carried to be refused
        named = name or kind == "head"
        if kind not in READOUT_KINDS or not named or not isinstance(model.get("repos"), list):
            raise ValueError(f"model {key!r} needs a readout of {' or '.join(READOUT_KINDS)} "
                             f"with a name, or a bare head, and a list of repos")
    body, status = {"version": TABLE_VERSION, "models": models}, "added"
    if os.path.exists(path):
        with open(path) as f:
            status = "unchanged" if json.load(f) == body else "changed"
    if write and status != "unchanged":
        with open(path, "w") as f:
            f.write(json.dumps(body, indent=2) + "\n")
    return status


def as_file(name, block):
    """a table block as verdict's layout file."""
    return spec.check_layout({
        "spec_version": spec.constants()["spec_version"], "layout": name,
        **{k: block[k] for k in BLOCK_KEYS if k in block}})


def import_layouts(table, layouts_dir, write=True):
    """bring a directory of layout files in line with the table and say what
    moved. files are compared by content, so one that already says the same
    thing is left as it is, formatting included."""
    if table.get("version") != TABLE_VERSION:
        raise ValueError(f"layout table version {table.get('version')!r}; this verdict "
                         f"reads version {TABLE_VERSION}")
    files = {name: as_file(name, block) for name, block in table["layouts"].items()}
    os.makedirs(layouts_dir, exist_ok=True)
    report = {"added": [], "changed": [], "unchanged": [], "local_only": sorted(
        p.removesuffix(".json") for p in os.listdir(layouts_dir)
        if p.removesuffix(".json") not in files)}
    for name, body in sorted(files.items()):
        path = os.path.join(layouts_dir, f"{name}.json")
        status = "added"
        if os.path.exists(path):
            with open(path) as f:
                status = "unchanged" if json.load(f) == body else "changed"
        report[status].append(name)
        if write and status != "unchanged":
            with open(path, "w") as f:
                f.write(json.dumps(body, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?",
                    default=config.get("VERDICT_LAYOUTS_SOURCE") or REGISTRY_TABLE,
                    help="path or url of the registry's layout table as json; the "
                         "registry's published one unless named [VERDICT_LAYOUTS_SOURCE]")
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 if the table differs from spec/layouts")
    ap.add_argument("--export", action="store_true",
                    help="print spec/layouts in the table's shape and exit")
    args = ap.parse_args()
    if args.export:
        print(json.dumps(table_from(LAYOUTS), indent=1))
        return 0
    try:
        table = load(args.source)
        report = import_layouts(table, LAYOUTS, write=not args.check)
        models = import_models(table, os.path.join(spec.SPEC_DIR, MODELS_FILE),
                               write=not args.check)
    except (ValueError, OSError) as e:
        raise SystemExit(f"nothing imported from {args.source}: {e}") from e
    for status, names in report.items():
        print(f"  {status.replace('_', ' '):<10} {', '.join(names) or '-'}")
    print(f"  {'models':<10} {models}")
    moved = report["added"] + report["changed"] + [models] * (models in ("added", "changed"))
    if moved and not args.check:
        print("now run make fixtures and review the diff")
    return 1 if moved and args.check else 0


if __name__ == "__main__":
    sys.exit(main())
