#!/usr/bin/env python3
"""dump a page's text through the same snapshot the agent sees.

a `--collect` pattern is written against `document.body.innerText`, which is
neither the html nor what the page looks like: hacker news separates a byline
from its comment with a newline in the source and with nothing in innerText.
guessing at that cost a run that reported `witnessed 2/2` and collected zero.

usage: scripts/page_text.py https://news.ycombinator.com/item?id=1 > /tmp/hn.txt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "demos"))
from browser_agent import Chrome


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url")
    ap.add_argument("--cdp", default=os.environ.get("BU_CDP_URL",
                                                    "http://127.0.0.1:9222"))
    args = ap.parse_args()

    chrome = Chrome(args.cdp, args.url)
    try:
        chrome.navigate(args.url)
        print(chrome.snapshot()["text"])
    finally:
        chrome.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
