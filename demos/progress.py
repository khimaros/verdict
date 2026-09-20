"""carry progress across decisions, without a model.

jev-ultrafast is deliberately plannerless: the human's goal list is joined into
one string and passed unchanged to every decision, and the only memory between
steps is the last ten actions. those actions are labelled things like
"178 comments", which does not say which story, so an agent asked to visit
three stories cannot tell how many it has visited.

that is a missing-state problem, not a missing-model problem. the harness knows
exactly which pages it has been on, so it can simply say so.

this tracker is hacker news aware, which is honest for a demo and would need
generalising for anything else.
"""

import re

# a hacker news comments page. the story id is the identity that matters:
# the same story reached twice is not progress.
ITEM = re.compile(r"/item\?id=(\d+)")


class Progress:
    """what the agent has actually visited, accumulated across requests."""

    def __init__(self, target):
        self.target = target
        self.visited = []

    def observe(self, page):
        """record a story page the agent has landed on."""
        url = (page or {}).get("url", "") or ""
        match = ITEM.search(url)
        if not match:
            return
        story_id = match.group(1)
        if any(v["id"] == story_id for v in self.visited):
            return
        title = (page.get("title") or "").removesuffix(" | Hacker News").strip()
        self.visited.append({"id": story_id, "title": title, "url": url})

    def summary(self):
        done = len(self.visited)
        return {
            "stories_required": self.target,
            "stories_visited": done,
            "stories_remaining": max(0, self.target - done),
            "already_visited": [v["title"] for v in self.visited] or
                               ["none yet, this is the first"],
            "note": ("Every required story has been visited." if done >= self.target
                     else "Do not revisit a story listed above; choose one not listed."),
        }

    def inject(self, body):
        """observe the current page, then state the progress in the state.

        it goes in the state rather than the instructions because the state is
        the shared prefix: every question in the request sees it, and it is
        prefilled once.
        """
        state = body.get("state")
        if not isinstance(state, dict):
            return body, None
        self.observe(state.get("page"))
        state["progress"] = self.summary()
        return body, state["progress"]
