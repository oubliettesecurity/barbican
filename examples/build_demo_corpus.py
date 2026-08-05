"""Build the bundled demo corpus.

Deterministic and dependency-free, so the committed corpus can be regenerated
and audited rather than taken on trust.

**Provenance.** Every sentence here was written for this fixture and is
dedicated to the public domain (CC0). No third-party corpus is used, quoted, or
redistributed. That is deliberate: the datasets that contain real influence
operations carry licences that restrict commercial use and redistribution, so
bundling one would put a licence encumbrance inside a shipped product.

**What this fixture is.** A demonstration of the detector's *behaviour*: a
coordinated push looks structurally different from organic conversation, and
BARBICAN separates them. Ground truth is exact because the campaigns are
constructed.

**What it is not.** Evidence of real-world detection performance. The posts are
invented, the topic is fictional, and no real person, organisation, or campaign
is depicted. Performance claims belong to evaluation against attributed
real-world IO data, which is a separate exercise and is not bundled.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

# A fictional municipal dispute. Chosen over any real political topic so the
# fixture cannot be mistaken for, or repurposed as, commentary on real events.
CAMPAIGN_A = [
    "The Rivermouth ferry closure is a disaster for working families on the east bank",
    "This Rivermouth ferry closure is a disaster for working families right across the east bank",
    "The ferry closure at Rivermouth is a disaster for working families, plain and simple",
    "Rivermouth ferry closure: a disaster for working families on the east bank, full stop",
    "The Rivermouth ferry closure is a disaster for the working families of the east bank",
]

CAMPAIGN_B = [
    "Nobody asked the harbour board before they raised the mooring levy again",
    "Nobody asked the harbour board before that mooring levy went up again",
    "The harbour board raised the mooring levy again and nobody was asked",
    "Again the harbour board raised the mooring levy, and nobody asked us",
]

# Unrelated people, unrelated subjects, spread across days.
ORGANIC = [
    "finally fixed the leaking radiator in the spare room, took the whole weekend",
    "anyone know a decent noodle place near the old station? craving something hot",
    "third straight loss at home. the defence needs rebuilding from scratch",
    "reading about deep sea vents tonight and the chemistry down there is wild",
    "my sourdough starter has survived two house moves and i am unreasonably proud",
    "the 6am train was cancelled again so i walked. surprisingly nice morning for it",
    "took the dog to the long beach at low tide, she has never been happier",
    "trying to learn bass guitar at 41. my neighbours are being very patient",
    "the library extension finally opened and the reading room is genuinely lovely",
    "grew actual tomatoes this year instead of just leaves. small victories",
    "watched a documentary about cartography and now i want to draw maps forever",
    "someone left a box of free books outside number 14 and i have no self control",
]

START = datetime(2026, 3, 2, 9, 0, 0)


def build() -> list[dict[str, object]]:
    posts: list[dict[str, object]] = []

    # Coordinated pushes: distinct personas, same line, minutes apart. The
    # temporal clustering is the point -- each post alone reads as ordinary.
    for campaign_id, texts, offset_min, persona_prefix in (
        ("ferry-push", CAMPAIGN_A, 0, "eastbank"),
        ("levy-push", CAMPAIGN_B, 220, "harbourwatch"),
    ):
        for i, text in enumerate(texts):
            ts = START + timedelta(minutes=offset_min + i * 3)
            posts.append(
                {
                    "post_id": f"{campaign_id}-{i + 1}",
                    "author_id": f"{persona_prefix}_{i + 1}",
                    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "text": text,
                    "campaign_id": campaign_id,
                    "label": "synthetic",
                }
            )

    # Organic controls, spread over days so neither narrative nor timing links
    # them. These are what a detector must NOT flag.
    for i, text in enumerate(ORGANIC):
        ts = START + timedelta(hours=i * 7 + 2, minutes=(i * 17) % 60)
        posts.append(
            {
                "post_id": f"organic-{i + 1}",
                "author_id": f"resident_{i + 1}",
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "text": text,
                "campaign_id": None,
                "label": "authentic",
            }
        )

    posts.sort(key=lambda p: str(p["timestamp"]))
    return posts


def main() -> int:
    out = Path(__file__).parent / "demo_corpus.jsonl"
    posts = build()
    out.write_text(
        "\n".join(json.dumps(p) for p in posts) + "\n",
        encoding="utf-8",
    )
    campaigns = {p["campaign_id"] for p in posts if p["campaign_id"]}
    print(f"Wrote {out}: {len(posts)} posts, {len(campaigns)} campaigns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
