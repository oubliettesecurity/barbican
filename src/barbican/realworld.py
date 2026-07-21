"""Real-world CIB dataset loader (public/defensive; the generalization check).

Normalizes a documented intermediate JSONL into the BARBICAN Post/Dataset schema.
Does NOT bundle the source dataset (CC BY-NC-ND). Absent dataset -> None -> the
harness marks the real-world split SKIPPED (never faked). Does NOT import spectre.
"""

import json
import logging
from pathlib import Path

from .types import Dataset, Post

log = logging.getLogger(__name__)

_LABEL_MAP = {"io": "synthetic", "control": "authentic"}


def load_realworld(path: str | None) -> Dataset | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None

    posts: list[Post] = []
    dropped = 0
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            label = _LABEL_MAP[rec["label"]]
            campaign_id = rec.get("campaign_id") if label == "synthetic" else None
            posts.append(
                Post(
                    post_id=f"RW-{i:06d}",
                    text=rec["text"],
                    author_id=rec["author_id"],
                    campaign_id=campaign_id,
                    timestamp=rec["timestamp"],
                    backend_model=None,
                    label=label,
                )
            )
        except (json.JSONDecodeError, KeyError) as exc:
            dropped += 1
            log.warning("realworld: dropped malformed record at line %d: %s", i, exc)
    if dropped:
        log.warning("realworld: dropped %d malformed record(s) total", dropped)
    return Dataset(posts=posts)
