"""Fail-fast integrity audit for the frozen confirmatory artifact."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    "v1_core": {"trials": 256, "blocks": 64, "posts": 20_352,
                "voter_calls": 18_432},
    "v1_size_extension": {"trials": 192, "blocks": 48, "posts": 15_264,
                          "voter_calls": 13_824},
}
CONDITIONS = {"control", "likes", "single_source", "distributed_sources"}
HEADLINES = {
    ("v1_core", "pdi_distributed_minus_single", "delta_alignment"):
        (0.05729166666666666, 0.11238688761311239),
    ("v1_size_extension", "pdi_distributed_minus_single", "delta_alignment"):
        (-0.03993055555555556, 0.33171666828333174),
    ("v1_core", "engagement_likes_minus_control",
     "opposite_side_survival"):
        (-0.0390625, 0.0068359375),
    ("v1_size_extension", "engagement_likes_minus_control",
     "opposite_side_survival"):
        (-0.01041666666666667, 0.5),
    ("v1_core", "engagement_likes_minus_control",
     "final_honest_pairwise_similarity"):
        (0.008235154749652246, 0.000104999895000105),
    ("v1_size_extension", "engagement_likes_minus_control",
     "final_honest_pairwise_similarity"):
        (0.01093770031668457, 9.99999000001e-07),
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def audit_run(name: str, expected: dict[str, int]) -> dict[str, object]:
    root = ROOT / "runs_confirmatory" / name
    rows = _rows(root / "results.csv")
    assert len(rows) == expected["trials"], (name, len(rows))
    assert Counter(row["condition"] for row in rows) == {
        condition: expected["trials"] // 4 for condition in CONDITIONS
    }
    blocks: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        blocks[(row["model"], row["topic_id"], row["seed"])].add(
            row["condition"]
        )
        assert row["exposure_audit_pass"] == "True"
        assert row["attacker_impressions"] == row["expected_attacker_impressions"]
    assert len(blocks) == expected["blocks"]
    assert all(conditions == CONDITIONS for conditions in blocks.values())
    assert sum(1 for _ in root.rglob("DONE")) == expected["trials"]
    assert sum(1 for _ in root.rglob("posts.jsonl")) == expected["trials"]
    assert sum(1 for _ in root.rglob("votes.json")) == expected["trials"]

    post_count = sum(
        sum(1 for _ in path.open()) for path in root.rglob("posts.jsonl")
    )
    voter_calls = 0
    for path in root.rglob("votes.json"):
        for round_record in json.loads(path.read_text()):
            voter_calls += len(round_record["voters"])
    assert post_count == expected["posts"], (name, post_count)
    assert voter_calls == expected["voter_calls"], (name, voter_calls)

    protocol_hash = hashlib.sha256(
        (ROOT / "CONFIRMATORY_PREREGISTRATION.md").read_bytes()
    ).hexdigest()
    assert (root / "preregistration.sha256").read_text().strip() == protocol_hash

    analysis = _rows(root / "confirmatory_analysis.csv")
    for (run, contrast, metric), (mean, p_value) in HEADLINES.items():
        if run != name:
            continue
        row = next(
            item for item in analysis
            if item["contrast"] == contrast
            and item["metric"] == metric
            and item["stratum_type"] == "overall"
        )
        assert abs(float(row["mean_difference"]) - mean) < 1e-12
        assert abs(float(row["randomization_p"]) - p_value) < 1e-12
    return {
        "trials": len(rows),
        "blocks": len(blocks),
        "posts": post_count,
        "voter_calls": voter_calls,
        "exposure_failures": 0,
        "protocol_sha256": protocol_hash,
    }


def main() -> None:
    report = {name: audit_run(name, expected)
              for name, expected in RUNS.items()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
