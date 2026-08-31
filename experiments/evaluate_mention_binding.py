"""Audit v3 role binding against an existing graph, without model or QA inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medhorizon_videorag.graph_rag import (
    build_evidence_graph,
    load_description_rows,
    load_evidence_graph,
    write_evidence_graph_artifacts,
)
from medhorizon_videorag.graph_rag.mention_binding import surface_key


def audit_bindings(
    baseline: dict[str, Any], graph: dict[str, Any]
) -> tuple[dict, list]:
    """Count role edges, not physical identity errors or clinical accuracy."""

    def index(g):
        nodes = {n["id"]: n for n in g["nodes"]}
        roles = {
            (e["source"], e["relation"]): e["target"]
            for e in g["edges"]
            if e["relation"] in {"has_subject", "acts_on"}
        }
        clips = {
            n["metadata"]["clip_id"]: n
            for n in nodes.values()
            if n["node_type"] == "segment"
        }
        mentions = defaultdict(list)
        for n in nodes.values():
            if n["node_type"] == "entity_mention":
                mentions[n["metadata"]["clip_id"]].append(n)
        return nodes, roles, clips, mentions

    old, old_roles, old_clips, old_mentions = index(baseline)
    new, roles, clips, mentions = index(graph)
    assert old_clips.keys() == clips.keys(), "Clip set changed"
    assert all(
        old_clips[c]["metadata"]["observation"] == clips[c]["metadata"]["observation"]
        for c in clips
    ), "Observations changed"
    counts = Counter()
    details = []
    statuses = Counter()
    methods = Counter()
    for aid, node in old.items():
        if node["node_type"] != "action_event":
            continue
        assert aid in new and new[aid]["label"] == node["label"], (
            "Atomic actions changed"
        )
        cid = node["metadata"]["clip_id"]
        raw = old_clips[cid]["metadata"]["observation"]["observed_facts"]["actions"][
            int(aid.split(":")[-2])
        ]
        for role, relation in [("subject", "has_subject"), ("target", "acts_on")]:
            counts["role_edges"] += 1
            previous = old[old_roles[aid, relation]]
            linked = new[roles[aid, relation]]
            surface = raw.get(role, "unidentified object")
            exact = lambda text, surface=surface: (
                text.lower().strip() == surface.lower().strip()
            )
            old_alternatives = [
                m
                for m in old_mentions[cid]
                if m["metadata"]["canonical"] == previous["metadata"]["canonical"]
                and m["metadata"]["category"] == previous["metadata"]["category"]
                and exact(m["label"])
            ]
            old_bad = not exact(previous["label"]) and bool(old_alternatives)
            new_bad = not exact(linked["label"]) and any(
                exact(m["label"]) for m in mentions[cid]
            )
            binding = new[aid]["metadata"][f"{role}_binding"]
            statuses[binding["status"]] += 1
            methods[binding["method"]] += 1
            visible = [
                m
                for m in mentions[cid]
                if m["metadata"]["source_field"].startswith("visible_")
            ]
            exact_visible = [
                m for m in visible if surface_key(m["label"]) == surface_key(surface)
            ]
            unique_exact = len(exact_visible) == 1
            counts["unique_exact_visible_roles"] += unique_exact
            counts["unique_exact_not_resolved"] += unique_exact and (
                binding["status"] != "resolved"
                or linked["id"] != exact_visible[0]["id"]
            )
            counts["baseline_exact_alternative_mislinks"] += old_bad
            counts["v3_exact_alternative_mislinks"] += new_bad
            counts["baseline_mislinks_now_resolved_exact"] += (
                old_bad
                and binding["method"] == "exact_surface"
                and exact(linked["label"])
            )
            counts["baseline_mislinks_now_independent"] += (
                old_bad and binding["status"] != "resolved"
            )
            if binding["status"] != "resolved":
                counts["independent_argument_policy_violations"] += not (
                    linked["metadata"]["source_field"] == f"action_{role}"
                    and linked["label"] == surface
                    and binding["selected_mention_id"] is None
                    and linked["id"] not in binding["candidate_mention_ids"]
                )
            else:
                counts["resolved_policy_violations"] += not (
                    binding["compatible_mention_ids"] == [linked["id"]]
                    and binding["selected_mention_id"] == linked["id"]
                    and linked["metadata"]["source_field"].startswith("visible_")
                )
            details.append(
                {
                    "action_id": aid,
                    "clip_id": cid,
                    "role": role,
                    "raw_surface": surface,
                    "baseline_label": previous["label"],
                    "v3_label": linked["label"],
                    "baseline_exact_alternative_mislink": old_bad,
                    "v3_exact_alternative_mislink": new_bad,
                    "binding": binding,
                }
            )
    assert len(roles) == len(old_roles), "Action role edge set changed"
    passed = all(
        counts[k] == 0
        for k in [
            "v3_exact_alternative_mislinks",
            "unique_exact_not_resolved",
            "independent_argument_policy_violations",
            "resolved_policy_violations",
        ]
    )
    return {
        "video_id": graph["video_id"],
        "builder_version": graph["metadata"]["builder_version"],
        "counts": dict(counts),
        "binding_status_counts": dict(statuses),
        "binding_method_counts": dict(methods),
        "baseline_temporal_events": sum(
            n["node_type"] == "temporal_event" for n in old.values()
        ),
        "v3_temporal_events": sum(
            n["node_type"] == "temporal_event" for n in new.values()
        ),
        "baseline_mentions": sum(
            n["node_type"] == "entity_mention" for n in old.values()
        ),
        "v3_mentions": sum(n["node_type"] == "entity_mention" for n in new.values()),
        "passed": passed,
        "scope": "Textual reference integrity, not clinical or physical-ID accuracy",
    }, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-graph", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--graph", help="Audit an already built v3 graph")
    source.add_argument(
        "--descriptions",
        help="Build v3 using the baseline frame references, then audit",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    baseline = load_evidence_graph(args.baseline_graph).to_dict()
    artifacts = None
    if args.descriptions:
        frames = {
            n["metadata"]["clip_id"]: [
                p for e in n["evidence"] for p in e["frame_paths"]
            ]
            for n in baseline["nodes"]
            if n["node_type"] == "segment"
        }
        artifacts = build_evidence_graph(
            load_description_rows(args.descriptions),
            frame_paths_by_clip=frames,
            merge_threshold=baseline["metadata"]["merge_threshold"],
            max_merged_clips=baseline["metadata"]["max_merged_clips"],
            max_representative_clips=baseline["metadata"]["max_representative_clips"],
        )
        graph = artifacts.graph.to_dict()
    else:
        graph = load_evidence_graph(args.graph).to_dict()
    report, details = audit_bindings(baseline, graph)
    report["baseline_sha256"] = hashlib.sha256(
        Path(args.baseline_graph).read_bytes()
    ).hexdigest()
    if args.descriptions:
        report["descriptions_sha256"] = hashlib.sha256(
            Path(args.descriptions).read_bytes()
        ).hexdigest()
    output = Path(args.output_dir)
    if artifacts is not None:
        write_evidence_graph_artifacts(artifacts, output)
        output = output / "mention_binding_audit"
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "details.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in details)
    )
    print(json.dumps(report, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit("Mention binding acceptance checks failed")


if __name__ == "__main__":
    main()
