from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .phase_projection import append_sequence_phases, intersect_evidence
from .schemas import EvidenceInterval, GraphEdge, GraphNode, VideoEvidenceGraph

SEMANTIC_LAYER_VERSION = "phase-boundary-instrument-track-v1"
SEMANTIC_GRAPH_SCHEMA_VERSION = "medical-video-evidence-graph-v3-pilot"
APPEARANCE_TRACK_VERSION = "appearance-instrument-track-v1"

_CONFIDENCE = {"high": 0.9, "medium": 0.65, "low": 0.35}
_UNKNOWN_LABELS = {"", "none", "null", "unknown", "uncertain", "not visible"}


@dataclass(frozen=True)
class SemanticLayerArtifacts:
    graph: VideoEvidenceGraph
    hypotheses: list[dict[str, Any]]
    report: dict[str, Any]


def load_semantic_hypotheses(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_ids = [str(row.get("event_id", "")) for row in rows]
    if not rows:
        raise ValueError("Semantic hypothesis JSONL is empty")
    if any(not event_id for event_id in event_ids):
        raise ValueError("Every semantic hypothesis row requires event_id")
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Semantic hypothesis JSONL contains duplicate event IDs")
    return rows


def augment_with_semantic_hypotheses(
    graph: VideoEvidenceGraph,
    hypotheses: Sequence[dict[str, Any]],
    *,
    max_instrument_gap_events: int = 1,
    instrument_track_source: str = "semantic_hypotheses",
) -> SemanticLayerArtifacts:
    """Add explicitly derived medical hypotheses without changing observation facts."""
    if max_instrument_gap_events < 0:
        raise ValueError("max_instrument_gap_events must be non-negative")
    if instrument_track_source not in {
        "semantic_hypotheses",
        "appearance_mentions",
    }:
        raise ValueError(
            "instrument_track_source must be semantic_hypotheses or appearance_mentions"
        )
    if any(
        node.node_type in {"phase_hypothesis", "phase_boundary", "instrument_track"}
        for node in graph.nodes
    ):
        raise ValueError("Graph already contains the semantic hypothesis layer")

    events = sorted(
        (node for node in graph.nodes if node.node_type == "temporal_event"),
        key=_node_start,
    )
    event_by_id = {node.id: node for node in events}
    event_position = {node.id: index for index, node in enumerate(events)}
    row_by_event: dict[str, dict[str, Any]] = {}
    for row in hypotheses:
        event_id = str(row.get("event_id", ""))
        if event_id not in event_by_id:
            raise ValueError(
                f"Semantic hypothesis references unknown event: {event_id}"
            )
        if event_id in row_by_event:
            raise ValueError(f"Duplicate semantic hypothesis event: {event_id}")
        row_by_event[event_id] = dict(row)

    nodes = list(graph.nodes)
    edges = list(graph.edges)
    phase_event_membership: dict[str, dict[str, list[EvidenceInterval]]] = {}
    interval_projection = any("phase_projection_version" in row for row in hypotheses)
    if interval_projection:
        phase_event_membership, phase_count = append_sequence_phases(
            graph, hypotheses, nodes, edges
        )
    else:
        phase_runs = _phase_runs(events, row_by_event)
        for phase_index, run in enumerate(phase_runs):
            phase_id = f"phase_hypothesis:{graph.video_id}:{phase_index:05d}"
            event_ids = [item[0].id for item in run]
            label = run[0][1]
            confidence = _mean(item[2] for item in run)
            evidence = _dedupe_intervals(
                interval for event, _, _, _ in run for interval in event.evidence
            )
            nodes.append(
                GraphNode(
                    phase_id,
                    graph.video_id,
                    "phase_hypothesis",
                    label,
                    evidence,
                    confidence=confidence,
                    metadata={
                        "canonical_label": _canonical_label(label),
                        "supporting_event_ids": event_ids,
                        "bases": [item[3] for item in run if item[3]],
                        "hypothesis_source": _sources(run, row_by_event),
                        "derived": True,
                        "fact_status": "medical_hypothesis",
                        "semantic_layer_version": SEMANTIC_LAYER_VERSION,
                    },
                )
            )
            for event, _, event_confidence, basis in run:
                phase_event_membership[event.id] = {phase_id: list(event.evidence)}
                edges.append(
                    GraphEdge(
                        phase_id,
                        event.id,
                        "derived_from",
                        event.evidence,
                        event_confidence,
                        {"basis": basis},
                    )
                )
            for kind, event in (("onset", run[0][0]), ("offset", run[-1][0])):
                boundary_id = (
                    f"phase_boundary:{graph.video_id}:{phase_index:05d}:{kind}"
                )
                timestamp = _node_start(event) if kind == "onset" else _node_end(event)
                nodes.append(
                    GraphNode(
                        boundary_id,
                        graph.video_id,
                        "phase_boundary",
                        f"{kind} of {label}",
                        list(event.evidence),
                        confidence=confidence,
                        metadata={
                            "boundary_kind": kind,
                            "timestamp_seconds": timestamp,
                            "phase_hypothesis_id": phase_id,
                            "grounding_event_id": event.id,
                            "derived": True,
                            "fact_status": "medical_hypothesis",
                        },
                    )
                )
                edges.extend(
                    [
                        GraphEdge(
                            phase_id,
                            boundary_id,
                            "has_boundary",
                            list(event.evidence),
                            confidence,
                            {"boundary_kind": kind},
                        ),
                        GraphEdge(
                            boundary_id,
                            event.id,
                            "grounded_by",
                            list(event.evidence),
                            confidence,
                        ),
                    ]
                )
        phase_count = len(phase_runs)

    if instrument_track_source == "appearance_mentions":
        track_count = _append_appearance_instrument_tracks(
            graph,
            events,
            event_position,
            phase_event_membership,
            nodes,
            edges,
            max_gap=max_instrument_gap_events,
        )
    else:
        track_count = _append_semantic_instrument_tracks(
            graph,
            events,
            event_position,
            row_by_event,
            phase_event_membership,
            nodes,
            edges,
            max_gap=max_instrument_gap_events,
        )

    metadata = dict(graph.metadata)
    metadata.update(
        {
            "base_schema_version": graph.schema_version,
            "semantic_layer_version": SEMANTIC_LAYER_VERSION,
            "semantic_hypotheses_used": True,
            "sequence_interval_projection": interval_projection,
            "semantic_nodes_are_observed_facts": False,
            "instrument_track_source": instrument_track_source,
        }
    )
    semantic_graph = VideoEvidenceGraph(
        graph.video_id,
        nodes,
        edges,
        SEMANTIC_GRAPH_SCHEMA_VERSION,
        metadata,
    )
    node_counts = Counter(node.node_type for node in semantic_graph.nodes)
    edge_counts = Counter(edge.relation for edge in semantic_graph.edges)
    report = {
        "video_id": graph.video_id,
        "schema_version": semantic_graph.schema_version,
        "semantic_layer_version": SEMANTIC_LAYER_VERSION,
        "source_hypothesis_count": len(hypotheses),
        "covered_event_count": len(row_by_event),
        "phase_hypothesis_count": phase_count,
        "phase_boundary_count": 2 * phase_count,
        "instrument_track_count": track_count,
        "instrument_track_source": instrument_track_source,
        "node_type_counts": dict(sorted(node_counts.items())),
        "edge_type_counts": dict(sorted(edge_counts.items())),
        "semantic_nodes_are_observed_facts": False,
    }
    return SemanticLayerArtifacts(semantic_graph, list(hypotheses), report)


def retrieve_phase_boundary_instruments(
    graph: VideoEvidenceGraph,
    phase_label: str,
    *,
    boundary_kind: str = "onset",
    context_events: int = 1,
) -> dict[str, Any]:
    if boundary_kind not in {"onset", "offset"}:
        raise ValueError("boundary_kind must be onset or offset")
    if context_events < 0:
        raise ValueError("context_events must be non-negative")
    phases = [node for node in graph.nodes if node.node_type == "phase_hypothesis"]
    if not phases:
        raise ValueError("Graph contains no phase_hypothesis nodes")
    query = _canonical_label(phase_label)
    ranked = sorted(
        (
            (_label_score(query, str(node.metadata.get("canonical_label", ""))), node)
            for node in phases
        ),
        key=lambda item: (-item[0], _node_start(item[1])),
    )
    score, phase = ranked[0]
    if score < 0.5:
        raise ValueError(f"No phase hypothesis matches {phase_label!r}")

    node_by_id = {node.id: node for node in graph.nodes}
    boundaries = [
        node_by_id[edge.target]
        for edge in graph.edges
        if edge.source == phase.id
        and edge.relation == "has_boundary"
        and node_by_id[edge.target].metadata.get("boundary_kind") == boundary_kind
    ]
    if not boundaries:
        raise ValueError(f"Phase {phase.id} has no {boundary_kind} boundary")
    boundary = boundaries[0]
    anchor_ids = [
        edge.target
        for edge in graph.edges
        if edge.source == boundary.id and edge.relation == "grounded_by"
    ]
    events = sorted(
        (node for node in graph.nodes if node.node_type == "temporal_event"),
        key=_node_start,
    )
    position = {node.id: index for index, node in enumerate(events)}
    selected_ids: list[str] = []
    for anchor_id in anchor_ids:
        anchor_position = position[anchor_id]
        if boundary_kind == "onset":
            selected_ids.extend(
                node.id
                for node in events[
                    anchor_position : anchor_position + context_events + 1
                ]
            )
        else:
            selected_ids.extend(
                node.id
                for node in events[
                    max(0, anchor_position - context_events) : anchor_position + 1
                ]
            )
    selected_ids = list(dict.fromkeys(selected_ids))
    phase_scoped = bool(phase.metadata.get("phase_projection_version"))
    phase_support = {
        edge.target: edge.evidence
        for edge in graph.edges
        if edge.source == phase.id
        and edge.relation == "derived_from"
        and node_by_id[edge.target].node_type == "temporal_event"
    }
    if phase_scoped:
        selected_ids = [eid for eid in selected_ids if eid in phase_support]
    tracks = {}
    track_support = defaultdict(list)
    for edge in graph.edges:
        if edge.relation != "visible_during" or edge.target not in selected_ids:
            continue
        intervals = (
            intersect_evidence(edge.evidence, phase_support[edge.target])
            if phase_scoped
            else edge.evidence
        )
        if phase_scoped and not intervals:
            continue
        tracks[edge.source] = node_by_id[edge.source]
        track_support[edge.source].append(edge.target)
    instruments = sorted(
        (
            {
                "track_id": node.id,
                "label": node.label,
                "canonical_label": node.metadata.get("canonical_label"),
                "canonical_instrument": node.metadata.get(
                    "canonical_instrument", node.metadata.get("canonical_label")
                ),
                "appearance_signature": node.metadata.get("appearance_signature"),
                "surface_forms": node.metadata.get("surface_forms", []),
                "tracking_scope": node.metadata.get("tracking_scope"),
                "physical_identity_confirmed": node.metadata.get(
                    "physical_identity_confirmed"
                ),
                "fact_status": node.metadata.get("fact_status"),
                "confidence": node.confidence,
                "supporting_event_ids": [
                    event_id
                    for event_id in node.metadata.get("supporting_event_ids", [])
                    if event_id in track_support[node.id]
                ],
            }
            for node in tracks.values()
        ),
        key=lambda item: (-float(item["confidence"]), str(item["canonical_label"])),
    )
    evidence = _dedupe_intervals(
        interval
        for event_id in selected_ids
        for interval in (
            phase_support[event_id] if phase_scoped else node_by_id[event_id].evidence
        )
    )
    return {
        "phase_scoped_evidence": phase_scoped,
        "phase_hypothesis_id": phase.id,
        "phase_label": phase.label,
        "phase_match_score": round(score, 4),
        "phase_confidence": phase.confidence,
        "boundary_id": boundary.id,
        "boundary_kind": boundary_kind,
        "boundary_seconds": boundary.metadata.get("timestamp_seconds"),
        "event_ids": selected_ids,
        "instruments": instruments,
        "evidence": [item.to_dict() for item in evidence],
        "reasoning_path": [
            phase.id,
            "has_boundary",
            boundary.id,
            "grounded_by",
            *selected_ids,
            "inverse:visible_during",
            *[item["track_id"] for item in instruments],
        ],
    }


def extract_phase_name(question: str) -> str | None:
    patterns = (
        r"transition into the (.+?) phase",
        r"phase onset for (.+?)(?:,|\?| which)",
        r"when the (.+?) phase begins",
        r"start of the (.+?) phase",
        r"beginning of (.+?)(?:,|\?| which)",
        r"opening moments of the (.+?) phase",
    )
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return match.group(1).strip(" '\"")
    return None


def _append_semantic_instrument_tracks(
    graph: VideoEvidenceGraph,
    events: Sequence[GraphNode],
    event_position: dict[str, int],
    row_by_event: dict[str, dict[str, Any]],
    phase_event_membership: dict[str, dict[str, list[EvidenceInterval]]],
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    *,
    max_gap: int,
) -> int:
    instrument_detections: dict[str, list[tuple[GraphNode, str, float, str]]] = (
        defaultdict(list)
    )
    for event in events:
        row = row_by_event.get(event.id, {})
        seen: set[str] = set()
        for item in _instrument_items(row):
            label = str(item.get("label", "")).strip()
            canonical = _canonical_label(label)
            if canonical in _UNKNOWN_LABELS or canonical in seen:
                continue
            seen.add(canonical)
            instrument_detections[canonical].append(
                (
                    event,
                    label,
                    _confidence(item.get("confidence")),
                    str(item.get("basis", "")).strip(),
                )
            )

    track_count = 0
    for canonical, detections in sorted(instrument_detections.items()):
        runs = _detection_runs(detections, event_position, max_gap=max_gap)
        for run in runs:
            track_id = f"instrument_track:{graph.video_id}:{track_count:05d}"
            track_count += 1
            event_ids = [item[0].id for item in run]
            confidence = _mean(item[2] for item in run)
            evidence = _dedupe_intervals(
                interval for event, _, _, _ in run for interval in event.evidence
            )
            nodes.append(
                GraphNode(
                    track_id,
                    graph.video_id,
                    "instrument_track",
                    run[0][1],
                    evidence,
                    confidence=confidence,
                    metadata={
                        "canonical_label": canonical,
                        "canonical_instrument": canonical,
                        "supporting_event_ids": event_ids,
                        "detections": [
                            {
                                "event_id": event.id,
                                "confidence": item_confidence,
                                "basis": basis,
                            }
                            for event, _, item_confidence, basis in run
                        ],
                        "tracking_scope": "type_presence_not_physical_identity",
                        "physical_identity_confirmed": False,
                        "derived": True,
                        "fact_status": "medical_hypothesis",
                        "semantic_layer_version": SEMANTIC_LAYER_VERSION,
                    },
                )
            )
            co_occurring_phases: dict[str, list[EvidenceInterval]] = defaultdict(list)
            for event, _, item_confidence, basis in run:
                edges.append(
                    GraphEdge(
                        track_id,
                        event.id,
                        "visible_during",
                        list(event.evidence),
                        item_confidence,
                        {"basis": basis},
                    )
                )
                for phase_id, windows in phase_event_membership.get(
                    event.id, {}
                ).items():
                    co_occurring_phases[phase_id].extend(
                        intersect_evidence(event.evidence, windows)
                    )
            for phase_id, intervals in co_occurring_phases.items():
                if not intervals:
                    continue
                edges.append(
                    GraphEdge(
                        track_id,
                        phase_id,
                        "co_occurs",
                        _dedupe_intervals(intervals),
                        confidence,
                    )
                )
    return track_count


def _append_appearance_instrument_tracks(
    graph: VideoEvidenceGraph,
    events: Sequence[GraphNode],
    event_position: dict[str, int],
    phase_event_membership: dict[str, dict[str, list[EvidenceInterval]]],
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    *,
    max_gap: int,
) -> int:
    """Track conservative appearance types without asserting medical identity."""
    event_by_clip: dict[str, GraphNode] = {}
    event_by_id = {event.id: event for event in events}
    for event in events:
        for clip_id in event.metadata.get("supporting_clip_ids", []):
            event_by_clip[str(clip_id)] = event

    node_by_id = {node.id: node for node in graph.nodes}
    roles_by_mention: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.relation not in {"has_subject", "acts_on"}:
            continue
        action = node_by_id.get(edge.source)
        mention = node_by_id.get(edge.target)
        if not action or action.node_type != "action_event" or not mention:
            continue
        role = "subject" if edge.relation == "has_subject" else "target"
        roles_by_mention[mention.id].add(
            f"{role}:{_canonical_label(action.label).replace(' ', '_')}"
        )

    detections_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped: dict[tuple[str, str], list[GraphNode]] = defaultdict(list)
    signatures: dict[tuple[str, str], dict[str, Any]] = {}
    for mention in graph.nodes:
        if mention.node_type != "entity_mention":
            continue
        if mention.metadata.get("category") != "instrument":
            continue
        if mention.metadata.get("source_field") != "visible_instruments":
            continue
        clip_id = str(mention.metadata.get("clip_id", ""))
        event = event_by_clip.get(clip_id)
        if event is None:
            continue
        signature = _appearance_signature(mention)
        base_key = _appearance_signature_key(signature)
        if signature["canonical_family"] == "generic_instrument" and not any(
            signature[field] for field in ("colors", "shapes", "markings")
        ):
            base_key = f"{base_key}|event:{event.id}"
        group_key = (event.id, base_key)
        grouped[group_key].append(mention)
        signatures[group_key] = signature

    for (event_id, key), mentions in grouped.items():
        event = event_by_id[event_id]
        roles = sorted(
            {role for mention in mentions for role in roles_by_mention[mention.id]}
        )
        detections_by_key[key].append(
            {
                "event": event,
                "mentions": mentions,
                "signature": signatures[(event_id, key)],
                "surface_forms": sorted({item.label for item in mentions}),
                "action_roles": roles,
            }
        )

    track_count = 0
    for key, detections in sorted(detections_by_key.items()):
        ordered = sorted(detections, key=lambda item: event_position[item["event"].id])
        runs: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        previous_position: int | None = None
        for detection in ordered:
            position = event_position[detection["event"].id]
            if (
                previous_position is not None
                and position - previous_position > max_gap + 1
            ):
                runs.append(current)
                current = []
            current.append(detection)
            previous_position = position
        if current:
            runs.append(current)

        for run in runs:
            track_id = f"instrument_track:{graph.video_id}:{track_count:05d}"
            track_count += 1
            source_mentions = [
                mention for detection in run for mention in detection["mentions"]
            ]
            event_ids = [detection["event"].id for detection in run]
            surface_forms = sorted({item.label for item in source_mentions})
            signature = _merge_appearance_signatures(
                detection["signature"] for detection in run
            )
            confidence = min(0.5 + 0.1 * (len(event_ids) - 1), 0.85)
            evidence = _dedupe_intervals(
                interval for mention in source_mentions for interval in mention.evidence
            )
            nodes.append(
                GraphNode(
                    track_id,
                    graph.video_id,
                    "instrument_track",
                    _appearance_track_label(signature, surface_forms),
                    evidence,
                    confidence=confidence,
                    metadata={
                        "canonical_label": signature["canonical_family"],
                        "canonical_instrument": "unknown",
                        "semantic_candidates": [],
                        "appearance_signature": signature,
                        "supporting_event_ids": event_ids,
                        "source_mention_ids": [item.id for item in source_mentions],
                        "surface_forms": surface_forms,
                        "detections": [
                            {
                                "event_id": detection["event"].id,
                                "mention_ids": [
                                    item.id for item in detection["mentions"]
                                ],
                                "surface_forms": detection["surface_forms"],
                                "action_roles": detection["action_roles"],
                            }
                            for detection in run
                        ],
                        "tracking_scope": (
                            "appearance_type_presence_not_physical_identity"
                        ),
                        "physical_identity_confirmed": False,
                        "derived": True,
                        "fact_status": "derived_observation_track",
                        "appearance_track_version": APPEARANCE_TRACK_VERSION,
                        "semantic_layer_version": SEMANTIC_LAYER_VERSION,
                    },
                )
            )
            co_occurring_phases: dict[str, list[EvidenceInterval]] = defaultdict(list)
            for detection in run:
                event = detection["event"]
                event_mentions = detection["mentions"]
                event_evidence = _dedupe_intervals(
                    interval
                    for mention in event_mentions
                    for interval in mention.evidence
                )
                edges.append(
                    GraphEdge(
                        track_id,
                        event.id,
                        "visible_during",
                        event_evidence,
                        confidence,
                        {
                            "mention_ids": [item.id for item in event_mentions],
                            "action_roles": detection["action_roles"],
                        },
                    )
                )
                for mention in event_mentions:
                    edges.append(
                        GraphEdge(
                            track_id,
                            mention.id,
                            "derived_from",
                            list(mention.evidence),
                            confidence,
                            {"derivation": "appearance_signature_grouping"},
                        )
                    )
                for phase_id, windows in phase_event_membership.get(
                    event.id, {}
                ).items():
                    co_occurring_phases[phase_id].extend(
                        intersect_evidence(event_evidence, windows)
                    )
            for phase_id, intervals in co_occurring_phases.items():
                if not intervals:
                    continue
                edges.append(
                    GraphEdge(
                        track_id,
                        phase_id,
                        "co_occurs",
                        _dedupe_intervals(intervals),
                        confidence,
                    )
                )
    return track_count


def _appearance_signature(mention: GraphNode) -> dict[str, Any]:
    attributes = mention.metadata.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}
    canonical = str(mention.metadata.get("canonical", "generic_instrument"))
    canonical = _canonical_label(canonical).replace(" ", "_") or "generic_instrument"
    signature: dict[str, Any] = {"canonical_family": canonical}
    for field, output in (
        ("color", "colors"),
        ("shape", "shapes"),
        ("material", "materials"),
        ("appearance", "appearance"),
        ("size", "sizes"),
    ):
        value = attributes.get(field, [])
        values = value if isinstance(value, list) else [value]
        signature[output] = sorted(
            {
                _canonical_label(str(item)).replace(" ", "_")
                for item in values
                if _canonical_label(str(item)) not in _UNKNOWN_LABELS
            }
        )
    signature["markings"] = _extract_visible_markings(mention.label)
    return signature


def _appearance_signature_key(signature: dict[str, Any]) -> str:
    family = str(signature["canonical_family"])
    fields = [family]
    names = ("markings",)
    if family == "generic_instrument":
        names = ("colors", "shapes", "markings")
    for name in names:
        fields.append(f"{name}:{','.join(signature[name])}")
    return "|".join(fields)


def _merge_appearance_signatures(
    signatures: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    items = list(signatures)
    result: dict[str, Any] = {"canonical_family": items[0]["canonical_family"]}
    for name in ("colors", "shapes", "materials", "appearance", "sizes", "markings"):
        result[name] = sorted({value for item in items for value in item[name]})
    return result


def _extract_visible_markings(label: str) -> list[str]:
    values = re.findall(r"['\"]([^'\"]{1,40})['\"]", str(label))
    values.extend(
        match.group(1)
        for match in re.finditer(
            r"(?:marked|labeled|engraved)\s+(?:with\s+)?([a-z0-9][a-z0-9 -]{0,39})",
            str(label),
            re.IGNORECASE,
        )
    )
    return sorted(
        {
            _canonical_label(value).replace(" ", "_")
            for value in values
            if _canonical_label(value) not in _UNKNOWN_LABELS
        }
    )


def _appearance_track_label(
    signature: dict[str, Any], surface_forms: Sequence[str]
) -> str:
    family = str(signature["canonical_family"]).replace("_", " ")
    descriptors = [
        *signature["colors"],
        *signature["shapes"],
        *[f"marked {item}" for item in signature["markings"]],
    ]
    if descriptors:
        return f"{family} ({', '.join(item.replace('_', ' ') for item in descriptors)})"
    if surface_forms:
        return min(surface_forms, key=lambda item: (len(item), item.lower()))
    return family


def build_video_semantic_ontology(
    questions: Sequence[Any], video_key: str
) -> dict[str, Any]:
    """Build a candidate-only diagnostic ontology without reading QA answers."""
    phases: set[str] = set()
    instruments: set[str] = set()
    source_uids = []
    for item in questions:
        if str(getattr(item, "video_key", "")) != str(video_key):
            continue
        task_name = getattr(item, "task_name", None)
        metadata = getattr(item, "metadata", {}) or {}
        if task_name == "Phase-Instrument Association":
            phase = extract_phase_name(str(getattr(item, "question", "")))
            if phase:
                phases.add(phase)
            instruments.update(
                _option_text(option) for option in getattr(item, "options", [])
            )
            source_uids.append(str(getattr(item, "uid", "")))
        elif (
            task_name == "Action Recognition"
            and metadata.get("natural_rewrite_v1_kind") == "surgical_phase"
        ):
            phases.update(
                _option_text(option) for option in getattr(item, "options", [])
            )
            source_uids.append(str(getattr(item, "uid", "")))
    if not phases or not instruments:
        raise ValueError(
            f"Video {video_key} has no phase-instrument candidate ontology"
        )
    return {
        "video_id": str(video_key),
        "phases": sorted(phases),
        "instruments": sorted(item for item in instruments if item),
        "source": "phase_instrument_question_candidates",
        "source_uids": source_uids,
        "answers_used": False,
        "evaluation_mode": "candidate_aware_diagnostic",
    }


def write_semantic_layer_artifacts(
    artifacts: SemanticLayerArtifacts, output_dir: str | Path
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "semantic_evidence_graph.json").write_text(
        json.dumps(artifacts.graph.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "semantic_hypotheses.jsonl").open("w", encoding="utf-8") as handle:
        for row in artifacts.hypotheses:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "semantic_graph_report.json").write_text(
        json.dumps(artifacts.report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _phase_runs(
    events: Sequence[GraphNode], row_by_event: dict[str, dict[str, Any]]
) -> list[list[tuple[GraphNode, str, float, str]]]:
    runs: list[list[tuple[GraphNode, str, float, str]]] = []
    current: list[tuple[GraphNode, str, float, str]] = []
    current_label = ""
    for event in events:
        item = row_by_event.get(event.id, {}).get("phase_hypothesis")
        if not isinstance(item, dict):
            if current:
                runs.append(current)
                current, current_label = [], ""
            continue
        label = str(item.get("label", "")).strip()
        canonical = _canonical_label(label)
        if canonical in _UNKNOWN_LABELS:
            if current:
                runs.append(current)
                current, current_label = [], ""
            continue
        detection = (
            event,
            label,
            _confidence(item.get("confidence")),
            str(item.get("basis", "")).strip(),
        )
        if current and canonical != current_label:
            runs.append(current)
            current = []
        current.append(detection)
        current_label = canonical
    if current:
        runs.append(current)
    return runs


def _instrument_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items = row.get("instrument_hypotheses", [])
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        raise TypeError("instrument_hypotheses must be a list")
    return [item for item in items if isinstance(item, dict)]


def _detection_runs(
    detections: Sequence[tuple[GraphNode, str, float, str]],
    positions: dict[str, int],
    *,
    max_gap: int,
) -> list[list[tuple[GraphNode, str, float, str]]]:
    ordered = sorted(detections, key=lambda item: positions[item[0].id])
    runs: list[list[tuple[GraphNode, str, float, str]]] = []
    current: list[tuple[GraphNode, str, float, str]] = []
    previous_position: int | None = None
    for item in ordered:
        position = positions[item[0].id]
        if previous_position is not None and position - previous_position > max_gap + 1:
            runs.append(current)
            current = []
        current.append(item)
        previous_position = position
    if current:
        runs.append(current)
    return runs


def _sources(
    run: Sequence[tuple[GraphNode, str, float, str]],
    row_by_event: dict[str, dict[str, Any]],
) -> list[str]:
    return sorted(
        {
            str(row_by_event[event.id].get("source", "unspecified"))
            for event, _, _, _ in run
        }
    )


def _confidence(value: Any) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _CONFIDENCE:
            return _CONFIDENCE[lowered]
        value = float(lowered)
    result = float(value if value is not None else _CONFIDENCE["low"])
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"Semantic confidence must be between 0 and 1: {value}")
    return result


def _canonical_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _label_score(query: str, candidate: str) -> float:
    if query == candidate:
        return 1.0
    query_tokens, candidate_tokens = set(query.split()), set(candidate.split())
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)


def _node_start(node: GraphNode) -> float:
    return min(item.start_seconds for item in node.evidence)


def _node_end(node: GraphNode) -> float:
    return max(item.end_seconds for item in node.evidence)


def _dedupe_intervals(items: Iterable[EvidenceInterval]) -> list[EvidenceInterval]:
    result: list[EvidenceInterval] = []
    seen: set[tuple[str, float, float]] = set()
    for item in items:
        key = (item.video_id, item.start_seconds, item.end_seconds)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return sorted(result, key=lambda item: (item.start_seconds, item.end_seconds))


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 4)


def _option_text(option: str) -> str:
    return re.sub(r"^\s*[A-Za-z0-9]+[.):：]\s*", "", str(option)).strip()
