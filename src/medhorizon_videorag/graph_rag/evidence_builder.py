from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from .schemas import EvidenceInterval, GraphEdge, GraphNode, VideoEvidenceGraph

BUILDER_VERSION = "observation-evidence-graph-v1"

_COLOR_TERMS = (
    "red",
    "reddish",
    "blue",
    "white",
    "whitish",
    "yellow",
    "yellowish",
    "pink",
    "pinkish",
    "purple",
    "orange",
    "dark",
    "clear",
    "translucent",
    "metallic",
)

_ENTITY_RULES: tuple[tuple[str, str, str], ...] = (
    (r"\b(?:blood\s+)?vessels?\b", "tubular_structure", "anatomy"),
    (r"\btubular (?:structure|object)s?\b", "tubular_structure", "anatomy"),
    (r"\b(?:organ|body cavity|cavity)\b", "tissue_region", "anatomy"),
    (
        r"\bmembran(?:e|ous)(?: layer| structure| surface)?s?\b",
        "membranous_structure",
        "anatomy",
    ),
    (
        r"\b(?:reddish|pinkish|yellowish|white|whitish|smooth|pale|fatty-looking)?\s*tissue(?: layers?| mass| fragments?)?\b",
        "tissue",
        "anatomy",
    ),
    (r"\b(?:metal(?:lic)? )?forceps\b", "grasping_instrument", "instrument"),
    (
        r"\b(?:metal(?:lic)? )?(?:grasper|grasping (?:instrument|tool))\b",
        "grasping_instrument",
        "instrument",
    ),
    (r"\bneedle-like instrument\b", "needle_like_instrument", "instrument"),
    (
        r"\b(?:metal(?:lic)? )?(?:scissors|cutting instrument)\b",
        "cutting_instrument",
        "instrument",
    ),
    (r"\bprobe\b", "probe_instrument", "instrument"),
    (
        r"\b(?:suction|aspiration) (?:instrument|tool|tube)\b",
        "suction_instrument",
        "instrument",
    ),
    (r"\b(?:instrument|tool)s?(?: with [^,]+)?\b", "generic_instrument", "instrument"),
    (
        r"\b(?:suture|(?:(?:multiple|thin|blue|white|dark)\s+)*thread-like materials?)\b",
        "thread_like_material",
        "material",
    ),
    (
        r"\b(?:mesh|mesh-like|grid-like)(?: structures?| materials?| rings?| coverings?)?\b",
        "grid_like_material",
        "material",
    ),
    (r"\bblood\b|\bred fluid(?: droplets?)?\b", "red_fluid", "object"),
    (r"\bclear fluid(?: droplets?)?\b", "clear_fluid", "object"),
)

_ACTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"\bgrasp(?:s|ed|ing)? and pull(?:s|ed|ing)?\b", ("grasp", "pull")),
    (r"\binsert(?:s|ed|ing)? and guid(?:e|es|ed|ing)?\b", ("insert_into", "guide")),
    (
        r"\b(?:pass(?:es|ed|ing)?|push(?:es|ed|ing)?|insert(?:s|ed|ing)?)(?: [a-z-]+){0,4} through(?: tissue)?\b",
        ("pass_through",),
    ),
    (r"\bpass(?:es|ed|ing)?(?: thread-like material)?\b", ("pass_through",)),
    (r"\binsert(?:s|ed|ing)? into(?: tissue)?\b", ("insert_into",)),
    (r"\bhold(?:s|ing)?(?: tissue)?\b", ("hold",)),
    (r"\bgrasp(?:s|ed|ing)?\b", ("grasp",)),
    (r"\bpull(?:s|ed|ing)?(?: thread-like material)?\b", ("pull",)),
    (r"\bpress(?:es|ed|ing)? against(?: tissue)?\b", ("press_against",)),
    (r"\bcontact(?:s|ed|ing)?\b", ("contact",)),
    (r"\btouch(?:es|ed|ing)?\b", ("contact",)),
    (r"\bmanipulat(?:e|es|ed|ing)\b", ("manipulate",)),
    (r"\bmov(?:e|es|ed|ing)\b", ("move",)),
    (r"\bretract(?:s|ed|ing)?\b", ("pull",)),
    (r"\bguid(?:e|es|ed|ing)?\b", ("guide",)),
    (r"\bemit(?:s|ted|ting)?\b", ("emit",)),
    (r"\bdeliver(?:s|ed|ing)?\b", ("deliver",)),
    (r"\bremov(?:e|es|ed|ing)\b", ("remove",)),
    (r"\bsecur(?:e|es|ed|ing)\b", ("attach",)),
    (r"\bappl(?:y|ies|ied|ying)\b", ("apply",)),
)

_MERGE_STOP_CONCEPTS = frozenset(
    {"tissue", "tissue_region", "generic_instrument", "red_fluid", "clear_fluid"}
)


@dataclass(frozen=True)
class NormalizedMention:
    id: str
    surface: str
    canonical: str
    category: str
    source_field: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedAction:
    id: str
    predicate: str
    original_action: str
    subject_mention_id: str
    target_mention_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedClip:
    clip_id: str
    video_id: str
    clip_index: int
    start_seconds: float
    end_seconds: float
    summary: str
    observation: dict[str, Any]
    mentions: list[NormalizedMention]
    actions: list[NormalizedAction]
    source_frames: int
    input_frames: int
    padding_frames: int
    prompt_version: str
    frame_paths: list[str] = field(default_factory=list)

    @property
    def concepts(self) -> set[str]:
        return {mention.canonical for mention in self.mentions}

    @property
    def predicates(self) -> set[str]:
        return {action.predicate for action in self.actions}

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "video_id": self.video_id,
            "clip_index": self.clip_index,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "summary": self.summary,
            "mentions": [item.to_dict() for item in self.mentions],
            "actions": [item.to_dict() for item in self.actions],
            "state_changes": self.observation["observed_facts"]["state_changes"],
            "visual_evidence": self.observation["observed_facts"]["visual_evidence"],
            "uncertainties": self.observation["uncertainties"],
            "medical_inferences_excluded_from_graph": self.observation[
                "medical_inferences"
            ],
            "source_frames": self.source_frames,
            "input_frames": self.input_frames,
            "padding_frames": self.padding_frames,
            "prompt_version": self.prompt_version,
        }


@dataclass(frozen=True)
class TemporalEvent:
    id: str
    video_id: str
    start_seconds: float
    end_seconds: float
    supporting_clip_ids: list[str]
    concepts: list[str]
    predicates: list[str]
    merge_scores: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceGraphArtifacts:
    graph: VideoEvidenceGraph
    normalized_clips: list[NormalizedClip]
    temporal_events: list[TemporalEvent]
    report: dict[str, Any]


def normalize_entity(
    surface: str, *, category_hint: str | None = None
) -> tuple[str, str, dict[str, Any]]:
    """Return a conservative canonical concept while preserving the surface form."""
    lowered = " ".join(str(surface).lower().split())
    attributes = {
        "colors": sorted(
            {term for term in _COLOR_TERMS if re.search(rf"\b{term}\b", lowered)}
        )
    }
    for pattern, canonical, category in _ENTITY_RULES:
        if re.search(pattern, lowered):
            return canonical, category, attributes
    category = category_hint or "object"
    return _slug(lowered) or "unidentified_object", category, attributes


def normalize_action(action: str) -> tuple[str, ...]:
    lowered = " ".join(str(action).lower().split())
    for pattern, predicates in _ACTION_RULES:
        if re.search(pattern, lowered):
            return predicates
    return (_slug(lowered) or "unspecified_action",)


def load_description_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Description JSONL is empty")
    clip_ids = [str(row["clip_id"]) for row in rows]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("Description JSONL contains duplicate clip IDs")
    video_ids = {str(row["video_id"]) for row in rows}
    if len(video_ids) != 1:
        raise ValueError("Evidence graph input must contain exactly one video")
    return sorted(rows, key=lambda row: int(row["clip_index"]))


def load_manifest_frame_paths(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = payload.get("clips")
    if not isinstance(clips, list):
        raise TypeError("Video manifest must contain a clips list")
    return {
        str(clip["id"]): [str(item) for item in clip["frame_paths"]] for clip in clips
    }


def normalize_description_rows(
    rows: list[dict[str, Any]],
    *,
    frame_paths_by_clip: dict[str, list[str]] | None = None,
) -> list[NormalizedClip]:
    frame_paths_by_clip = frame_paths_by_clip or {}
    return [
        _normalize_description_row(
            row, frame_paths_by_clip.get(str(row["clip_id"]), [])
        )
        for row in rows
    ]


def merge_temporal_events(
    clips: list[NormalizedClip],
    *,
    threshold: float = 0.45,
    max_merged_clips: int = 5,
    max_gap_seconds: float = 1.0,
) -> list[TemporalEvent]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if max_merged_clips < 1:
        raise ValueError("max_merged_clips must be at least 1")
    if not clips:
        return []
    groups: list[tuple[list[NormalizedClip], list[float]]] = []
    current = [clips[0]]
    scores: list[float] = []
    for clip in clips[1:]:
        previous = current[-1]
        score, has_continuity = _clip_continuity(previous, clip)
        gap = clip.start_seconds - previous.end_seconds
        can_merge = (
            gap <= max_gap_seconds
            and len(current) < max_merged_clips
            and score >= threshold
            and has_continuity
        )
        if can_merge:
            current.append(clip)
            scores.append(round(score, 4))
        else:
            groups.append((current, scores))
            current = [clip]
            scores = []
    groups.append((current, scores))

    events = []
    for index, (group, merge_scores) in enumerate(groups):
        events.append(
            TemporalEvent(
                id=f"event:{group[0].video_id}:{index:05d}",
                video_id=group[0].video_id,
                start_seconds=group[0].start_seconds,
                end_seconds=group[-1].end_seconds,
                supporting_clip_ids=[clip.clip_id for clip in group],
                concepts=sorted(set().union(*(clip.concepts for clip in group))),
                predicates=sorted(set().union(*(clip.predicates for clip in group))),
                merge_scores=merge_scores,
            )
        )
    return events


def build_evidence_graph(
    rows: list[dict[str, Any]],
    *,
    frame_paths_by_clip: dict[str, list[str]] | None = None,
    merge_threshold: float = 0.45,
    max_merged_clips: int = 5,
) -> EvidenceGraphArtifacts:
    clips = normalize_description_rows(rows, frame_paths_by_clip=frame_paths_by_clip)
    video_id = clips[0].video_id
    events = merge_temporal_events(
        clips,
        threshold=merge_threshold,
        max_merged_clips=max_merged_clips,
    )
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    concept_evidence: dict[tuple[str, str], dict[str, EvidenceInterval]] = defaultdict(
        dict
    )
    concept_mentions: dict[tuple[str, str], list[str]] = defaultdict(list)
    action_by_clip: dict[str, list[NormalizedAction]] = {}
    mention_by_id: dict[str, NormalizedMention] = {}

    for clip in clips:
        interval = _interval(clip, include_frames=True)
        clip_node_id = _clip_node_id(clip.clip_id)
        nodes.append(
            GraphNode(
                id=clip_node_id,
                video_id=video_id,
                node_type="segment",
                label=clip.summary,
                evidence=[interval],
                metadata={
                    "clip_id": clip.clip_id,
                    "clip_index": clip.clip_index,
                    "prompt_version": clip.prompt_version,
                    "source_frames": clip.source_frames,
                    "input_frames": clip.input_frames,
                    "padding_frames": clip.padding_frames,
                    "observation": clip.observation,
                },
            )
        )
        action_by_clip[clip.clip_id] = clip.actions
        for mention in clip.mentions:
            mention_by_id[mention.id] = mention
            evidence = _interval(clip)
            nodes.append(
                GraphNode(
                    id=mention.id,
                    video_id=video_id,
                    node_type="entity_mention",
                    label=mention.surface,
                    evidence=[evidence],
                    metadata={
                        "canonical": mention.canonical,
                        "category": mention.category,
                        "source_field": mention.source_field,
                        "attributes": mention.attributes,
                        "clip_id": clip.clip_id,
                    },
                )
            )
            concept_key = (mention.category, mention.canonical)
            concept_id = _concept_node_id(*concept_key)
            concept_evidence[concept_key][clip.clip_id] = evidence
            concept_mentions[concept_key].append(mention.id)
            edges.extend(
                [
                    GraphEdge(mention.id, clip_node_id, "observed_in", [evidence]),
                    GraphEdge(mention.id, concept_id, "instance_of", [evidence], 0.95),
                ]
            )
        for action in clip.actions:
            evidence = _interval(clip)
            nodes.append(
                GraphNode(
                    id=action.id,
                    video_id=video_id,
                    node_type="action_event",
                    label=action.predicate,
                    evidence=[evidence],
                    metadata={
                        "original_action": action.original_action,
                        "clip_id": clip.clip_id,
                    },
                )
            )
            action_key = ("action", action.predicate)
            action_concept_id = _concept_node_id(*action_key)
            concept_evidence[action_key][clip.clip_id] = evidence
            concept_mentions[action_key].append(action.id)
            edges.extend(
                [
                    GraphEdge(action.id, clip_node_id, "observed_in", [evidence]),
                    GraphEdge(
                        action.id, action_concept_id, "instance_of", [evidence], 0.98
                    ),
                    GraphEdge(
                        action.id, action.subject_mention_id, "has_subject", [evidence]
                    ),
                    GraphEdge(
                        action.id, action.target_mention_id, "acts_on", [evidence]
                    ),
                ]
            )

    for (category, canonical), evidence_by_clip in sorted(concept_evidence.items()):
        nodes.append(
            GraphNode(
                id=_concept_node_id(category, canonical),
                video_id=video_id,
                node_type="concept",
                label=canonical,
                evidence=list(evidence_by_clip.values()),
                confidence=0.95,
                metadata={
                    "category": category,
                    "mention_ids": concept_mentions[(category, canonical)],
                    "mention_count": len(concept_mentions[(category, canonical)]),
                },
            )
        )

    for previous, current in pairwise(clips):
        edges.append(
            GraphEdge(
                _clip_node_id(previous.clip_id),
                _clip_node_id(current.clip_id),
                "temporal_before",
                [_interval(previous), _interval(current)],
            )
        )
        edges.extend(
            _continuation_edges(previous, current, mention_by_id, action_by_clip)
        )

    clip_by_id = {clip.clip_id: clip for clip in clips}
    for event in events:
        event_intervals = [
            _interval(clip_by_id[clip_id]) for clip_id in event.supporting_clip_ids
        ]
        nodes.append(
            GraphNode(
                id=event.id,
                video_id=video_id,
                node_type="temporal_event",
                label=_event_label(event),
                evidence=event_intervals,
                confidence=min(event.merge_scores, default=1.0),
                metadata={
                    "supporting_clip_ids": event.supporting_clip_ids,
                    "concepts": event.concepts,
                    "predicates": event.predicates,
                    "merge_scores": event.merge_scores,
                    "derived": True,
                },
            )
        )
        for clip_id in event.supporting_clip_ids:
            clip = clip_by_id[clip_id]
            edges.append(
                GraphEdge(
                    event.id, _clip_node_id(clip_id), "contains", [_interval(clip)]
                )
            )
            for action in action_by_clip[clip_id]:
                edges.append(
                    GraphEdge(action.id, event.id, "part_of", [_interval(clip)])
                )
    for previous, current in pairwise(events):
        edges.append(
            GraphEdge(
                previous.id,
                current.id,
                "temporal_before",
                [
                    EvidenceInterval(
                        video_id, previous.start_seconds, previous.end_seconds
                    ),
                    EvidenceInterval(
                        video_id, current.start_seconds, current.end_seconds
                    ),
                ],
            )
        )

    graph = VideoEvidenceGraph(
        video_id=video_id,
        nodes=nodes,
        edges=edges,
        schema_version="medical-video-evidence-graph-v1",
        metadata={
            "builder_version": BUILDER_VERSION,
            "source_clip_count": len(clips),
            "merge_threshold": merge_threshold,
            "max_merged_clips": max_merged_clips,
            "medical_inferences_used": False,
        },
    )
    report = _build_report(graph, clips, events)
    return EvidenceGraphArtifacts(graph, clips, events, report)


def write_evidence_graph_artifacts(
    artifacts: EvidenceGraphArtifacts, output_dir: str | Path
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(
        output / "normalized_observations.jsonl",
        [clip.to_dict() for clip in artifacts.normalized_clips],
    )
    _write_jsonl(
        output / "temporal_events.jsonl",
        [event.to_dict() for event in artifacts.temporal_events],
    )
    _write_json(output / "evidence_graph.json", artifacts.graph.to_dict())
    _write_json(output / "graph_report.json", artifacts.report)


def _normalize_description_row(
    row: dict[str, Any], frame_paths: list[str]
) -> NormalizedClip:
    description = row["description"]
    observed = description["observed_facts"]
    mentions: list[NormalizedMention] = []
    mentions_by_key: dict[tuple[str, str], list[NormalizedMention]] = defaultdict(list)
    sequence = 0

    def add_mention(
        surface: str, source_field: str, category_hint: str
    ) -> NormalizedMention:
        nonlocal sequence
        canonical, category, attributes = normalize_entity(
            surface, category_hint=category_hint
        )
        mention = NormalizedMention(
            id=f"mention:{row['clip_id']}:{sequence:03d}",
            surface=str(surface),
            canonical=canonical,
            category=category,
            source_field=source_field,
            attributes=attributes,
        )
        sequence += 1
        mentions.append(mention)
        mentions_by_key[(category, canonical)].append(mention)
        return mention

    for field_name, category_hint in (
        ("visible_anatomy", "anatomy"),
        ("visible_instruments", "instrument"),
        ("visible_objects", "object"),
    ):
        for surface in observed[field_name]:
            add_mention(str(surface), field_name, category_hint)

    def argument_mention(surface: str, role: str) -> NormalizedMention:
        canonical, category, _ = normalize_entity(surface)
        existing = mentions_by_key.get((category, canonical))
        if existing:
            return existing[0]
        return add_mention(surface, role, category)

    actions: list[NormalizedAction] = []
    for action_index, action in enumerate(observed["actions"]):
        subject = argument_mention(
            str(action.get("subject", "unidentified object")), "action_subject"
        )
        target = argument_mention(
            str(action.get("target", "unidentified object")), "action_target"
        )
        original = str(action.get("action", "unspecified action"))
        for predicate_index, predicate in enumerate(normalize_action(original)):
            actions.append(
                NormalizedAction(
                    id=f"action:{row['clip_id']}:{action_index:03d}:{predicate_index:02d}",
                    predicate=predicate,
                    original_action=original,
                    subject_mention_id=subject.id,
                    target_mention_id=target.id,
                )
            )

    return NormalizedClip(
        clip_id=str(row["clip_id"]),
        video_id=str(row["video_id"]),
        clip_index=int(row["clip_index"]),
        start_seconds=float(row["start_seconds"]),
        end_seconds=float(row["end_seconds"]),
        summary=str(description["summary"]),
        observation=description,
        mentions=mentions,
        actions=actions,
        source_frames=int(row.get("source_frames", len(frame_paths))),
        input_frames=int(row.get("input_frames", len(frame_paths))),
        padding_frames=int(row.get("padding_frames", 0)),
        prompt_version=str(row.get("prompt_version", "unknown")),
        frame_paths=frame_paths,
    )


def _clip_continuity(
    first: NormalizedClip, second: NormalizedClip
) -> tuple[float, bool]:
    action_score = _jaccard(first.predicates, second.predicates)
    informative_first = first.concepts - _MERGE_STOP_CONCEPTS
    informative_second = second.concepts - _MERGE_STOP_CONCEPTS
    informative_score = _jaccard(informative_first, informative_second)
    all_concept_score = _jaccard(first.concepts, second.concepts)
    score = 0.55 * action_score + 0.3 * informative_score + 0.15 * all_concept_score
    has_continuity = bool(first.predicates & second.predicates) and (
        bool(informative_first & informative_second) or all_concept_score >= 0.5
    )
    return score, has_continuity


def _continuation_edges(
    previous: NormalizedClip,
    current: NormalizedClip,
    mention_by_id: dict[str, NormalizedMention],
    action_by_clip: dict[str, list[NormalizedAction]],
) -> list[GraphEdge]:
    evidence = [_interval(previous), _interval(current)]
    edges: list[GraphEdge] = []
    previous_mentions: dict[tuple[str, str], list[NormalizedMention]] = defaultdict(
        list
    )
    current_mentions: dict[tuple[str, str], list[NormalizedMention]] = defaultdict(list)
    for mention in previous.mentions:
        previous_mentions[(mention.category, mention.canonical)].append(mention)
    for mention in current.mentions:
        current_mentions[(mention.category, mention.canonical)].append(mention)
    for key in sorted(previous_mentions.keys() & current_mentions.keys()):
        if key[1] in _MERGE_STOP_CONCEPTS:
            continue
        edges.append(
            GraphEdge(
                previous_mentions[key][0].id,
                current_mentions[key][0].id,
                "possible_continuation",
                evidence,
                0.6,
                {"canonical": key[1], "strong_identity_claim": False},
            )
        )
    previous_actions = action_by_clip[previous.clip_id]
    current_actions = action_by_clip[current.clip_id]
    for left in previous_actions:
        left_signature = _action_signature(left, mention_by_id)
        for right in current_actions:
            if left_signature == _action_signature(right, mention_by_id):
                edges.append(
                    GraphEdge(
                        left.id,
                        right.id,
                        "possible_continuation",
                        evidence,
                        0.65,
                        {"signature": list(left_signature)},
                    )
                )
                break
    return edges


def _action_signature(
    action: NormalizedAction, mention_by_id: dict[str, NormalizedMention]
) -> tuple[str, str, str]:
    return (
        action.predicate,
        mention_by_id[action.subject_mention_id].canonical,
        mention_by_id[action.target_mention_id].canonical,
    )


def _build_report(
    graph: VideoEvidenceGraph,
    clips: list[NormalizedClip],
    events: list[TemporalEvent],
) -> dict[str, Any]:
    node_types = Counter(node.node_type for node in graph.nodes)
    edge_types = Counter(edge.relation for edge in graph.edges)
    concepts = Counter(mention.canonical for clip in clips for mention in clip.mentions)
    actions = Counter(action.predicate for clip in clips for action in clip.actions)
    return {
        "builder_version": BUILDER_VERSION,
        "video_id": graph.video_id,
        "schema_version": graph.schema_version,
        "source_clip_count": len(clips),
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "node_type_counts": dict(sorted(node_types.items())),
        "edge_type_counts": dict(sorted(edge_types.items())),
        "temporal_event_count": len(events),
        "merged_temporal_event_count": sum(
            len(event.supporting_clip_ids) > 1 for event in events
        ),
        "maximum_clips_per_event": max(
            (len(event.supporting_clip_ids) for event in events), default=0
        ),
        "medical_inference_item_count": sum(
            len(clip.observation["medical_inferences"]) for clip in clips
        ),
        "medical_inferences_used": False,
        "clips_with_frame_paths": sum(bool(clip.frame_paths) for clip in clips),
        "missing_frame_path_count": sum(
            not Path(path).is_file() for clip in clips for path in clip.frame_paths
        ),
        "top_concepts": [
            {"concept": key, "mentions": value}
            for key, value in concepts.most_common(20)
        ],
        "top_actions": [
            {"action": key, "events": value} for key, value in actions.most_common(20)
        ],
    }


def _interval(
    clip: NormalizedClip, *, include_frames: bool = False
) -> EvidenceInterval:
    return EvidenceInterval(
        video_id=clip.video_id,
        start_seconds=clip.start_seconds,
        end_seconds=clip.end_seconds,
        frame_paths=clip.frame_paths if include_frames else [],
        metadata={"clip_id": clip.clip_id, "clip_index": clip.clip_index},
    )


def _event_label(event: TemporalEvent) -> str:
    actions = ", ".join(event.predicates[:3]) or "no_normalized_action"
    concepts = ", ".join(event.concepts[:4]) or "no_normalized_concept"
    return f"{actions} | {concepts}"


def _clip_node_id(clip_id: str) -> str:
    return f"clip:{clip_id}"


def _concept_node_id(category: str, canonical: str) -> str:
    return f"concept:{category}:{canonical}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
