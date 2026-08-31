from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from statistics import mean, median
from typing import Any

from .mention_binding import MENTION_BINDING_VERSION, bind_mention
from .schemas import EvidenceInterval, GraphEdge, GraphNode, VideoEvidenceGraph

BUILDER_VERSION = "observation-evidence-graph-v3"
GRAPH_SCHEMA_VERSION = "medical-video-evidence-graph-v3"
EVENT_SUPPORT_VERSION = "event-structural-support-v1"
REPRESENTATIVE_EVIDENCE_VERSION = "event-representative-evidence-v1"

ACTION_VOCABULARY = frozenset(
    {
        "apply",
        "attach",
        "contact",
        "cut",
        "deliver",
        "emit",
        "grasp",
        "guide",
        "hold",
        "insert",
        "loop_around",
        "manipulate",
        "move",
        "other_action",
        "pass_through",
        "pierce",
        "position",
        "press",
        "pull",
        "push",
        "remove",
        "tighten",
    }
)

ENTITY_VOCABULARY = frozenset(
    {
        "clear_fluid",
        "clip_like_object",
        "cutting_instrument",
        "drape",
        "generic_instrument",
        "generic_material",
        "generic_object",
        "generic_structure",
        "grasping_instrument",
        "grid_like_material",
        "hand",
        "membranous_structure",
        "needle_like_instrument",
        "opening",
        "patch_material",
        "probe_instrument",
        "red_fluid",
        "ring_like_object",
        "suction_instrument",
        "surface_region",
        "thread_like_material",
        "tissue",
        "tissue_region",
        "tubular_instrument",
        "tubular_structure",
    }
)

_ATTRIBUTE_RULES: dict[str, tuple[tuple[str, str], ...]] = {
    "color": (
        (r"\b(?:red|reddish)\b", "red"),
        (r"\b(?:blue|bluish)\b", "blue"),
        (r"\b(?:white|whitish)\b", "white"),
        (r"\b(?:yellow|yellowish)\b", "yellow"),
        (r"\b(?:pink|pinkish)\b", "pink"),
        (r"\bpurple\b", "purple"),
        (r"\borange\b", "orange"),
        (r"\bblack\b", "black"),
        (r"\bdark\b", "dark"),
        (r"\bpale\b", "pale"),
    ),
    "appearance": (
        (r"\bfatty-looking\b", "fatty-looking"),
        (r"\b(?:glossy|glistening|shiny)\b", "glossy"),
        (r"\bsmooth\b", "smooth"),
        (r"\bmoist\b", "moist"),
        (r"\bfibrous\b", "fibrous"),
        (r"\bgranular\b", "granular"),
        (r"\btextured\b", "textured"),
        (r"\bporous\b", "porous"),
        (r"\birregular(?:ly shaped)?\b", "irregular"),
        (r"\b(?:clear|transparent|translucent)\b", "translucent"),
    ),
    "shape": (
        (r"\b(?:tubular|tube-like|cylindrical)\b", "tubular"),
        (r"\b(?:rounded|round)\b", "round"),
        (r"\boval(?:-shaped)?\b", "oval"),
        (r"\bcircular\b", "circular"),
        (r"\b(?:ring-like|ring)\b", "ring"),
        (r"\b(?:lobed|lobulated)\b", "lobed"),
        (r"\bsquare\b", "square"),
        (r"\brectangular\b", "rectangular"),
        (r"\bflat\b", "flat"),
        (r"\bcurved\b", "curved"),
        (r"\bpointed\b", "pointed"),
        (r"\bserrated\b", "serrated"),
        (r"\bbranching\b", "branching"),
    ),
    "size": (
        (r"\bsmall\b", "small"),
        (r"\bthin\b", "thin"),
        (r"\blong\b", "long"),
    ),
    "material": (
        (r"\b(?:metal|metallic)\b", "metal"),
        (r"\bplastic\b", "plastic"),
    ),
}

_ENTITY_RULES: tuple[tuple[str, str, str], ...] = (
    (r"\bmesh-covered tubular structure\b", "tubular_structure", "anatomy"),
    (
        r"\b(?:suture|thread|strands?)(?:-like)?(?: materials?)?\b",
        "thread_like_material",
        "material",
    ),
    (
        r"\b(?:mesh|mesh-like|grid-like)(?: structures?| materials?| rings?| coverings?)?\b",
        "grid_like_material",
        "material",
    ),
    (
        r"\bblood\b(?!\s+vessels?\b)|\bred fluid(?: droplets?| pools?| streaks?)?\b",
        "red_fluid",
        "object",
    ),
    (r"\bclear fluid(?: droplets?)?\b", "clear_fluid", "object"),
    (
        r"\b(?:forceps|grasper|clamp|needle holder)(?:-like)?\b",
        "grasping_instrument",
        "instrument",
    ),
    (r"\bneedle-like instrument\b", "needle_like_instrument", "instrument"),
    (
        r"\b(?:scissors?|scissor-like tool|cutting instrument|blade-like instrument)\b",
        "cutting_instrument",
        "instrument",
    ),
    (
        r"\b(?:suction|aspiration) (?:instrument|tool|tube)\b",
        "suction_instrument",
        "instrument",
    ),
    (r"\b(?:probe|orange-tipped tool)\b", "probe_instrument", "instrument"),
    (
        r"\b(?:tubular|tube-like|cylindrical|clear|red) (?:instrument|tool|device|tube)\b|\btubular access port\b",
        "tubular_instrument",
        "instrument",
    ),
    (r"\b(?:instrument|tool)s?(?: with [^,]+)?\b", "generic_instrument", "instrument"),
    (r"\b(?:blood\s+)?vessels?\b", "tubular_structure", "anatomy"),
    (r"\btubular (?:structure|object)s?\b", "tubular_structure", "anatomy"),
    (r"\b(?:opening|cavity)\b", "opening", "anatomy"),
    (r"\b(?:organ|body cavity|surgical site)\b", "tissue_region", "anatomy"),
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
    (
        r"\b(?:patch|square material|rectangular material)\b",
        "patch_material",
        "material",
    ),
    (r"\b(?:clip|anchor)\b", "clip_like_object", "object"),
    (r"\bring-like object\b|\bwhite ring\b", "ring_like_object", "object"),
    (r"\b(?:gloved )?hand\b", "hand", "object"),
    (r"\b(?:drape|gown)s?\b", "drape", "object"),
    (r"\bsurfaces?\b", "surface_region", "anatomy"),
)

_ACTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        r"\b(?:is |being )?pull(?:s|ed|ing)? and (?:tighten(?:s|ed|ing)?|tension(?:s|ed|ing)?)\b",
        ("pull", "tighten"),
    ),
    (
        r"\b(?:is |being )?loop(?:s|ed|ing)? and (?:tighten(?:s|ed|ing)?|tension(?:s|ed|ing)?)\b",
        ("loop_around", "tighten"),
    ),
    (
        r"\b(?:is |being )?pull(?:s|ed|ing)? and loop(?:s|ed|ing)?\b",
        ("pull", "loop_around"),
    ),
    (r"\bgrasp(?:s|ed|ing)? and pull(?:s|ed|ing)?\b", ("grasp", "pull")),
    (r"\bgrasp(?:s|ed|ing)? and retract(?:s|ed|ing)?\b", ("grasp", "pull")),
    (r"\bhold(?:s|ing)? and pull(?:s|ed|ing)?\b", ("hold", "pull")),
    (r"\bgrasp(?:s|ed|ing)? and hold(?:s|ing)?\b", ("grasp", "hold")),
    (r"\bhold(?:s|ing)? and retract(?:s|ed|ing)?\b", ("hold", "pull")),
    (r"\bpull(?:s|ed|ing)? and retract(?:s|ed|ing)?\b", ("pull",)),
    (r"\bpull(?:s|ed|ing)? and position(?:s|ed|ing)?\b", ("pull", "position")),
    (r"\bhold(?:s|ing)? and position(?:s|ed|ing)?\b", ("hold", "position")),
    (r"\bhold(?:s|ing)? and guid(?:e|es|ed|ing)?\b", ("hold", "guide")),
    (r"\bhold(?:s|ing)? and maneuver(?:s|ed|ing)?\b", ("hold", "manipulate")),
    (r"\bpush(?:es|ed|ing)? and guid(?:e|es|ed|ing)?\b", ("push", "guide")),
    (r"\bpush(?:es|ed|ing)? and pull(?:s|ed|ing)?\b", ("push", "pull")),
    (r"\binsert(?:s|ed|ing)? and guid(?:e|es|ed|ing)?\b", ("insert", "guide")),
    (r"\binsert(?:s|ed|ing)? and pull(?:s|ed|ing)?\b", ("insert", "pull")),
    (
        r"\bcontact(?:s|ed|ing)? and manipulat(?:e|es|ed|ing)\b",
        ("contact", "manipulate"),
    ),
    (r"\bemit(?:s|ted|ting)? [^,]* and contact(?:s|ed|ing)?\b", ("emit", "contact")),
    (r"\bgrasp(?:s|ed|ing)? and adjust(?:s|ed|ing)?\b", ("grasp", "manipulate")),
    (r"\bpierc(?:e|es|ed|ing) and pull(?:s|ed|ing)?\b", ("pierce", "pull")),
    (
        r"\b(?:pass(?:es|ed|ing)?|push(?:es|ed|ing)?|insert(?:s|ed|ing)?)(?: [a-z-]+){0,4} through(?: tissue)?\b",
        ("pass_through",),
    ),
    (r"\bpull(?:s|ed|ing)?(?: [a-z-]+){0,4} through(?: tissue)?\b", ("pass_through",)),
    (r"\bmov(?:e|es|ed|ing) through\b", ("pass_through",)),
    (r"\bpass(?:es|ed|ing)?(?: thread-like material)?\b", ("pass_through",)),
    (r"\b(?:form(?:s|ed|ing)? )?loops? around(?: tissue)?\b", ("loop_around",)),
    (r"\b(?:is |being )?loop(?:s|ed|ing)? around\b", ("loop_around",)),
    (r"\barrang(?:e|es|ed|ing) around\b", ("loop_around",)),
    (r"\bstretch(?:es|ed|ing)?\b|\bheld taut\b", ("tighten",)),
    (r"\btighten(?:s|ed|ing)?\b|\btension(?:s|ed|ing)?\b", ("tighten",)),
    (r"\bappl(?:y|ies|ied|ying) pressure\b", ("press",)),
    (r"\binsert(?:s|ed|ing)?(?: into(?: tissue| opening)?)?\b", ("insert",)),
    (r"\bhold(?:s|ing)?(?: tissue)?\b", ("hold",)),
    (r"\bgrasp(?:s|ed|ing)?\b", ("grasp",)),
    (r"\bpull(?:s|ed|ing)?(?: thread-like material)?\b", ("pull",)),
    (r"\bpress(?:es|ed|ing)?(?: against)?(?: tissue)?\b", ("press",)),
    (r"\bcontact(?:s|ed|ing)?\b", ("contact",)),
    (r"\btouch(?:es|ed|ing)?\b", ("contact",)),
    (r"\bmanipulat(?:e|es|ed|ing)\b", ("manipulate",)),
    (r"\bmov(?:e|es|ed|ing)\b", ("move",)),
    (r"\bretract(?:s|ed|ing)?\b", ("pull",)),
    (r"\bguid(?:e|es|ed|ing)?\b", ("guide",)),
    (
        r"\b(?:emit(?:s|ted|ting)?|expel(?:s|led|ling)?|releas(?:e|es|ed|ing))\b",
        ("emit",),
    ),
    (r"\bdeliver(?:s|ed|ing)?\b", ("deliver",)),
    (r"\bremov(?:e|es|ed|ing)\b", ("remove",)),
    (r"\bsecur(?:e|es|ed|ing)\b", ("attach",)),
    (r"\battach(?:es|ed|ing)?\b", ("attach",)),
    (r"\bplac(?:e|es|ed|ing) (?:on|onto)\b", ("position",)),
    (r"\bposition(?:s|ed|ing)?\b", ("position",)),
    (r"\bpush(?:es|ed|ing)?\b", ("push",)),
    (r"\bpierc(?:e|es|ed|ing)\b", ("pierce",)),
    (r"\bcut(?:s|ting)?\b", ("cut",)),
    (r"\bappl(?:y|ies|ied|ying)\b", ("apply",)),
)

_MERGE_STOP_CONCEPTS = frozenset(
    {
        "clear_fluid",
        "generic_instrument",
        "generic_material",
        "generic_object",
        "generic_structure",
        "red_fluid",
        "surface_region",
        "tissue",
        "tissue_region",
    }
)

_ACTION_TRANSITION_SUPPORT = {
    ("grasp", "pull"): frozenset(
        {
            "grid_like_material",
            "membranous_structure",
            "thread_like_material",
            "tubular_structure",
        }
    ),
    ("guide", "pass_through"): frozenset(
        {"needle_like_instrument", "thread_like_material"}
    ),
    ("hold", "pull"): frozenset(
        {"grid_like_material", "thread_like_material", "tubular_structure"}
    ),
    ("insert", "guide"): frozenset({"needle_like_instrument", "thread_like_material"}),
    ("insert", "pass_through"): frozenset(
        {"needle_like_instrument", "thread_like_material", "tubular_structure"}
    ),
    ("loop_around", "pull"): frozenset({"thread_like_material"}),
    ("loop_around", "tighten"): frozenset({"thread_like_material"}),
    ("pass_through", "loop_around"): frozenset({"thread_like_material"}),
    ("pass_through", "pull"): frozenset(
        {"grid_like_material", "thread_like_material", "tubular_structure"}
    ),
    ("position", "loop_around"): frozenset({"thread_like_material"}),
    ("pull", "loop_around"): frozenset({"thread_like_material"}),
    ("pull", "position"): frozenset({"thread_like_material"}),
    ("pull", "tighten"): frozenset({"thread_like_material"}),
}
_ACTION_TRANSITIONS = frozenset(_ACTION_TRANSITION_SUPPORT)


@dataclass(frozen=True)
class NormalizedMention:
    id: str
    surface: str
    canonical: str
    category: str
    source_field: str
    attributes: dict[str, Any] = field(default_factory=dict)
    binding: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedAction:
    id: str
    predicate: str
    original_action: str
    subject_mention_id: str
    target_mention_id: str
    subject_binding: dict[str, Any] = field(default_factory=dict)
    target_binding: dict[str, Any] = field(default_factory=dict)

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
    merge_details: list[dict[str, Any]] = field(default_factory=list)
    structural_support_score: float = 0.0
    support_mode: str = "unknown"
    support_components: dict[str, float | None] = field(default_factory=dict)
    representative_evidence: list[dict[str, Any]] = field(default_factory=list)
    representative_action_coverage: float = 0.0
    representative_entity_coverage: float = 0.0

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
    """Separate a finite base entity from directly visible attributes."""
    lowered = " ".join(str(surface).lower().split())
    attributes = _extract_attributes(lowered)
    for pattern, canonical, category in _ENTITY_RULES:
        if re.search(pattern, lowered):
            return canonical, category, attributes
    canonical, category = _fallback_entity(lowered, category_hint)
    return canonical, category, attributes


def normalize_action(action: str) -> tuple[str, ...]:
    lowered = " ".join(str(action).lower().split())
    for pattern, predicates in _ACTION_RULES:
        if re.search(pattern, lowered):
            return predicates
    return ("other_action",)


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
    max_representative_clips: int = 3,
) -> list[TemporalEvent]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if max_merged_clips < 1:
        raise ValueError("max_merged_clips must be at least 1")
    if max_representative_clips < 1:
        raise ValueError("max_representative_clips must be at least 1")
    if not clips:
        return []
    groups: list[tuple[list[NormalizedClip], list[float], list[dict[str, Any]]]] = []
    current = [clips[0]]
    scores: list[float] = []
    details: list[dict[str, Any]] = []
    for clip in clips[1:]:
        previous = current[-1]
        score, has_continuity, detail = _clip_continuity(previous, clip)
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
            details.append(detail)
        else:
            groups.append((current, scores, details))
            current = [clip]
            scores = []
            details = []
    groups.append((current, scores, details))

    events = []
    for index, (group, merge_scores, merge_details) in enumerate(groups):
        support_score, support_mode, support_components = _event_support(
            group, merge_scores, merge_details
        )
        representatives, action_coverage, entity_coverage = (
            _select_representative_evidence(
                group,
                merge_details,
                max_representatives=max_representative_clips,
            )
        )
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
                merge_details=merge_details,
                structural_support_score=support_score,
                support_mode=support_mode,
                support_components=support_components,
                representative_evidence=representatives,
                representative_action_coverage=action_coverage,
                representative_entity_coverage=entity_coverage,
            )
        )
    return events


def build_evidence_graph(
    rows: list[dict[str, Any]],
    *,
    frame_paths_by_clip: dict[str, list[str]] | None = None,
    merge_threshold: float = 0.45,
    max_merged_clips: int = 5,
    max_representative_clips: int = 3,
) -> EvidenceGraphArtifacts:
    clips = normalize_description_rows(rows, frame_paths_by_clip=frame_paths_by_clip)
    video_id = clips[0].video_id
    events = merge_temporal_events(
        clips,
        threshold=merge_threshold,
        max_merged_clips=max_merged_clips,
        max_representative_clips=max_representative_clips,
    )
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    concept_evidence: dict[tuple[str, str], dict[str, EvidenceInterval]] = defaultdict(
        dict
    )
    concept_mentions: dict[tuple[str, str], list[str]] = defaultdict(list)
    concept_attributes: dict[tuple[str, str], dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
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
                        "argument_binding": mention.binding,
                    },
                )
            )
            concept_key = (mention.category, mention.canonical)
            concept_id = _concept_node_id(*concept_key)
            concept_evidence[concept_key][clip.clip_id] = evidence
            concept_mentions[concept_key].append(mention.id)
            for attribute_name, values in mention.attributes.items():
                concept_attributes[concept_key][attribute_name].update(values)
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
                        "subject_binding": action.subject_binding,
                        "target_binding": action.target_binding,
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
                        action.id,
                        action.subject_mention_id,
                        "has_subject",
                        [evidence],
                        metadata={"argument_binding": action.subject_binding},
                    ),
                    GraphEdge(
                        action.id,
                        action.target_mention_id,
                        "acts_on",
                        [evidence],
                        metadata={"argument_binding": action.target_binding},
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
                    "attribute_counts": {
                        attribute_name: dict(sorted(counts.items()))
                        for attribute_name, counts in sorted(
                            concept_attributes[(category, canonical)].items()
                        )
                    },
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
                confidence=event.structural_support_score,
                metadata={
                    "supporting_clip_ids": event.supporting_clip_ids,
                    "concepts": event.concepts,
                    "predicates": event.predicates,
                    "merge_scores": event.merge_scores,
                    "merge_details": event.merge_details,
                    "structural_support_score": event.structural_support_score,
                    "support_mode": event.support_mode,
                    "support_components": event.support_components,
                    "support_score_version": EVENT_SUPPORT_VERSION,
                    "representative_evidence": event.representative_evidence,
                    "representative_action_coverage": (
                        event.representative_action_coverage
                    ),
                    "representative_entity_coverage": (
                        event.representative_entity_coverage
                    ),
                    "representative_evidence_version": (
                        REPRESENTATIVE_EVIDENCE_VERSION
                    ),
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
        schema_version=GRAPH_SCHEMA_VERSION,
        metadata={
            "builder_version": BUILDER_VERSION,
            "mention_binding_version": MENTION_BINDING_VERSION,
            "action_vocabulary": sorted(ACTION_VOCABULARY),
            "entity_vocabulary": sorted(ENTITY_VOCABULARY),
            "source_clip_count": len(clips),
            "merge_threshold": merge_threshold,
            "max_merged_clips": max_merged_clips,
            "max_representative_clips": max_representative_clips,
            "event_support_version": EVENT_SUPPORT_VERSION,
            "representative_evidence_version": REPRESENTATIVE_EVIDENCE_VERSION,
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
    sequence = 0

    def add_mention(
        surface: str,
        source_field: str,
        category_hint: str,
        binding: dict[str, Any] | None = None,
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
            binding=binding or {},
        )
        sequence += 1
        mentions.append(mention)
        return mention

    for field_name, category_hint in (
        ("visible_anatomy", "anatomy"),
        ("visible_instruments", "instrument"),
        ("visible_objects", "object"),
    ):
        for surface in observed[field_name]:
            add_mention(str(surface), field_name, category_hint)

    visible_mentions = {mention.id: mention for mention in mentions}
    candidates = [
        (mention.id, mention.surface) for mention in visible_mentions.values()
    ]

    def argument_mention(
        surface: str, role: str
    ) -> tuple[NormalizedMention, dict[str, Any]]:
        selected, binding = bind_mention(surface, candidates)
        if selected is not None:
            return visible_mentions[selected], binding
        _, category, _ = normalize_entity(surface)
        mention = add_mention(surface, role, category, binding)
        return mention, binding

    actions: list[NormalizedAction] = []
    for action_index, action in enumerate(observed["actions"]):
        subject, subject_binding = argument_mention(
            str(action.get("subject", "unidentified object")), "action_subject"
        )
        target, target_binding = argument_mention(
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
                    subject_binding=subject_binding,
                    target_binding=target_binding,
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


def _event_support(
    clips: list[NormalizedClip],
    merge_scores: list[float],
    merge_details: list[dict[str, Any]],
) -> tuple[float, str, dict[str, float | None]]:
    observation_scores = []
    argument_scores = []
    for clip in clips:
        observation_score, argument_score = _clip_specificity(clip)
        observation_scores.append(observation_score)
        argument_scores.append(argument_score)
    observation_specificity = mean(observation_scores)
    argument_specificity = mean(argument_scores)

    if len(clips) == 1:
        score = 0.6 * observation_specificity + 0.4 * argument_specificity
        components: dict[str, float | None] = {
            "mean_transition_score": None,
            "minimum_transition_score": None,
            "entity_continuity": None,
            "role_consistency": None,
            "observation_specificity": round(observation_specificity, 4),
            "action_argument_specificity": round(argument_specificity, 4),
        }
        return round(score, 4), "singleton_evidence", components

    entity_continuity = mean(
        detail["score_components"]["informative_entity"] for detail in merge_details
    )
    role_consistency = mean(
        detail["score_components"]["role"] for detail in merge_details
    )
    mean_transition_score = mean(merge_scores)
    minimum_transition_score = min(merge_scores)
    score = (
        0.3 * mean_transition_score
        + 0.15 * minimum_transition_score
        + 0.2 * entity_continuity
        + 0.15 * role_consistency
        + 0.1 * observation_specificity
        + 0.1 * argument_specificity
    )
    components = {
        "mean_transition_score": round(mean_transition_score, 4),
        "minimum_transition_score": round(minimum_transition_score, 4),
        "entity_continuity": round(entity_continuity, 4),
        "role_consistency": round(role_consistency, 4),
        "observation_specificity": round(observation_specificity, 4),
        "action_argument_specificity": round(argument_specificity, 4),
    }
    return round(score, 4), "merged_event", components


def _clip_specificity(clip: NormalizedClip) -> tuple[float, float]:
    informative_mentions = sum(
        mention.canonical not in _MERGE_STOP_CONCEPTS for mention in clip.mentions
    )
    mention_ratio = informative_mentions / len(clip.mentions) if clip.mentions else 0.0
    recognized_actions = sum(
        action.predicate != "other_action" for action in clip.actions
    )
    action_ratio = recognized_actions / len(clip.actions) if clip.actions else 0.0
    observation_specificity = 0.7 * mention_ratio + 0.3 * action_ratio

    mention_by_id = {mention.id: mention for mention in clip.mentions}
    argument_count = 2 * len(clip.actions)
    informative_arguments = sum(
        mention_by_id[mention_id].canonical not in _MERGE_STOP_CONCEPTS
        and (not binding or binding.get("status") == "resolved")
        for action in clip.actions
        for mention_id, binding in (
            (action.subject_mention_id, action.subject_binding),
            (action.target_mention_id, action.target_binding),
        )
    )
    argument_specificity = (
        informative_arguments / argument_count if argument_count else 0.0
    )
    return observation_specificity, argument_specificity


def _select_representative_evidence(
    clips: list[NormalizedClip],
    merge_details: list[dict[str, Any]],
    *,
    max_representatives: int,
) -> tuple[list[dict[str, Any]], float, float]:
    target_count = min(max_representatives, len(clips))
    event_actions = set().union(*(clip.predicates for clip in clips))
    event_concepts = (
        set().union(*(clip.concepts for clip in clips)) - _MERGE_STOP_CONCEPTS
    )
    action_frequency = Counter(
        predicate for clip in clips for predicate in clip.predicates
    )
    concept_frequency = Counter(
        concept for clip in clips for concept in clip.concepts - _MERGE_STOP_CONCEPTS
    )
    transition_clip_ids = {
        clip_id
        for detail in merge_details
        if detail["action_relation"] == "transition"
        for clip_id in (detail["from_clip_id"], detail["to_clip_id"])
    }
    terminal_actions = {"attach", "cut", "remove", "tighten"}
    terminal_clips = [clip for clip in clips if clip.predicates & terminal_actions]
    selected: list[tuple[NormalizedClip, float, str]] = []
    covered_actions: set[str] = set()
    covered_concepts: set[str] = set()

    def utility(clip: NormalizedClip) -> float:
        new_actions = clip.predicates - covered_actions
        new_concepts = (clip.concepts - _MERGE_STOP_CONCEPTS) - covered_concepts
        action_gain = sum(1.0 / action_frequency[item] for item in new_actions)
        concept_gain = sum(1.0 / concept_frequency[item] for item in new_concepts)
        action_denominator = sum(1.0 / action_frequency[item] for item in event_actions)
        concept_denominator = sum(
            1.0 / concept_frequency[item] for item in event_concepts
        )
        action_gain = action_gain / action_denominator if action_denominator else 0.0
        concept_gain = (
            concept_gain / concept_denominator if concept_denominator else 0.0
        )
        observation_score, argument_score = _clip_specificity(clip)
        specificity = 0.6 * observation_score + 0.4 * argument_score
        return 0.55 * action_gain + 0.25 * concept_gain + 0.2 * specificity

    def choose(candidates: list[NormalizedClip], reason: str) -> None:
        available = [
            clip
            for clip in candidates
            if all(
                clip.clip_id != selected_clip.clip_id
                for selected_clip, _, _ in selected
            )
        ]
        if not available or len(selected) >= target_count:
            return
        clip = max(available, key=lambda item: (utility(item), -item.clip_index))
        score = utility(clip)
        selected.append((clip, score, reason))
        covered_actions.update(clip.predicates)
        covered_concepts.update(clip.concepts - _MERGE_STOP_CONCEPTS)

    choose(clips, "primary_event_coverage")
    choose(terminal_clips, "terminal_action_coverage")
    choose(
        [clip for clip in clips if clip.clip_id in transition_clip_ids],
        "action_transition_coverage",
    )
    while len(selected) < target_count:
        choose(clips, "marginal_event_coverage")

    representatives = []
    for selection_index, (clip, score, reason) in enumerate(selected):
        if selection_index == 0:
            role = "primary"
        elif clip.predicates & terminal_actions:
            role = "terminal"
        elif clip.clip_id in transition_clip_ids:
            role = "transition"
        else:
            role = "supporting"
        reasons = [reason]
        if clip.predicates & terminal_actions and reason != "terminal_action_coverage":
            reasons.append("covers_terminal_action")
        if (
            clip.clip_id in transition_clip_ids
            and reason != "action_transition_coverage"
        ):
            reasons.append("covers_action_transition")
        representatives.append(
            {
                "clip_id": clip.clip_id,
                "clip_index": clip.clip_index,
                "start_seconds": clip.start_seconds,
                "end_seconds": clip.end_seconds,
                "role": role,
                "selection_score": round(score, 4),
                "covered_actions": sorted(clip.predicates),
                "covered_informative_concepts": sorted(
                    clip.concepts - _MERGE_STOP_CONCEPTS
                ),
                "reasons": reasons,
            }
        )

    action_coverage = (
        len(covered_actions & event_actions) / len(event_actions)
        if event_actions
        else 1.0
    )
    entity_coverage = (
        len(covered_concepts & event_concepts) / len(event_concepts)
        if event_concepts
        else 1.0
    )
    return representatives, round(action_coverage, 4), round(entity_coverage, 4)


def _clip_continuity(
    first: NormalizedClip, second: NormalizedClip
) -> tuple[float, bool, dict[str, Any]]:
    exact_predicates = sorted(first.predicates & second.predicates)
    compatible_transitions = sorted(
        (left, right)
        for left in first.predicates
        for right in second.predicates
        if (left, right) in _ACTION_TRANSITIONS
    )
    informative_first = first.concepts - _MERGE_STOP_CONCEPTS
    informative_second = second.concepts - _MERGE_STOP_CONCEPTS
    shared_informative = sorted(informative_first & informative_second)
    shared_informative_set = set(shared_informative)
    supported_transitions = [
        transition
        for transition in compatible_transitions
        if shared_informative_set & _ACTION_TRANSITION_SUPPORT[transition]
    ]
    exact_role_matches = _shared_action_roles(
        first, second, {(predicate, predicate) for predicate in exact_predicates}
    )
    transition_role_matches = _shared_action_roles(
        first, second, set(supported_transitions)
    )
    if exact_predicates and exact_role_matches:
        action_score = 1.0
        action_relation = "exact"
        role_matches = exact_role_matches
    elif supported_transitions:
        action_score = 0.8
        action_relation = "transition"
        role_matches = transition_role_matches
    else:
        action_score = 0.0
        action_relation = "none"
        role_matches = []
    informative_score = _jaccard(informative_first, informative_second)
    all_concept_score = _jaccard(first.concepts, second.concepts)
    role_score = min(len(role_matches) / 2.0, 1.0)
    score = (
        0.5 * action_score
        + 0.3 * informative_score
        + 0.1 * all_concept_score
        + 0.1 * role_score
    )
    has_continuity = bool(shared_informative) and action_relation != "none"
    detail = {
        "from_clip_id": first.clip_id,
        "to_clip_id": second.clip_id,
        "action_relation": action_relation,
        "exact_predicates": exact_predicates,
        "compatible_transitions": [list(item) for item in compatible_transitions],
        "supported_transitions": [list(item) for item in supported_transitions],
        "shared_informative_concepts": shared_informative,
        "shared_action_roles": role_matches,
        "score_components": {
            "action": round(action_score, 4),
            "informative_entity": round(informative_score, 4),
            "all_entity": round(all_concept_score, 4),
            "role": round(role_score, 4),
        },
    }
    return score, has_continuity, detail


def _shared_action_roles(
    first: NormalizedClip,
    second: NormalizedClip,
    predicate_pairs: set[tuple[str, str]],
) -> list[dict[str, str]]:
    first_mentions = {mention.id: mention for mention in first.mentions}
    second_mentions = {mention.id: mention for mention in second.mentions}
    matches: set[tuple[str, str, str, str]] = set()
    for first_action in first.actions:
        for second_action in second.actions:
            if (first_action.predicate, second_action.predicate) not in predicate_pairs:
                continue
            for role, first_id, second_id in (
                (
                    "subject",
                    first_action.subject_mention_id,
                    second_action.subject_mention_id,
                ),
                (
                    "target",
                    first_action.target_mention_id,
                    second_action.target_mention_id,
                ),
            ):
                if any(
                    binding and binding.get("status") != "resolved"
                    for binding in (
                        getattr(first_action, f"{role}_binding"),
                        getattr(second_action, f"{role}_binding"),
                    )
                ):
                    continue
                first_canonical = first_mentions[first_id].canonical
                second_canonical = second_mentions[second_id].canonical
                if (
                    first_canonical == second_canonical
                    and first_canonical not in _MERGE_STOP_CONCEPTS
                ):
                    matches.add(
                        (
                            role,
                            first_canonical,
                            first_action.predicate,
                            second_action.predicate,
                        )
                    )
    return [
        {
            "role": role,
            "canonical": canonical,
            "from_predicate": from_predicate,
            "to_predicate": to_predicate,
        }
        for role, canonical, from_predicate, to_predicate in sorted(matches)
    ]


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
        if mention.binding:
            continue
        previous_mentions[(mention.category, mention.canonical)].append(mention)
    for mention in current.mentions:
        if mention.binding:
            continue
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
        if any(
            b and b.get("status") != "resolved"
            for b in (left.subject_binding, left.target_binding)
        ):
            continue
        left_signature = _action_signature(left, mention_by_id)
        for right in current_actions:
            if any(
                b and b.get("status") != "resolved"
                for b in (right.subject_binding, right.target_binding)
            ):
                continue
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
    attributes = Counter(
        (attribute_name, value)
        for clip in clips
        for mention in clip.mentions
        for attribute_name, values in mention.attributes.items()
        for value in values
    )
    merge_relations = Counter(
        detail["action_relation"] for event in events for detail in event.merge_details
    )
    merged_support_scores = [
        event.structural_support_score
        for event in events
        if event.support_mode == "merged_event"
    ]
    singleton_support_scores = [
        event.structural_support_score
        for event in events
        if event.support_mode == "singleton_evidence"
    ]
    representative_counts = [len(event.representative_evidence) for event in events]
    representative_action_coverages = [
        event.representative_action_coverage for event in events
    ]
    representative_entity_coverages = [
        event.representative_entity_coverage for event in events
    ]
    return {
        "builder_version": BUILDER_VERSION,
        "video_id": graph.video_id,
        "schema_version": graph.schema_version,
        "mention_binding_version": MENTION_BINDING_VERSION,
        "argument_binding_status_counts": dict(
            Counter(
                binding["status"]
                for clip in clips
                for action in clip.actions
                for binding in (action.subject_binding, action.target_binding)
            )
        ),
        "argument_binding_method_counts": dict(
            Counter(
                binding["method"]
                for clip in clips
                for action in clip.actions
                for binding in (action.subject_binding, action.target_binding)
            )
        ),
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
        "action_vocabulary": sorted(ACTION_VOCABULARY),
        "entity_vocabulary": sorted(ENTITY_VOCABULARY),
        "other_action_count": actions["other_action"],
        "action_transition_merge_count": merge_relations["transition"],
        "exact_action_merge_count": merge_relations["exact"],
        "event_support_version": EVENT_SUPPORT_VERSION,
        "event_support_score_summary": _score_summary(
            [event.structural_support_score for event in events]
        ),
        "merged_event_support_score_summary": _score_summary(merged_support_scores),
        "singleton_support_score_summary": _score_summary(singleton_support_scores),
        "representative_evidence_version": REPRESENTATIVE_EVIDENCE_VERSION,
        "representative_clip_count_summary": _score_summary(representative_counts),
        "representative_action_coverage_summary": _score_summary(
            representative_action_coverages
        ),
        "representative_entity_coverage_summary": _score_summary(
            representative_entity_coverages
        ),
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
        "top_attributes": [
            {"attribute": name, "value": value, "mentions": count}
            for (name, value), count in attributes.most_common(20)
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


def _extract_attributes(surface: str) -> dict[str, list[str]]:
    attributes: dict[str, list[str]] = {}
    for attribute_name, rules in _ATTRIBUTE_RULES.items():
        values = sorted(
            {normalized for pattern, normalized in rules if re.search(pattern, surface)}
        )
        if values:
            attributes[attribute_name] = values
    return attributes


def _score_summary(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "median": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "minimum": round(float(min(values)), 4),
        "mean": round(float(mean(values)), 4),
        "median": round(float(median(values)), 4),
        "maximum": round(float(max(values)), 4),
    }


def _fallback_entity(surface: str, category_hint: str | None) -> tuple[str, str]:
    if re.search(r"\b(?:material|substance|strands?)\b", surface):
        return "generic_material", "material"
    if re.search(r"\b(?:surface|lining|wall|edge)\b", surface):
        return "surface_region", category_hint or "anatomy"
    if re.search(r"\b(?:structure|mass)\b", surface):
        return "generic_structure", category_hint or "object"
    if re.search(r"\b(?:object|device)\b", surface):
        return "generic_object", category_hint or "object"
    if category_hint == "instrument":
        return "generic_instrument", "instrument"
    if category_hint == "anatomy":
        return "tissue_region", "anatomy"
    if category_hint == "material":
        return "generic_material", "material"
    return "generic_object", category_hint or "object"


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
