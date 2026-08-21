from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .evidence_builder import normalize_action, normalize_entity
from .schemas import (
    EvidenceInterval,
    GraphEdge,
    GraphNode,
    GraphRetrievalResult,
    VideoEvidenceGraph,
)

RETRIEVER_VERSION = "deterministic-event-graph-retriever-v1"

_GENERIC_QUERY_CONCEPTS = frozenset(
    {
        "generic_instrument",
        "generic_material",
        "generic_object",
        "generic_structure",
        "surface_region",
        "tissue_region",
    }
)
_QUERY_ACTION_ALIASES: dict[str, tuple[str, ...]] = {
    "apply": ("application",),
    "attach": ("attachment", "secure", "securing"),
    "cut": ("cutting", "division"),
    "grasp": ("grasping", "grip", "gripping"),
    "guide": ("guidance", "guiding"),
    "insert": ("insertion", "introduce", "introduction"),
    "loop_around": ("looping", "loop around", "loops around"),
    "pass_through": ("passage", "pass through", "passes through", "passing through"),
    "pierce": ("piercing", "puncture", "puncturing"),
    "position": ("placement", "positioning"),
    "press": ("pressure", "compression", "pressing"),
    "pull": ("traction", "retraction", "pulling"),
    "remove": ("removal", "withdrawal"),
    "tighten": ("tightening", "tensioning", "tension"),
}
_STOPWORDS = frozenset(
    {
        "a",
        "after",
        "an",
        "and",
        "are",
        "at",
        "being",
        "before",
        "by",
        "does",
        "during",
        "for",
        "from",
        "happen",
        "happens",
        "how",
        "in",
        "interval",
        "intervals",
        "is",
        "it",
        "like",
        "of",
        "on",
        "or",
        "show",
        "shows",
        "the",
        "then",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
)


@dataclass(frozen=True)
class NormalizedGraphQuery:
    question: str
    predicates: tuple[str, ...]
    concepts: tuple[str, ...]
    attributes: dict[str, tuple[str, ...]]
    lexical_tokens: tuple[str, ...]
    temporal_relation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _EventCandidate:
    event_ids: list[str]
    score: float
    anchor_event_id: str | None = None
    matched_predicates: set[str] = field(default_factory=set)
    matched_concepts: set[str] = field(default_factory=set)
    matched_attributes: set[str] = field(default_factory=set)
    components: dict[str, float] = field(default_factory=dict)
    expansion_reason: str | None = None


def load_evidence_graph(path: str | Path) -> VideoEvidenceGraph:
    """Load and validate a serialized per-video evidence graph."""
    source = Path(path)
    if source.is_dir():
        source = source / "evidence_graph.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    evidence_graph = VideoEvidenceGraph(
        video_id=str(payload["video_id"]),
        nodes=[_load_node(item) for item in payload["nodes"]],
        edges=[_load_edge(item) for item in payload["edges"]],
        schema_version=str(payload.get("schema_version", "medical-video-graph-v1")),
        metadata=dict(payload.get("metadata", {})),
    )
    if not any(node.node_type == "temporal_event" for node in evidence_graph.nodes):
        raise ValueError("Evidence graph contains no temporal_event nodes")
    return evidence_graph


def normalize_graph_query(
    question: str, graph: VideoEvidenceGraph
) -> NormalizedGraphQuery:
    """Map a question to the graph's finite observation vocabulary."""
    cleaned = " ".join(str(question).strip().lower().split())
    if not cleaned:
        raise ValueError("question must not be empty")

    action_labels = {
        node.label
        for node in graph.nodes
        if node.node_type == "concept" and node.metadata.get("category") == "action"
    }
    entity_labels = {
        node.label
        for node in graph.nodes
        if node.node_type == "concept" and node.metadata.get("category") != "action"
    }
    predicates = _extract_query_actions(cleaned, action_labels)
    concepts = _extract_query_concepts(cleaned, entity_labels)
    attributes = {
        name: tuple(values) for name, values in normalize_entity(cleaned)[2].items()
    }
    return NormalizedGraphQuery(
        question=question,
        predicates=tuple(sorted(predicates)),
        concepts=tuple(sorted(concepts)),
        attributes=attributes,
        lexical_tokens=tuple(sorted(_lexical_tokens(cleaned))),
        temporal_relation=_temporal_relation(cleaned),
    )


class DeterministicEventGraphRetriever:
    """Auditable event retrieval without an LLM or embedding dependency."""

    def __init__(
        self,
        *,
        max_hops: int = 2,
        max_evidence_intervals: int = 5,
        max_representatives_per_event: int = 2,
    ) -> None:
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        if max_evidence_intervals < 1:
            raise ValueError("max_evidence_intervals must be at least 1")
        if max_representatives_per_event < 1:
            raise ValueError("max_representatives_per_event must be at least 1")
        self.max_hops = max_hops
        self.max_evidence_intervals = max_evidence_intervals
        self.max_representatives_per_event = max_representatives_per_event

    def retrieve(
        self,
        question_id: str,
        question: str,
        graph: VideoEvidenceGraph,
        top_k: int = 5,
    ) -> GraphRetrievalResult:
        if not question_id:
            raise ValueError("question_id must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query = normalize_graph_query(question, graph)
        events = sorted(
            (node for node in graph.nodes if node.node_type == "temporal_event"),
            key=lambda node: (node.evidence[0].start_seconds, node.id),
        )
        event_by_id = {event.id: event for event in events}
        event_attributes = _event_attributes(graph, events)
        event_text = _event_text_index(graph, events)
        idf = _query_idf(events, query)

        seeds = [
            candidate
            for event in events
            if (
                candidate := _score_event(
                    event,
                    query,
                    idf,
                    event_attributes[event.id],
                    event_text[event.id],
                )
            )
            is not None
        ]
        if not seeds:
            raise ValueError(
                "Question has no lexical, action, entity, or attribute match in the graph"
            )
        seeds.sort(
            key=lambda item: (-item.score, _event_start(event_by_id[item.event_ids[0]]))
        )

        previous_by_id, next_by_id = _event_neighbors(graph, event_by_id)
        expanded = [
            _expand_candidate(
                seed,
                query,
                event_by_id,
                event_attributes,
                event_text,
                idf,
                previous_by_id,
                next_by_id,
                self.max_hops,
            )
            for seed in seeds[: max(top_k * 4, 20)]
        ]
        ranked = _deduplicate_candidates(expanded)
        ranked.sort(
            key=lambda item: (
                -item.score,
                _event_start(event_by_id[item.event_ids[0]]),
                tuple(item.event_ids),
            )
        )
        ranked = ranked[:top_k]

        selected_evidence, selection_metadata = _select_result_evidence(
            ranked,
            query,
            event_by_id,
            graph,
            max_intervals=self.max_evidence_intervals,
            max_per_event=self.max_representatives_per_event,
        )
        node_ids = list(
            dict.fromkeys(event_id for item in ranked for event_id in item.event_ids)
        )
        candidate_payloads = []
        for rank, item in enumerate(ranked, start=1):
            payload = _candidate_payload(item, event_by_id, query)
            payload["rank"] = rank
            candidate_payloads.append(payload)
        return GraphRetrievalResult(
            question_id=question_id,
            video_id=graph.video_id,
            evidence=selected_evidence,
            node_ids=node_ids,
            reasoning_path=candidate_payloads[0]["reasoning_path"],
            score=ranked[0].score,
            metadata={
                "retriever_version": RETRIEVER_VERSION,
                "question": question,
                "normalized_query": query.to_dict(),
                "ranked_event_groups": candidate_payloads,
                "evidence_selection": selection_metadata,
                "max_hops": self.max_hops,
                "top_k": top_k,
            },
        )


def _extract_query_actions(question: str, graph_actions: set[str]) -> set[str]:
    fragments = [
        question,
        *re.split(
            r"[?.,;:]|\b(?:after|before|then|while|followed by|and)\b",
            question,
        ),
    ]
    predicates = {
        predicate
        for fragment in fragments
        for predicate in normalize_action(fragment)
        if predicate != "other_action" and predicate in graph_actions
    }
    for predicate in graph_actions:
        phrases = (
            predicate.replace("_", " "),
            *_QUERY_ACTION_ALIASES.get(predicate, ()),
        )
        if any(_contains_phrase(question, phrase) for phrase in phrases):
            predicates.add(predicate)
    return predicates


def _extract_query_concepts(question: str, graph_entities: set[str]) -> set[str]:
    concepts = {
        concept
        for concept in graph_entities
        if concept not in _GENERIC_QUERY_CONCEPTS
        and _contains_phrase(question, concept.replace("_", " "))
    }
    words = re.findall(r"[a-z0-9-]+", question)
    for size in range(1, min(6, len(words) + 1)):
        for start in range(len(words) - size + 1):
            phrase = " ".join(words[start : start + size])
            canonical = normalize_entity(phrase)[0]
            if canonical in graph_entities and canonical not in _GENERIC_QUERY_CONCEPTS:
                concepts.add(canonical)
    return concepts


def _score_event(
    event: GraphNode,
    query: NormalizedGraphQuery,
    idf: dict[tuple[str, str], float],
    attributes: set[str],
    text_tokens: set[str],
) -> _EventCandidate | None:
    predicates = set(event.metadata.get("predicates", []))
    concepts = set(event.metadata.get("concepts", []))
    matched_predicates = predicates & set(query.predicates)
    matched_concepts = concepts & set(query.concepts)
    query_attributes = {
        value for values in query.attributes.values() for value in values
    }
    matched_attributes = attributes & query_attributes
    lexical_overlap = text_tokens & set(query.lexical_tokens)
    if not (
        matched_predicates or matched_concepts or matched_attributes or lexical_overlap
    ):
        return None

    action_coverage = _weighted_coverage(
        matched_predicates, set(query.predicates), idf, "action"
    )
    concept_coverage = _weighted_coverage(
        matched_concepts, set(query.concepts), idf, "concept"
    )
    attribute_coverage = (
        len(matched_attributes) / len(query_attributes) if query_attributes else 0.0
    )
    lexical_coverage = (
        len(lexical_overlap) / len(set(query.lexical_tokens))
        if query.lexical_tokens
        else 0.0
    )
    components: list[tuple[float, float]] = []
    if query.predicates:
        components.append((action_coverage, 0.50))
    if query.concepts:
        components.append((concept_coverage, 0.35))
    if query_attributes:
        components.append((attribute_coverage, 0.05))
    if query.lexical_tokens:
        components.append((lexical_coverage, 0.10))
    relevance = _weighted_mean(components)
    support = float(event.metadata.get("structural_support_score", event.confidence))
    representative_coverage = (
        float(event.metadata.get("representative_action_coverage", 0.0))
        + float(event.metadata.get("representative_entity_coverage", 0.0))
    ) / 2.0
    joint_bonus = 0.05 if matched_predicates and matched_concepts else 0.0
    score = min(
        1.0,
        0.77 * relevance
        + 0.10 * support
        + 0.08 * representative_coverage
        + joint_bonus,
    )
    return _EventCandidate(
        event_ids=[event.id],
        score=round(score, 6),
        anchor_event_id=event.id,
        matched_predicates=matched_predicates,
        matched_concepts=matched_concepts,
        matched_attributes=matched_attributes,
        components={
            "query_relevance": round(relevance, 6),
            "action_coverage": round(action_coverage, 6),
            "concept_coverage": round(concept_coverage, 6),
            "attribute_coverage": round(attribute_coverage, 6),
            "lexical_coverage": round(lexical_coverage, 6),
            "structural_support": round(support, 6),
            "representative_coverage": round(representative_coverage, 6),
            "joint_match_bonus": joint_bonus,
        },
    )


def _expand_candidate(
    seed: _EventCandidate,
    query: NormalizedGraphQuery,
    event_by_id: dict[str, GraphNode],
    event_attributes: dict[str, set[str]],
    event_text: dict[str, set[str]],
    idf: dict[tuple[str, str], float],
    previous_by_id: dict[str, str],
    next_by_id: dict[str, str],
    max_hops: int,
) -> _EventCandidate:
    selected = list(seed.event_ids)
    expansion_reason = None
    for hop in range(max_hops):
        neighbor_ids: list[tuple[str, str]] = []
        first, last = selected[0], selected[-1]
        if query.temporal_relation != "after" and first in previous_by_id:
            neighbor_ids.append((previous_by_id[first], "temporal_before"))
        if query.temporal_relation != "before" and last in next_by_id:
            neighbor_ids.append((next_by_id[last], "temporal_after"))

        current_terms = _group_matched_terms(
            selected, query, event_by_id, event_attributes
        )
        options: list[tuple[int, float, str, str]] = []
        for event_id, relation in neighbor_ids:
            candidate_terms = _group_matched_terms(
                [*selected, event_id], query, event_by_id, event_attributes
            )
            gain = sum(
                len(candidate_terms[key] - current_terms[key])
                for key in ("predicates", "concepts", "attributes")
            )
            lexical_gain = len(
                (event_text[event_id] & set(query.lexical_tokens))
                - set().union(*(event_text[item] for item in selected))
            )
            options.append((gain, lexical_gain, event_id, relation))
        if not options:
            break
        options.sort(key=lambda item: (-item[0], -item[1], item[2]))
        gain, lexical_gain, event_id, relation = options[0]
        directional_context = hop == 0 and query.temporal_relation in {
            "after",
            "before",
        }
        if gain <= 0 and lexical_gain <= 0 and not directional_context:
            break
        if relation == "temporal_before":
            selected.insert(0, event_id)
        else:
            selected.append(event_id)
        expansion_reason = (
            f"query_{query.temporal_relation}_context"
            if directional_context
            else "additional_query_term_coverage"
        )

    if len(selected) == 1:
        return seed
    group = _score_event_group(
        selected, query, event_by_id, event_attributes, event_text, idf
    )
    group.anchor_event_id = seed.anchor_event_id
    group.expansion_reason = expansion_reason
    return group


def _score_event_group(
    event_ids: list[str],
    query: NormalizedGraphQuery,
    event_by_id: dict[str, GraphNode],
    event_attributes: dict[str, set[str]],
    event_text: dict[str, set[str]],
    idf: dict[tuple[str, str], float],
) -> _EventCandidate:
    predicates = set().union(
        *(set(event_by_id[item].metadata.get("predicates", [])) for item in event_ids)
    )
    concepts = set().union(
        *(set(event_by_id[item].metadata.get("concepts", [])) for item in event_ids)
    )
    attributes = set().union(*(event_attributes[item] for item in event_ids))
    text_tokens = set().union(*(event_text[item] for item in event_ids))
    synthetic = GraphNode(
        id="group:" + "+".join(event_ids),
        video_id=event_by_id[event_ids[0]].video_id,
        node_type="temporal_event",
        label="event group",
        evidence=[
            interval for item in event_ids for interval in event_by_id[item].evidence
        ],
        confidence=sum(event_by_id[item].confidence for item in event_ids)
        / len(event_ids),
        metadata={
            "predicates": sorted(predicates),
            "concepts": sorted(concepts),
            "structural_support_score": sum(
                float(event_by_id[item].metadata.get("structural_support_score", 0.0))
                for item in event_ids
            )
            / len(event_ids),
            "representative_action_coverage": sum(
                float(
                    event_by_id[item].metadata.get(
                        "representative_action_coverage", 0.0
                    )
                )
                for item in event_ids
            )
            / len(event_ids),
            "representative_entity_coverage": sum(
                float(
                    event_by_id[item].metadata.get(
                        "representative_entity_coverage", 0.0
                    )
                )
                for item in event_ids
            )
            / len(event_ids),
        },
    )
    candidate = _score_event(synthetic, query, idf, attributes, text_tokens)
    assert candidate is not None
    candidate.event_ids = event_ids
    singleton_relevance = []
    for event_id in event_ids:
        singleton = _score_event(
            event_by_id[event_id],
            query,
            idf,
            event_attributes[event_id],
            event_text[event_id],
        )
        if singleton is not None:
            singleton_relevance.append(singleton.components["query_relevance"])
    coverage_gain = max(
        0.0,
        candidate.components["query_relevance"] - max(singleton_relevance),
    )
    candidate.components["multi_event_coverage_gain"] = round(coverage_gain, 6)
    candidate.components["expansion_penalty"] = round(0.015 * (len(event_ids) - 1), 6)
    candidate.score = round(
        min(1.0, candidate.score + 0.08 * coverage_gain)
        - candidate.components["expansion_penalty"],
        6,
    )
    return candidate


def _group_matched_terms(
    event_ids: list[str],
    query: NormalizedGraphQuery,
    event_by_id: dict[str, GraphNode],
    event_attributes: dict[str, set[str]],
) -> dict[str, set[str]]:
    return {
        "predicates": set(query.predicates)
        & set().union(
            *(
                set(event_by_id[item].metadata.get("predicates", []))
                for item in event_ids
            )
        ),
        "concepts": set(query.concepts)
        & set().union(
            *(set(event_by_id[item].metadata.get("concepts", [])) for item in event_ids)
        ),
        "attributes": {
            value for values in query.attributes.values() for value in values
        }
        & set().union(*(event_attributes[item] for item in event_ids)),
    }


def _candidate_payload(
    candidate: _EventCandidate,
    event_by_id: dict[str, GraphNode],
    query: NormalizedGraphQuery,
) -> dict[str, Any]:
    events = [event_by_id[item] for item in candidate.event_ids]
    anchor_event_id = candidate.anchor_event_id or candidate.event_ids[0]
    reasoning_path = [
        *(f"query_action:{item}" for item in sorted(candidate.matched_predicates)),
        *(f"query_concept:{item}" for item in sorted(candidate.matched_concepts)),
        anchor_event_id,
    ]
    anchor_index = candidate.event_ids.index(anchor_event_id)
    for event_id in reversed(candidate.event_ids[:anchor_index]):
        reasoning_path.extend(["inverse:temporal_before", event_id])
    for event_id in candidate.event_ids[anchor_index + 1 :]:
        reasoning_path.extend(["temporal_before", event_id])
    return {
        "rank": 0,
        "event_ids": candidate.event_ids,
        "anchor_event_id": anchor_event_id,
        "start_seconds": min(_event_start(item) for item in events),
        "end_seconds": max(_event_end(item) for item in events),
        "score": candidate.score,
        "matched_predicates": sorted(candidate.matched_predicates),
        "matched_concepts": sorted(candidate.matched_concepts),
        "matched_attributes": sorted(candidate.matched_attributes),
        "components": candidate.components,
        "expansion_reason": candidate.expansion_reason,
        "reasoning_path": reasoning_path,
        "query_temporal_relation": query.temporal_relation,
    }


def _select_result_evidence(
    ranked: list[_EventCandidate],
    query: NormalizedGraphQuery,
    event_by_id: dict[str, GraphNode],
    graph: VideoEvidenceGraph,
    *,
    max_intervals: int,
    max_per_event: int,
) -> tuple[list[EvidenceInterval], list[dict[str, Any]]]:
    clips = {
        str(node.metadata.get("clip_id")): node
        for node in graph.nodes
        if node.node_type == "segment"
    }
    selected: list[EvidenceInterval] = []
    metadata: list[dict[str, Any]] = []
    seen_clips: set[str] = set()
    for rank, candidate in enumerate(ranked, start=1):
        for event_id in candidate.event_ids:
            event = event_by_id[event_id]
            representatives = list(event.metadata.get("representative_evidence", []))
            representatives.sort(
                key=lambda item: (
                    -len(set(item.get("covered_actions", [])) & set(query.predicates)),
                    -len(
                        set(item.get("covered_informative_concepts", []))
                        & set(query.concepts)
                    ),
                    -float(item.get("selection_score", 0.0)),
                    int(item.get("clip_index", 0)),
                )
            )
            event_count = 0
            for representative in representatives:
                clip_id = str(representative["clip_id"])
                if clip_id in seen_clips or clip_id not in clips:
                    continue
                clip = clips[clip_id]
                source = clip.evidence[0]
                selected.append(
                    EvidenceInterval(
                        video_id=source.video_id,
                        start_seconds=source.start_seconds,
                        end_seconds=source.end_seconds,
                        frame_paths=list(source.frame_paths),
                        confidence=candidate.score,
                        metadata={
                            "clip_id": clip_id,
                            "event_id": event_id,
                            "candidate_rank": rank,
                            "representative_role": representative.get("role"),
                            "representative_selection_score": representative.get(
                                "selection_score"
                            ),
                        },
                    )
                )
                metadata.append(
                    {
                        "clip_id": clip_id,
                        "event_id": event_id,
                        "candidate_rank": rank,
                        "frame_count": len(source.frame_paths),
                    }
                )
                seen_clips.add(clip_id)
                event_count += 1
                if len(selected) >= max_intervals:
                    return selected, metadata
                if event_count >= max_per_event:
                    break
    return selected, metadata


def _event_attributes(
    graph: VideoEvidenceGraph, events: list[GraphNode]
) -> dict[str, set[str]]:
    attributes_by_clip: dict[str, set[str]] = {}
    for node in graph.nodes:
        if node.node_type != "entity_mention":
            continue
        clip_id = str(node.metadata.get("clip_id", ""))
        attributes_by_clip.setdefault(clip_id, set()).update(
            value
            for values in node.metadata.get("attributes", {}).values()
            for value in values
        )
    return {
        event.id: set().union(
            *(
                attributes_by_clip.get(str(clip_id), set())
                for clip_id in event.metadata.get("supporting_clip_ids", [])
            )
        )
        for event in events
    }


def _event_text_index(
    graph: VideoEvidenceGraph, events: list[GraphNode]
) -> dict[str, set[str]]:
    segment_text = {
        str(node.metadata.get("clip_id")): _lexical_tokens(node.label)
        for node in graph.nodes
        if node.node_type == "segment"
    }
    return {
        event.id: _lexical_tokens(event.label)
        | set().union(
            *(
                segment_text.get(str(clip_id), set())
                for clip_id in event.metadata.get("supporting_clip_ids", [])
            )
        )
        for event in events
    }


def _query_idf(
    events: list[GraphNode], query: NormalizedGraphQuery
) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    total = len(events)
    for kind, terms, key in (
        ("action", query.predicates, "predicates"),
        ("concept", query.concepts, "concepts"),
    ):
        for term in terms:
            frequency = sum(term in event.metadata.get(key, []) for event in events)
            result[(kind, term)] = math.log((total + 1) / (frequency + 1)) + 1.0
    return result


def _weighted_coverage(
    matched: set[str],
    requested: set[str],
    idf: dict[tuple[str, str], float],
    kind: str,
) -> float:
    if not requested:
        return 0.0
    denominator = sum(idf[(kind, item)] for item in requested)
    return sum(idf[(kind, item)] for item in matched) / denominator


def _event_neighbors(
    graph: VideoEvidenceGraph, event_by_id: dict[str, GraphNode]
) -> tuple[dict[str, str], dict[str, str]]:
    previous: dict[str, str] = {}
    following: dict[str, str] = {}
    for edge in graph.edges:
        if (
            edge.relation == "temporal_before"
            and edge.source in event_by_id
            and edge.target in event_by_id
        ):
            following[edge.source] = edge.target
            previous[edge.target] = edge.source
    return previous, following


def _deduplicate_candidates(candidates: list[_EventCandidate]) -> list[_EventCandidate]:
    best: dict[tuple[str, ...], _EventCandidate] = {}
    for candidate in candidates:
        key = tuple(candidate.event_ids)
        if key not in best or candidate.score > best[key].score:
            best[key] = candidate
    return list(best.values())


def _temporal_relation(question: str) -> str | None:
    if re.search(r"\b(?:after|following)\b", question):
        return "after"
    if re.search(r"\b(?:before|prior to|preceding)\b", question):
        return "before"
    if re.search(r"\b(?:then|followed by|sequence|order)\b", question):
        return "sequence"
    return None


def _lexical_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _weighted_mean(items: list[tuple[float, float]]) -> float:
    if not items:
        return 0.0
    return sum(value * weight for value, weight in items) / sum(
        weight for _, weight in items
    )


def _event_start(event: GraphNode) -> float:
    return min(item.start_seconds for item in event.evidence)


def _event_end(event: GraphNode) -> float:
    return max(item.end_seconds for item in event.evidence)


def _load_interval(payload: dict[str, Any]) -> EvidenceInterval:
    return EvidenceInterval(
        video_id=str(payload["video_id"]),
        start_seconds=float(payload["start_seconds"]),
        end_seconds=float(payload["end_seconds"]),
        frame_paths=[str(item) for item in payload.get("frame_paths", [])],
        confidence=float(payload.get("confidence", 1.0)),
        metadata=dict(payload.get("metadata", {})),
    )


def _load_node(payload: dict[str, Any]) -> GraphNode:
    return GraphNode(
        id=str(payload["id"]),
        video_id=str(payload["video_id"]),
        node_type=str(payload["node_type"]),
        label=str(payload["label"]),
        evidence=[_load_interval(item) for item in payload["evidence"]],
        description=str(payload.get("description", "")),
        confidence=float(payload.get("confidence", 1.0)),
        metadata=dict(payload.get("metadata", {})),
    )


def _load_edge(payload: dict[str, Any]) -> GraphEdge:
    return GraphEdge(
        source=str(payload["source"]),
        target=str(payload["target"]),
        relation=str(payload["relation"]),
        evidence=[_load_interval(item) for item in payload.get("evidence", [])],
        confidence=float(payload.get("confidence", 1.0)),
        metadata=dict(payload.get("metadata", {})),
    )
