"""Conservative, clip-local reference resolution; never physical-ID tracking."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

MENTION_BINDING_VERSION = "observation-mention-binding-v3"

# Only reference-level lexical aliases: do not change the graph concept ontology.
_HEADS = (
    (r"needle[- ](?:holder|driver)", "needle_holder"),
    (r"grasp(?:er|ing (?:tool|instrument))s?", "grasper"),
    (r"forceps", "forceps"),
    (r"clamps?", "clamp"),
    (r"retractors?", "retractor"),
    (r"scissors?", "scissors"),
    (r"(?:suction|aspiration) (?:instrument|tool|tube)", "suction"),
    (r"probes?", "probe"),
    (r"needle(?:-like)?(?: instruments?)?", "needle"),
    (r"(?:instrument|tool|device)s?", "instrument"),
    (r"(?:thread|suture|strand)s?(?:-like)?(?: materials?)?", "thread"),
    (r"(?:blood )?vessels?", "vessel"),
    (r"tissue(?: layers?)?", "tissue"),
    (r"(?:membrane|membranous layer)s?", "membrane"),
    (r"structures?", "structure"),
    (r"organs?", "organ"),
    (r"fluids?", "fluid"),
    (r"(?:tube|tubing)s?", "tube"),
    (r"materials?", "material"),
    (r"objects?", "object"),
    (r"openings?", "opening"),
    (r"cavit(?:y|ies)", "cavity"),
    (r"surfaces?", "surface"),
    (r"regions?", "region"),
    (r"clips?(?:-like)?", "clip"),
    (r"patch(?:es)?", "patch"),
    (r"mesh(?:-like)?", "mesh"),
    (r"rings?(?:-like)?", "ring"),
)
_INSTRUMENT_HEADS = {
    "needle_holder",
    "grasper",
    "forceps",
    "clamp",
    "retractor",
    "scissors",
    "suction",
    "probe",
    "needle",
    "instrument",
}
_RELATION = re.compile(
    r"\b(?:holding|manipulating|grasping(?! (?:tool|instrument))|pulling|"
    r"emitting|delivering|contacting|extending|emerging|near|around|into|"
    r"through|against|onto|from)\b"
)
_ALIASES = {
    "metallic": "metal",
    "reddish": "red",
    "pinkish": "pink",
    "whitish": "white",
    "bluish": "blue",
    "yellowish": "yellow",
    "grey": "gray",
    "cylindrical": "tubular",
    "tube": "tubular",
    "rounded": "round",
    "gripping": "grasping",
    "tipped": "tip",
    "jaws": "jaw",
    "tips": "tip",
    "ends": "end",
}
_STOP = {
    "a",
    "an",
    "the",
    "with",
    "of",
    "and",
    "or",
    "in",
    "on",
    "at",
    "like",
    "visible",
    "appearing",
    "shaped",
    "tip",
    "jaw",
    "end",
    "text",
    "labeled",
    "labelled",
    "marked",
    "engraved",
    "holding",
}


def surface_key(surface: str) -> str:
    """Case/spacing/punctuation normalization, without semantic rewriting."""
    return " ".join(re.findall(r"\w+", surface.casefold()))


def _features(surface: str) -> tuple[str | None, set[str]]:
    phrase = _RELATION.split(surface.casefold(), maxsplit=1)[0]
    # Do not treat a negated descriptor as positive matching evidence. Exact
    # reference matching still works, but this small parser abstains otherwise.
    if re.search(r"\b(?:not|no|without|or)\b|\bnon[- ]", phrase):
        return None, set()
    matches = [
        (m.start(), -len(m.group()), head, m)
        for pattern, head in _HEADS
        for m in re.finditer(r"\b(?:" + pattern + r")\b", phrase)
    ]
    if not matches:
        return None, set()
    _, _, head, match = min(matches, key=lambda item: item[:3])
    qualifiers = phrase[: match.start()] + " " + phrase[match.end() :]
    tokens = {_ALIASES.get(t, t) for t in re.findall(r"\w+", qualifiers)}
    return head, tokens - _STOP


def bind_mention(
    surface: str, candidates: Sequence[tuple[str, str]]
) -> tuple[str | None, dict[str, Any]]:
    """Resolve only a unique textual referent among the original visible mentions.

    All query qualifiers must be supported. Ties and missing qualifiers abstain;
    input order is never a tiebreaker. Created argument mentions are not candidates.
    The caller represents an abstention with a fresh, independent argument node.
    """
    exact = sorted(
        cid for cid, text in candidates if surface_key(text) == surface_key(surface)
    )
    head, qualifiers = _features(surface)
    info: dict[str, Any] = {
        "version": MENTION_BINDING_VERSION,
        "surface": surface,
        "head": head,
        "qualifiers": sorted(qualifiers),
        "status": "unmatched",
        "method": "independent_argument",
        "candidate_mention_ids": [],
        "compatible_mention_ids": [],
        "selected_mention_id": None,
        "physical_identity_confirmed": False,
    }
    if exact:
        info["candidate_mention_ids"] = exact
        info["compatible_mention_ids"] = exact
        if len(exact) == 1:
            info.update(
                status="resolved", method="exact_surface", selected_mention_id=exact[0]
            )
            return exact[0], info
        info.update(status="ambiguous", reason="duplicate_exact_mentions")
        return None, info

    # An unsupported specific head never binds to a generic instrument just
    # because canonical normalization put them in the same bucket.
    compatible = []
    for cid, text in candidates:
        other_head, other_qualifiers = _features(text)
        same_head = head is not None and (
            head == other_head
            or (head == "instrument" and other_head in _INSTRUMENT_HEADS)
        )
        if not same_head:
            continue
        info["candidate_mention_ids"].append(cid)
        if qualifiers <= other_qualifiers:
            compatible.append(cid)
    info["candidate_mention_ids"].sort()
    info["compatible_mention_ids"] = sorted(compatible)
    if len(compatible) == 1:
        info.update(
            status="resolved",
            method="head_attributes" if qualifiers else "unique_head",
            selected_mention_id=compatible[0],
        )
        return compatible[0], info
    info.update(
        status="ambiguous" if compatible else "unmatched",
        reason="multiple_compatible_mentions"
        if compatible
        else "no_supported_reference",
    )
    return None, info
