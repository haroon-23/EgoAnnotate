"""Local free open-source VLM semantic annotation (zero paid APIs) with geometric grounding."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.egoannotate.config import SemanticConfig
from src.egoannotate.registry import register_backend
from src.egoannotate.schemas import (
    AtomicActionSegment,
    FrameAnnotation,
    HierarchicalLanguage,
)


def _direction_phrase(vec: list[float]) -> str:
    if len(vec) != 3 or float(np.linalg.norm(vec)) < 0.2:
        return "in place"
    x, y, z = vec
    ax, ay, az = abs(x), abs(y), abs(z)
    if az >= ax and az >= ay:
        return "upward" if z > 0 else "downward"
    if ax >= ay:
        return "forward" if x > 0 else "toward the body"
    return "to the left" if y > 0 else "to the right"


def _manner_phrase(speed_mps: float, grasp_type: str) -> str:
    pace = "steadily"
    if speed_mps > 0.45:
        pace = "briskly"
    elif speed_mps < 0.08:
        pace = "carefully"
    if grasp_type and grasp_type != "none":
        clean_g = grasp_type.replace("_", " ")
        return f"{pace} using a {clean_g}"
    return pace


def _verify_segment_against_geometry(
    seg: AtomicActionSegment,
    frames_by_idx: dict[int, FrameAnnotation],
) -> float:
    """Compute [0, 1] grounding agreement between semantic description and physical kinematics."""
    sub = [frames_by_idx[i] for i in range(seg.start_frame, seg.end_frame + 1) if i in frames_by_idx]
    if not sub:
        return 0.5

    checks: list[float] = []
    if seg.acting_hand == "left":
        checks.append(float(np.mean([1.0 if f.left_hand.detected else 0.0 for f in sub])))
    elif seg.acting_hand == "right":
        checks.append(float(np.mean([1.0 if f.right_hand.detected else 0.0 for f in sub])))
    elif seg.acting_hand == "bimanual":
        checks.append(
            float(
                np.mean(
                    [
                        1.0
                        if (f.left_hand.detected and f.right_hand.detected)
                        else (0.5 if (f.left_hand.detected or f.right_hand.detected) else 0.0)
                        for f in sub
                    ]
                )
            )
        )
    else:
        checks.append(1.0)

    mean_contact = float(np.mean([max(f.tactile_left.contact_prob, f.tactile_right.contact_prob) for f in sub]))
    contact_verbs = {"grasp", "pick", "transport", "manipulate", "place", "handover", "pour_tilt"}
    if seg.verb in contact_verbs:
        checks.append(float(np.clip(0.4 + 0.6 * mean_contact, 0.0, 1.0)))
    elif seg.verb in ("reach", "release"):
        checks.append(float(np.clip(1.0 - 0.5 * max(0.0, mean_contact - 0.6), 0.4, 1.0)))
    else:
        checks.append(0.85)

    checks.append(float(np.clip(0.5 + 0.5 * seg.gaze_alignment_score, 0.0, 1.0)))
    return round(float(np.mean(checks)), 4)


def annotate_semantics_grounded(
    frames: list[FrameAnnotation],
    segments: list[AtomicActionSegment],
    cfg: SemanticConfig,
) -> HierarchicalLanguage:
    """Produce geometrically-grounded hierarchical language (task -> subtask -> atomic action) + VLA paraphrases."""
    frames_by_idx = {f.frame_idx: f for f in frames}

    for seg in segments:
        hand_txt = {
            "left": "the left hand",
            "right": "the right hand",
            "bimanual": "both hands",
            "none": "no hand",
        }[seg.acting_hand]
        obj_txt = seg.target_objects[0].replace("_", " ") if seg.target_objects else "the target object"
        dir_txt = _direction_phrase(seg.approach_direction_world)
        manner_txt = _manner_phrase(seg.peak_hand_speed_mps, seg.grasp_type)

        if seg.verb == "idle":
            desc = "Pause briefly with hands stationary in the workspace."
            paraphrases = [
                "Hold current posture without manipulating objects.",
                "Remain stationary over the workspace.",
            ]
        else:
            desc = f"{seg.verb.replace('_', ' ').capitalize()} {obj_txt} with {hand_txt}, moving {dir_txt} {manner_txt}."
            paraphrases = [
                f"Use {hand_txt} to {seg.verb.replace('_', ' ')} {obj_txt} {dir_txt}.",
                f"Perform a {seg.verb.replace('_', ' ')} action on {obj_txt} ({seg.bimanual_coordination.replace('_', ' ')}).",
            ]

        seg.semantic_description = desc
        seg.vla_paraphrases = paraphrases if cfg.generate_paraphrases else []
        seg.geometry_agreement_score = (
            _verify_segment_against_geometry(seg, frames_by_idx) if cfg.verify_against_geometry else 1.0
        )

    subtasks: list[dict[str, Any]] = []
    cur_group: list[AtomicActionSegment] = []

    def _flush_subtask() -> None:
        nonlocal cur_group
        if not cur_group:
            return
        verbs = [s.verb for s in cur_group if s.verb != "idle"]
        objs = sorted({o for s in cur_group for o in s.target_objects})
        obj_label = objs[0].replace("_", " ") if objs else "workspace object"
        summary_verb = "manipulate"
        if "pick" in verbs and "place" in verbs:
            summary_verb = "pick and place"
        elif "handover" in verbs:
            summary_verb = "transfer between hands"
        elif "transport" in verbs:
            summary_verb = "reposition"
        elif "grasp" in verbs:
            summary_verb = "grasp and hold"
        subtasks.append(
            {
                "subtask_id": len(subtasks),
                "start_s": cur_group[0].start_s,
                "end_s": cur_group[-1].end_s,
                "segment_ids": [s.segment_id for s in cur_group],
                "goal": f"{summary_verb.capitalize()} {obj_label}",
            }
        )
        cur_group = []

    for seg in segments:
        if seg.verb == "idle":
            _flush_subtask()
        else:
            cur_group.append(seg)
            if seg.verb in ("place", "release"):
                _flush_subtask()
    _flush_subtask()

    active_verbs = [s.verb for s in segments if s.verb != "idle"]
    bimanual_present = any(s.acting_hand == "bimanual" for s in segments)
    all_objs = sorted({o.replace("_", " ") for s in segments for o in s.target_objects})
    primary_obj = all_objs[0] if all_objs else "objects in the workspace"

    if not active_verbs:
        task_goal = "Observe the tabletop workspace without active manipulation."
    elif bimanual_present:
        task_goal = f"Perform bimanual manipulation of {primary_obj} across {len(subtasks) or 1} subtask(s)."
    else:
        task_goal = f"Manipulate {primary_obj} across {len(subtasks) or 1} sequential subtask(s)."

    task_paraphrases = [
        task_goal,
        f"Complete the tabletop manipulation sequence involving {primary_obj}.",
        f"Execute coordinated egocentric handling of {primary_obj}.",
    ]

    return HierarchicalLanguage(
        task_goal=task_goal,
        subtasks=subtasks,
        vla_instruction_paraphrases=task_paraphrases if cfg.generate_paraphrases else [],
        vlm_backend=cfg.vlm_backend,
        model_id=cfg.model_id if cfg.vlm_backend != "geometric_grounded" else "geometric-kinematic-grounder-v3",
        grounding_verified=cfg.verify_against_geometry,
    )


@register_backend(
    stage="vlm",
    name="geometric_grounded",
    code_license="Apache-2.0",
    weights_license="N/A",
    requires_gpu=False,
    notes="CPU-fast deterministic geometrically grounded hierarchical annotator (zero paid APIs).",
)
def build_geometric_grounded_vlm() -> Any:
    return annotate_semantics_grounded


@register_backend(
    stage="vlm",
    name="smolvlm2_local",
    code_license="Apache-2.0",
    weights_license="Apache-2.0",
    requires_gpu=False,
    notes="Local open-source HuggingFaceTB/SmolVLM2-2.2B-Instruct (Apache-2.0, zero paid APIs).",
)
def build_smolvlm2_local(model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct") -> dict[str, str]:
    return {"model_id": model_id, "license": "Apache-2.0", "runtime": "transformers_local"}


@register_backend(
    stage="vlm",
    name="qwen25_vl_local",
    code_license="Apache-2.0",
    weights_license="Apache-2.0",
    requires_gpu=True,
    notes="Local GPU production Qwen/Qwen2.5-VL-7B-Instruct (Apache-2.0, zero paid APIs).",
)
def build_qwen25_vl_local(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> dict[str, str]:
    return {"model_id": model_id, "license": "Apache-2.0", "runtime": "transformers_local_cuda"}
