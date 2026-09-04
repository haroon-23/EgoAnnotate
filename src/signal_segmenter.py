"""Signal-based temporal segmentation.

Derives candidate segment boundaries from frame-level contact and grasp signals,
then labels them using a VLM. Grounds segmentation in real per-frame detections
rather than guessing from raw video.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .datatypes import CandidateSegment, ContactState, GraspType

logger = logging.getLogger(__name__)


@dataclass
class SignalSegmenterConfig:
    """Configuration for the SignalSegmenter."""
    # Minimum segment duration in seconds (filters jitter)
    min_segment_duration_sec: float = 0.2
    # Maximum gap to merge adjacent same-type segments (seconds)
    merge_gap_sec: float = 0.3
    # Idle threshold: sustained no-contact period to start new segment
    idle_threshold_sec: float = 1.0


class SignalSegmenter:
    """Derives candidate segment boundaries from contact/grasp signal transitions.
    
    Does NOT use VLM — purely deterministic signal processing.
    Produces CandidateSegment objects that are later labeled by SegmentLabeler.
    """
    
    def __init__(self, config: Optional[SignalSegmenterConfig] = None):
        self.config = config or SignalSegmenterConfig()
    
    def get_candidates(
        self,
        left_contact_timeline: List[Optional[ContactState]],
        right_contact_timeline: List[Optional[ContactState]],
        left_grasp_timeline: List[Optional[GraspType]],
        right_grasp_timeline: List[Optional[GraspType]],
        frame_timestamps: List[float],
    ) -> List[CandidateSegment]:
        """Detect segment boundaries from signal timelines.
        
        Args:
            left/right_contact_timeline: Per-frame ContactState (or None).
            left/right_grasp_timeline: Per-frame GraspType (or None).
            frame_timestamps: Timestamp in seconds for each frame.
            
        Returns:
            List of CandidateSegment with boundaries, no VLM labels yet.
        """
        n_frames = len(frame_timestamps)
        if n_frames == 0:
            return []
        
        # --- 1. Determine per-frame dominant hand state ---
        # Use right hand if present, else left
        contact_states = []
        grasp_types = []
        object_names = []
        
        for i in range(n_frames):
            lc = left_contact_timeline[i] if i < len(left_contact_timeline) else None
            rc = right_contact_timeline[i] if i < len(right_contact_timeline) else None
            lg = left_grasp_timeline[i] if i < len(left_grasp_timeline) else None
            rg = right_grasp_timeline[i] if i < len(right_grasp_timeline) else None
            
            # Prefer right hand (typically dominant in egocentric)
            if rc is not None:
                contact = rc
                grasp = rg
                hand = "right"
            elif lc is not None:
                contact = lc
                grasp = lg
                hand = "left"
            else:
                contact = None
                grasp = None
                hand = "none"
            
            # Contact state: "contact" or "no_contact"
            if contact and contact.in_contact:
                contact_state = "contact"
                obj_name = contact.object_name
            else:
                contact_state = "no_contact"
                obj_name = None
            
            # Grasp type
            grasp_type = grasp.type if grasp else "none"
            
            contact_states.append(contact_state)
            grasp_types.append(grasp_type)
            object_names.append(obj_name)
        
        # --- 2. Find transition frames ---
        transition_frames = set()
        transition_types = {}  # frame_idx -> type
        
        for i in range(1, n_frames):
            # Contact on/off transitions
            if contact_states[i] != contact_states[i - 1]:
                transition_frames.add(i)
                if contact_states[i] == "contact":
                    transition_types[i] = "contact_on"
                else:
                    transition_types[i] = "contact_off"
            
            # Grasp type changes (only during contact)
            if grasp_types[i] != grasp_types[i - 1] and contact_states[i] == "contact":
                transition_frames.add(i)
                transition_types[i] = "grasp_change"
        
        # Always include start and end
        transition_frames.add(0)
        transition_frames.add(n_frames)
        transition_types[0] = "start"
        transition_types[n_frames] = "end"
        
        # --- 3. Add idle boundaries (sustained no-contact) ---
        idle_frames = self._find_idle_boundaries(contact_states, frame_timestamps)
        for idx in idle_frames:
            transition_frames.add(idx)
            if idx not in transition_types:
                transition_types[idx] = "idle"
        
        # --- 4. Sort transitions and create initial segments ---
        sorted_frames = sorted(transition_frames)
        
        raw_segments = []
        for i in range(len(sorted_frames) - 1):
            start_f = sorted_frames[i]
            end_f = sorted_frames[i + 1]
            
            start_t = frame_timestamps[start_f]
            end_t = frame_timestamps[min(end_f, n_frames - 1)]
            
            # Extract most frequent non-null object within segment frames
            seg_objs = [o for o in object_names[start_f:end_f] if o is not None]
            if seg_objs:
                from collections import Counter
                seg_obj = Counter(seg_objs).most_common(1)[0][0]
            else:
                seg_obj = None
            
            mid_f = (start_f + end_f) // 2
            if mid_f >= n_frames:
                mid_f = n_frames - 1
            
            trans_type = transition_types.get(end_f, "unknown")
            if trans_type == "idle":
                trans_type = "idle"
            
            raw_segments.append(CandidateSegment(
                start_frame=start_f,
                end_frame=end_f,
                start_time=start_t,
                end_time=end_t,
                transition_type=trans_type,
                contact_state=contact_states[mid_f],
                grasp_type=grasp_types[mid_f],
                object_name=seg_obj,
            ))
        
        # --- 5. Filter jitter (state-aware noise filtering) ---
        filtered = self._filter_jitter(raw_segments, frame_timestamps)
        
        # --- 6. Merge adjacent segments of same type ---
        merged = self._merge_adjacent_same_type(filtered)
        
        # --- 7. Ensure at least one segment & Sanity Check ---
        if not merged:
            merged = [CandidateSegment(
                start_frame=0,
                end_frame=n_frames,
                start_time=frame_timestamps[0],
                end_time=frame_timestamps[-1],
                transition_type="full_video",
                contact_state=contact_states[0],
                grasp_type=grasp_types[0],
                object_name=object_names[0],
            )]

        # Sanity check: Flag if segment count collapsed relative to raw contact transitions
        raw_contact_transitions = len([i for i in range(1, n_frames) if contact_states[i] != contact_states[i - 1]])
        if raw_contact_transitions >= 5 and len(merged) == 1:
            msg = (
                f"[SignalSegmenter] SANITY WARNING: Video has {raw_contact_transitions} contact state transitions "
                f"across {n_frames} frames but segmenter produced ONLY 1 segment. Segmentation collapse detected!"
            )
            logger.warning(msg)
            print(f"⚠ WARNING: {msg}")

        return merged
    
    def _find_idle_boundaries(
        self,
        contact_states: List[str],
        timestamps: List[float],
    ) -> List[int]:
        """Find frames where sustained idle periods begin/end."""
        idle_boundaries = []
        in_idle = False
        idle_start = None
        
        for i, state in enumerate(contact_states):
            if state == "no_contact" and not in_idle:
                # Potential idle start
                idle_start = i
                in_idle = True
            elif state == "contact" and in_idle:
                # Idle ended - check duration
                if idle_start is not None:
                    duration = timestamps[i] - timestamps[idle_start]
                    if duration >= self.config.idle_threshold_sec:
                        idle_boundaries.append(idle_start)
                        idle_boundaries.append(i)
                in_idle = False
                idle_start = None
        
        # Handle trailing idle
        if in_idle and idle_start is not None:
            duration = timestamps[-1] - timestamps[idle_start]
            if duration >= self.config.idle_threshold_sec:
                idle_boundaries.append(idle_start)
                idle_boundaries.append(len(contact_states) - 1)
        
        return idle_boundaries
    
    def _filter_jitter(
        self,
        segments: List[CandidateSegment],
        timestamps: List[float],
    ) -> List[CandidateSegment]:
        """Merge short jitter segments (< min_duration) into adjacent same-type segments."""
        if len(segments) <= 1:
            return segments
        
        filtered = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            duration = seg.end_time - seg.start_time
            
            if duration >= self.config.min_segment_duration_sec:
                filtered.append(seg)
                i += 1
            else:
                # Short segment: attempt to resolve noise without crossing contact boundaries
                if filtered and filtered[-1].contact_state == seg.contact_state:
                    prev = filtered[-1]
                    filtered[-1] = CandidateSegment(
                        start_frame=prev.start_frame,
                        end_frame=seg.end_frame,
                        start_time=prev.start_time,
                        end_time=seg.end_time,
                        transition_type=prev.transition_type,
                        contact_state=prev.contact_state,
                        grasp_type=prev.grasp_type if prev.grasp_type != "unknown" else seg.grasp_type,
                        object_name=prev.object_name or seg.object_name,
                    )
                elif filtered and i + 1 < len(segments) and filtered[-1].contact_state == segments[i + 1].contact_state:
                    # Glitch pulse (A -> B_short -> A): merge pulse into prev
                    prev = filtered[-1]
                    next_seg = segments[i + 1]
                    filtered[-1] = CandidateSegment(
                        start_frame=prev.start_frame,
                        end_frame=next_seg.end_frame,
                        start_time=prev.start_time,
                        end_time=next_seg.end_time,
                        transition_type=prev.transition_type,
                        contact_state=prev.contact_state,
                        grasp_type=prev.grasp_type if prev.grasp_type != "unknown" else next_seg.grasp_type,
                        object_name=prev.object_name or next_seg.object_name,
                    )
                    i += 1  # consumed next_seg
                else:
                    # Genuine state boundary transition — keep to preserve segment structure
                    filtered.append(seg)
                i += 1
        
        return filtered
    
    def _merge_adjacent_same_type(self, segments: List[CandidateSegment]) -> List[CandidateSegment]:
        """Merge adjacent segments with same contact_state, compatible grasp_type, and object."""
        if len(segments) <= 1:
            return segments
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            last = merged[-1]
            
            same_contact = (last.contact_state == seg.contact_state)
            same_grasp = (
                last.grasp_type == seg.grasp_type or 
                last.grasp_type in ("none", "unknown") or 
                seg.grasp_type in ("none", "unknown")
            )
            same_obj = (
                last.object_name == seg.object_name or 
                (last.object_name is None and seg.object_name is None)
            )
            
            gap = seg.start_time - last.end_time
            can_merge = same_contact and same_grasp and same_obj and (gap <= self.config.merge_gap_sec)
            
            if can_merge:
                merged[-1] = CandidateSegment(
                    start_frame=last.start_frame,
                    end_frame=seg.end_frame,
                    start_time=last.start_time,
                    end_time=seg.end_time,
                    transition_type=last.transition_type,
                    contact_state=last.contact_state,
                    grasp_type=seg.grasp_type if seg.grasp_type not in ("none", "unknown") else last.grasp_type,
                    object_name=last.object_name or seg.object_name,
                )
            else:
                merged.append(seg)
        
        return merged