import pytest
from src.contact_detector import (ContactStateMachine, ObjectIdentityTracker,
                                  fingertip_evidence, iou)
from src.grasp_classifier import TemporalGraspVoter
from src.segment_labeler import build_instruction
from src.retargeting import gripper_mapper as gm

def test_contact_requires_dwell():
    sm = ContactStateMachine(on_frames=3, off_frames=3)
    assert [sm.update(True) for _ in range(2)] == [False, False]
    assert sm.update(True) is True
    assert [sm.update(False) for _ in range(2)] == [True, True]
    assert sm.update(False) is False

def test_untouched_object_never_eligible():
    from src.contact_detector import eligible_objects
    assert "mouse" not in eligible_objects({"mouse": 1, "case": 40})

def test_identity_tracker_canonical_name():
    t = ObjectIdentityTracker()
    r1 = t.update([{"bbox": [0, 0, 10, 10], "name": "earphone case"}])
    r2 = t.update([{"bbox": [1, 1, 11, 11], "name": "white airpods case"}])
    assert r1[0]["id"] == r2[0]["id"]

def test_gripper_opening_doubles():
    assert gm.opening_from_finger_joint(0.04) == pytest.approx(0.08)
    assert gm.finger_joint_from_opening(0.08) == pytest.approx(0.04)

def test_voter_kills_single_frame_unknown():
    v = TemporalGraspVoter()
    lab = ["power_wrap", "unknown", "power_wrap", "power_wrap", "power_wrap"]
    out = v.smooth(lab, [True]*5)
    assert out[1][0] == "power_wrap"

def test_labeler_grammar():
    assert build_instruction("idle", "white airpods case", ["right"]) == "idle (no object)"
    assert build_instruction("contact", "unknown", ["left"]) == \
        "contact an unidentified object with left hand"
    assert build_instruction("grasp", "mug", ["left", "right"]) == \
        "grasp mug with both hands"

def test_fingertip_evidence_needs_two_tips():
    kps = [(0.5, 0.5, 0)]*21
    assert fingertip_evidence(kps, [0.4, 0.4, 0.6, 0.6]) is True
    kps2 = [(0.5, 0.5, 0)]*4 + [(0.1, 0.1, 0)]*17
    assert fingertip_evidence(kps2, [0.4, 0.4, 0.6, 0.6]) is False
