"""Perception privacy interface forwarding to src.privacy."""
from src.privacy import (
    REQUIRED_DPDP_CONSENT_FIELDS,
    FacePrivacyTracker,
    FaceTrack,
    blur_faces_in_frame,
    blur_video_mp4,
    strip_audio_from_mp4,
    validate_real_consent_manifest,
)

# Backwards compatibility alias
ensure_and_validate_consent_manifest = validate_real_consent_manifest

__all__ = [
    "REQUIRED_DPDP_CONSENT_FIELDS",
    "FacePrivacyTracker",
    "FaceTrack",
    "blur_faces_in_frame",
    "blur_video_mp4",
    "strip_audio_from_mp4",
    "validate_real_consent_manifest",
    "ensure_and_validate_consent_manifest",
]
