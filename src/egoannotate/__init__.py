"""EgoAnnotate v3: Commercial-grade egocentric video to robot-ready data & VLA pipeline."""

__version__ = "3.0.0"

from src.egoannotate.config import PipelineConfig, load_config, resolve_profile_path
from src.egoannotate.registry import REGISTRY, CommercialLicenseError, register_backend

__all__ = [
    "REGISTRY",
    "CommercialLicenseError",
    "PipelineConfig",
    "__version__",
    "load_config",
    "register_backend",
    "resolve_profile_path",
]
