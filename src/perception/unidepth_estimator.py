"""Back-compat alias: UniDepthEstimator moved to src.perception.depth.

.. note::
   The old ``project_to_3d`` method (hardcoded ~60-degree-FOV pinhole,
   single-pixel depth sample) was **removed** in Phase C. Use
   :func:`src.perception.pnp_pose.backproject_to_3d` (real intrinsics) or
   :func:`src.perception.depth.localize_objects_3d` (mask-median depth)
   instead.
"""

from .depth import (
    UNIDEPTH_DOWNLOAD_URL,
    DepthConfig,
    DepthEstimator,
    DisabledDepth,
    UniDepthEstimator,
    create_depth_estimator,
    localize_objects_3d,
)

__all__ = [
    "UniDepthEstimator",
    "DepthEstimator",
    "DepthConfig",
    "DisabledDepth",
    "create_depth_estimator",
    "localize_objects_3d",
    "UNIDEPTH_DOWNLOAD_URL",
]
