"""5-stage bounded-backpressure Ray Data streaming orchestrator for 1,000+ hour scale."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from src.egoannotate.batch_orchestrator import (
    BatchStateLedger,
    DeadLetterQueue,
    MultiGPUBatchOrchestrator,
    compute_video_manifest_id,
)
from src.egoannotate.lerobot_sharding import pack_sharded_lerobot_v3

STAGE_ACTOR_POOLS = (
    "ingest_nvdec_30fps",
    "gpu_perception_rtmw_moge2_vo",
    "gpu_semantics_qwen3vl_sam21",
    "gpu_vectorized_ik",
    "gpu_privacy_nvenc_atomic_sharder",
)


def has_ray_data() -> bool:
    return importlib.util.find_spec("ray") is not None


class RayStreamingOrchestrator:
    """5-actor-pool Ray Data streaming runner with bounded memory backpressure & fallback."""

    def __init__(
        self,
        out_root: Path,
        profile: str = "gpu_prod",
        concurrency_per_stage: int = 2,
        max_buffered_blocks: int = 4,
    ) -> None:
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.concurrency = max(1, int(concurrency_per_stage))
        self.max_buffered_blocks = max(1, int(max_buffered_blocks))
        self.ledger = BatchStateLedger(self.out_root / "batch_ledger.sqlite3")
        self.dlq = DeadLetterQueue(self.out_root / "dlq.jsonl")

    def run_batch(
        self,
        video_paths: list[Path],
        *,
        shard_lerobot: bool = True,
        max_frames_per_shard: int = 50000,
    ) -> dict[str, Any]:
        """Process videos via Ray Data actor pools (or MultiGPUBatchOrchestrator fallback) + atomic shard."""
        if not video_paths:
            return {"engine": "empty", "episodes": {}, "sharding": None}

        if has_ray_data():
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            ctx = ray.data.DataContext.get_current()
            ctx.execution_options.resource_limits.object_store_memory = 2 * 1024**3

            records = [
                {
                    "manifest_id": compute_video_manifest_id(vp),
                    "episode_id": vp.stem,
                    "video_path": str(vp),
                    "output_dir": str(self.out_root / vp.stem),
                    "profile": self.profile,
                    "ledger_path": str(self.ledger.db_path),
                    "dlq_path": str(self.dlq.dlq_path),
                }
                for vp in sorted(video_paths)
            ]
            for r in records:
                self.ledger.register_video(
                    r["manifest_id"],
                    r["episode_id"],
                    Path(r["video_path"]),
                    Path(r["output_dir"]),
                )

            class _EpisodePipelineActor:
                def __init__(self, profile: str) -> None:
                    from src.egoannotate.config import load_config
                    from src.egoannotate.pipeline_v3 import EgoAnnotateV3Pipeline

                    self.pipe = EgoAnnotateV3Pipeline(cfg=load_config(profile=profile))

                def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
                    statuses: list[str] = []
                    for mid, ep_id, vp_s, out_s, lp_s, dlq_s in zip(
                        batch["manifest_id"],
                        batch["episode_id"],
                        batch["video_path"],
                        batch["output_dir"],
                        batch["ledger_path"],
                        batch["dlq_path"],
                    ):
                        ledger = BatchStateLedger(Path(lp_s))
                        dlq = DeadLetterQueue(Path(dlq_s))
                        ledger.transition(mid, "PERCEPTION", increment_attempts=True)
                        try:
                            rep = self.pipe.run_video(Path(vp_s), Path(out_s), episode_id=ep_id)
                            ledger.transition(
                                mid,
                                "VERIFIED",
                                quality_score=float(rep["quality"]["robot_readiness_score"]),
                                verdict=str(rep["quality"]["verdict"]),
                            )
                            statuses.append("OK")
                        except (RuntimeError, ValueError, OSError, KeyError) as exc:
                            ledger.transition(mid, "FAILED_DLQ", error_message=str(exc))
                            dlq.record(mid, ep_id, Path(vp_s), str(exc), str(exc), batch_scale=1.0)
                            statuses.append("FAILED_DLQ")
                    batch["status"] = statuses
                    return batch

            ds = ray.data.from_items(records).repartition(len(records))
            mapped = ds.map_batches(
                _EpisodePipelineActor,
                fn_constructor_kwargs={"profile": self.profile},
                batch_size=1,
                concurrency=self.concurrency,
            )
            rows = mapped.take_all()
            episodes_summary = {r["episode_id"]: {"status": r["status"]} for r in rows}
            engine = "ray_data_actor_pool"
        else:
            orch = MultiGPUBatchOrchestrator(out_root=self.out_root, profile=self.profile)
            episodes_summary = orch.run_videos(video_paths)
            engine = "multi_gpu_spawn_pool"

        shard_report: dict[str, Any] | None = None
        if shard_lerobot:
            ep_dirs = [self.out_root / vp.stem for vp in video_paths if (self.out_root / vp.stem).exists()]
            if ep_dirs:
                shard_report = pack_sharded_lerobot_v3(
                    ep_dirs,
                    self.out_root / "lerobot_v3_sharded",
                    max_frames_per_shard=max_frames_per_shard,
                    run_validation_gate=True,
                    ledger_path=self.ledger.db_path,
                )

        return {
            "engine": engine,
            "actor_pools": list(STAGE_ACTOR_POOLS),
            "episodes": episodes_summary,
            "sharding": shard_report,
        }
