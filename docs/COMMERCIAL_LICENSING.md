# Commercial Licensing & Distribution Guide

EgoAnnotate v3 is architected from the ground up to be safely **rented, leased, licensed per-seat, or deployed on-premise** to robotics companies with **$0 recurring third-party software or API cost** and **zero telemetry**.

## 1. License Firewall Guarantees
- **100% Commercially Permissible Defaults:** Every default model and dependency uses Apache-2.0, MIT, BSD, or zlib (see `LICENSES.md` and `egoannotate audit-licenses`).
- **Zero Paid Cloud APIs:** No OpenAI, Anthropic, or Google Gemini API keys are required or contacted. All VLM, perception, geometry, retargeting, and export stages run strictly locally.
- **Quarantined Research Plugins:** Non-commercial models (`MANO`/`HaMeR`, `UniDepth`, `Depth-Anything-V2-Base/Large/Giant`, `MASt3R`, `VGGT-1B`) are programmatically blocked by `PipelineConfig` and `PluginRegistry` unless an operator explicitly overrides `allow_non_commercial_plugins: true` for non-commercial R&D.
- **Zero Telemetry & Air-Gapped Ready:** Use `python scripts/download_models.py --profile gpu_prod --offline` to pre-verify SHA-256 checksums for completely offline customer VPC / on-premise GPU clusters.

## 2. Recommended Commercial Licensing Models
1. **Per-Node / Per-GPU Annual Lease:** Deliver `Dockerfile.cuda` + signed container image running `egoannotate worker` on customer-owned Linux GPU nodes.
2. **Volume-Metered Data Processing License:** Track processed hours via the SQLite `jobs` audit log (`run_report.json` records `video.duration_s` and `video.video_sha256`).
3. **Enterprise Source-Access License:** Keep core `src/egoannotate/` under a commercial license agreement while shipping `THIRD_PARTY_NOTICES.md` alongside upstream Apache-2.0/MIT/BSD dependencies.
