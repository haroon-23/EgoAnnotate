#!/usr/bin/env python3
"""CLI entry point for the ego_annotate_vla_pipeline.

Allows executing the annotation pipeline on single or multiple egocentric video files
using configuration specifications.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

# Add project root directory to path to allow import of src
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Auto-load GEMINI_API_KEY from .env if not present in environment
if not os.environ.get("GEMINI_API_KEY"):
    for env_path in [project_root / ".env", Path(".env"), Path.home() / "sia_agent" / ".env"]:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        val = line.split("=", 1)[1].strip("'\"")
                        if val:
                            os.environ["GEMINI_API_KEY"] = val
                            break
            if os.environ.get("GEMINI_API_KEY"):
                break

from src.pipeline import EgoAnnotatePipeline

# Apply monkey patch to avoid Discovery API key checks
try:
    import google.generativeai.client as genai_client
    import googleapiclient.discovery
    import urllib.request

    def patched_setup_discovery_api(self, metadata=()):
        api_key = self._client_options.api_key
        discovery_url = "https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta"
        try:
            with urllib.request.urlopen(discovery_url) as response:
                discovery_doc = response.read().decode("utf-8")
        except Exception as e:
            print(f"[Warning] Failed to fetch discovery doc: {e}")
            raise
        self._local.discovery_api = googleapiclient.discovery.build_from_document(
            discovery_doc, developerKey=api_key
        )

    genai_client.FileServiceClient._setup_discovery_api = patched_setup_discovery_api
except Exception as e:
    print(f"Warning: Failed to apply Discovery API patch: {e}")


def run_export_only(export_rlds_val: str, config_path: str) -> None:
    import yaml
    from src.rlds_exporter import export_to_rlds, export_all_episodes
    
    # Load config to find output_dir
    output_dir = "data/output"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            pipe_cfg = config.get("pipeline", {})
            output_dir = pipe_cfg.get("output_dir", config.get("output_dir", "data/output"))
        except Exception as e:
            print(f"Warning: Failed to load config {config_path}, using default output_dir: {e}")
            
    if export_rlds_val == "":
        print("Scanning output directory for all episodes to export to RLDS...")
        export_all_episodes(output_dir)
    else:
        print(f"Exporting episode '{export_rlds_val}' to RLDS...")
        try:
            export_to_rlds(export_rlds_val, output_dir)
            print(f"Successfully exported '{export_rlds_val}' to RLDS HDF5.")
        except Exception as e:
            print(f"Error exporting RLDS for '{export_rlds_val}': {e}", file=sys.stderr)
            sys.exit(1)


def run_lerobot_export_only(export_lerobot_val: str, config_path: str) -> None:
    import yaml
    from src.lerobot_exporter import export_to_lerobot, export_all_lerobot
    
    # Load config to find output_dir
    output_dir = "data/output"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            pipe_cfg = config.get("pipeline", {})
            output_dir = pipe_cfg.get("output_dir", config.get("output_dir", "data/output"))
        except Exception as e:
            print(f"Warning: Failed to load config {config_path}, using default output_dir: {e}")
            
    if export_lerobot_val == "":
        print("Scanning output directory for all episodes to export to LeRobot...")
        export_all_lerobot(output_dir)
    else:
        print(f"Exporting episode '{export_lerobot_val}' to LeRobot...")
        try:
            export_to_lerobot(export_lerobot_val, output_dir)
            print(f"Successfully exported '{export_lerobot_val}' to LeRobot format.")
        except Exception as e:
            print(f"Error exporting LeRobot for '{export_lerobot_val}': {e}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EgoAnnotate VLA Pipeline - CLI tool to sample and annotate egocentric videos."
    )

    parser.add_argument(
        "--config",
        "-c",
        default="configs/default.yaml",
        help="Path to the YAML configuration file (default: configs/default.yaml)",
    )

    parser.add_argument(
        "--video",
        "-v",
        "--input",
        "-i",
        help="Path to a single video file to process.",
    )

    parser.add_argument(
        "--videos",
        "-vs",
        nargs="+",
        help="Multiple video paths or glob patterns (e.g. 'data/raw/*.mp4').",
    )

    parser.add_argument(
        "--export-rlds",
        nargs="?",
        const="",
        help="Export annotated dataset to RLDS HDF5. Optionally specify a single episode_id; otherwise, scans and exports all.",
    )

    parser.add_argument(
        "--export-lerobot",
        nargs="?",
        const="",
        help="Export annotated dataset to LeRobot v2.1 format. Optionally specify a single episode_id; otherwise, scans and exports all.",
    )

    parser.add_argument(
        "--export-overlay",
        action="store_true",
        help="Export burned-in overlay video with all annotations (contact bboxes, segments, grasps).",
    )

    args = parser.parse_args()

    # Collect all video paths
    video_paths = []

    if args.video:
        video_paths.append(args.video)

    if args.videos:
        for pattern in args.videos:
            # Expand glob patterns
            matched = glob.glob(pattern)
            if matched:
                video_paths.extend(matched)
            else:
                # Fallback to direct path check if glob pattern didn't match anything
                if os.path.exists(pattern):
                    video_paths.append(pattern)
                else:
                    print(f"Warning: Video path or pattern '{pattern}' not found.", file=sys.stderr)

    # Deduplicate and sort paths
    video_paths = sorted(list(set(video_paths)))

    # Validate we have at least one video to process
    if not video_paths:
        if args.export_lerobot is not None:
            # Standalone LeRobot export mode
            run_lerobot_export_only(args.export_lerobot, args.config)
            sys.exit(0)
        elif args.export_rlds is not None:
            # Standalone RLDS export mode
            run_export_only(args.export_rlds, args.config)
            sys.exit(0)
        else:
            print("Error: No valid video paths found to process.", file=sys.stderr)
            parser.print_help()
            sys.exit(1)

    print("=" * 60)
    print("EGO ANNOTATE - FOUND VIDEOS TO PROCESS:")
    for path in video_paths:
        print(f"  - {path}")
    print("=" * 60 + "\n")

    # Initialize and run pipeline
    try:
        pipeline = EgoAnnotatePipeline(config_path=args.config)
        successful_episodes = pipeline.process_videos(video_paths)
        
        if successful_episodes:
            print("\n" + "=" * 60)
            print("SUCCESSFULLY EXPORTED VLA DATASET")
            print("=" * 60)
            print("Next steps:")
            print(f"  1. Review exported annotations in output directory: {pipeline.dataset_exporter.config.output_dir}")
            print("  2. Inspect the generated HUD telemetry videos (visualization.mp4) under each episode folder.")
            print("  3. Train your downstream VLA policy using the standardized frame annotations.")
            
            output_dir = pipeline.dataset_exporter.config.output_dir

            # Post-run RLDS export if requested
            if args.export_rlds is not None:
                print("\nRunning post-pipeline RLDS HDF5 export...")
                from src.rlds_exporter import export_to_rlds
                
                if args.export_rlds != "":
                    print(f"  - Exporting specific episode '{args.export_rlds}' to RLDS...")
                    try:
                        export_to_rlds(args.export_rlds, output_dir)
                    except Exception as e:
                        print(f"Error exporting RLDS for '{args.export_rlds}': {e}", file=sys.stderr)
                else:
                    for ep in successful_episodes:
                        print(f"  - Exporting episode '{ep.episode_id}' to RLDS...")
                        try:
                            rlds_path = export_to_rlds(ep.episode_id, output_dir)
                            print(f"    Saved: {rlds_path.name}")
                        except Exception as e:
                            print(f"Error exporting RLDS for '{ep.episode_id}': {e}", file=sys.stderr)

            # Post-run LeRobot export if requested
            if args.export_lerobot is not None:
                print("\nRunning post-pipeline LeRobot format export...")
                from src.lerobot_exporter import export_to_lerobot
                
                if args.export_lerobot != "":
                    print(f"  - Exporting specific episode '{args.export_lerobot}' to LeRobot...")
                    try:
                        export_to_lerobot(args.export_lerobot, output_dir)
                    except Exception as e:
                        print(f"Error exporting LeRobot for '{args.export_lerobot}': {e}", file=sys.stderr)
                else:
                    for ep in successful_episodes:
                        print(f"  - Exporting episode '{ep.episode_id}' to LeRobot...")
                        try:
                            lerobot_path = export_to_lerobot(ep.episode_id, output_dir)
                            print(f"    Saved: {lerobot_path.relative_to(output_dir)}")
                        except Exception as e:
                            print(f"Error exporting LeRobot for '{ep.episode_id}': {e}", file=sys.stderr)

            # Post-run overlay video export if requested
            if args.export_overlay:
                print("\nRunning post-pipeline overlay video export...")
                from src.visualizer import render_annotated_video
                from pathlib import Path
                
                for ep in successful_episodes:
                    print(f"  - Exporting overlay for episode '{ep.episode_id}'...")
                    try:
                        overlay_path = Path(output_dir) / ep.episode_id / "overlay_annotated.mp4"
                        overlay_path.parent.mkdir(parents=True, exist_ok=True)
                        render_annotated_video(ep.video_path, ep, overlay_path)
                        print(f"    Saved: {overlay_path.relative_to(output_dir)}")
                    except Exception as e:
                        print(f"Error exporting overlay for '{ep.episode_id}': {e}", file=sys.stderr)
            
            print("=" * 60 + "\n")
        else:
            print("Processing complete: 0 videos successfully annotated.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Pipeline initialisation or run failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
