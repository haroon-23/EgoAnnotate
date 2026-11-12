#!/usr/bin/env python3
"""Gate 11: Segment caption provenance (`caption_source` in {vlm, template}) and minimum segment duration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 11: Segment Captions & Duration Hygiene")
    parser.add_argument("episode_dir", type=str)
    parser.add_argument("--max-template-fraction", type=float, default=1.0, help="Max allowed template caption fraction")
    parser.add_argument("--min-segment-sec", type=float, default=0.50, help="Minimum segment duration in seconds")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print("=== G11: SEGMENT CAPTIONS PROVENANCE & DURATION GATE ===")
    ep_dir = Path(args.episode_dir)
    ann_path = ep_dir / "annotations.json"
    if not ann_path.exists():
        print(f"  [FAIL] Missing {ann_path}")
        return 1

    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    segments = ann.get("segments", [])
    failures: List[str] = []

    if not segments:
        failures.append("No action segments present in annotations.json")
        vlm_cnt = tpl_cnt = missing_src_cnt = short_cnt = 0
        min_dur = 0.0
        tpl_frac = 1.0
    else:
        vlm_cnt = 0
        tpl_cnt = 0
        missing_src_cnt = 0
        short_cnt = 0
        durations = []

        for idx, s in enumerate(segments):
            dur = float(s.get("end_time", 0.0)) - float(s.get("start_time", 0.0))
            durations.append(dur)
            if dur < args.min_segment_sec - 1e-4:
                short_cnt += 1
                failures.append(
                    f"Segment {idx} ({s.get('name')!r}, {s.get('start_time')}..{s.get('end_time')}s) "
                    f"duration {dur:.3f}s < min {args.min_segment_sec:.3f}s"
                )
            src = s.get("caption_source")
            if src == "vlm":
                vlm_cnt += 1
            elif src == "template":
                tpl_cnt += 1
            else:
                missing_src_cnt += 1

        if missing_src_cnt > 0:
            failures.append(
                f"{missing_src_cnt}/{len(segments)} segments missing explicit `caption_source` in {{'vlm', 'template'}}"
            )

        tpl_frac = tpl_cnt / max(len(segments), 1)
        min_dur = min(durations) if durations else 0.0
        print(
            f"  Segments={len(segments)}, vlm={vlm_cnt}, template={tpl_cnt} ({tpl_frac*100:.1f}%), "
            f"missing_source={missing_src_cnt}, min_duration={min_dur:.3f}s"
        )
        if tpl_frac > args.max_template_fraction + 1e-6:
            failures.append(
                f"Template caption fraction {tpl_frac:.3f} exceeds max_template_fraction {args.max_template_fraction:.3f}"
            )

    passed = len(failures) == 0
    for f_msg in failures:
        print(f"  [FAIL] {f_msg}")
    print(f"G11 VERDICT: {'PASS' if passed else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gate": "g11_captions",
            "status": "PASS" if passed else "FAIL",
            "total_segments": len(segments),
            "vlm_segments": vlm_cnt,
            "template_segments": tpl_cnt,
            "template_fraction": round(tpl_frac, 4),
            "min_duration_sec": round(min_dur, 4),
            "failures": failures,
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
