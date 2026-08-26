# EgoAnnotate Evaluation Harness

This directory contains the evaluation infrastructure for measuring the accuracy of the EgoAnnotate pipeline's contact detection, grasp classification, and temporal segmentation stages against hand-labeled ground truth.

## Ground Truth CSV Format

Create a CSV file named `ground_truth_<video_id>.csv` (e.g., `ground_truth_demo_001.csv`) with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | string | Video identifier (must match filename stem) |
| `frame_idx` | int | 0-indexed frame number in the sampled video |
| `timestamp_sec` | float | Seconds into video |
| `object_name` | string | Object being interacted with (e.g., "cup", "knife", "screwdriver") |
| `hand` | string | `"left"` or `"right"` |
| `contact_true` | int | `1` if hand is touching the object, `0` otherwise |
| `grasp_type_true` | string | One of: `precision_pinch`, `power_wrap`, `hook`, `open`, `unknown` |
| `notes` | string | Optional: occlusions, ambiguity, or context |

See `ground_truth_template.csv` for an example.

## How to Hand-Label Ground Truth

1. **Extract frames first** (optional but recommended for consistent frame indexing):
   ```bash
   python scripts/run.py --config configs/default.yaml --video data/raw/demo_001.mp4
   ```
   This creates frames in `data/frames/demo_001/` with consistent numbering.

2. **Watch the video frame-by-frame** (using the extracted frames or video player with frame stepping).

3. **For each frame where a hand interacts with an object**, add a row:
   - Identify the object name consistently across frames
   - Label `contact_true=1` when fingertips are visibly touching the object surface
   - Label `contact_true=0` when hand is near but not touching (approach/retreat)
   - Classify grasp type:
     - `precision_pinch`: Thumb + index fingertip contact only
     - `power_wrap`: All fingers wrapped around object
     - `hook`: Fingers hooked over edge/handle, palm not contacting
     - `open`: Hand flat, no finger curl
     - `unknown`: Occluded or ambiguous

4. **Common mistakes to avoid**:
   - Don't label contact during "approach" phase before actual touch
   - Be consistent with object names across frames
   - Note occlusions in the `notes` column (e.g., "thumb hidden by cup")
   - Label every frame in a manipulation sequence, not just keyframes

## Running the Evaluation

```bash
# Single video evaluation
python scripts/eval_pipeline.py data/raw/demo_001.mp4

# With custom config
python scripts/eval_pipeline.py data/raw/demo_001.mp4 --config configs/my_config.yaml

# Custom ground truth directory
python scripts/eval_pipeline.py data/raw/demo_001.mp4 --eval-dir tests/eval

# Custom output directory
python scripts/eval_pipeline.py data/raw/demo_001.mp4 --output-dir eval_results
```

The script will:
1. Load the pipeline with your config
2. Run full annotation on the video
3. Find `tests/eval/ground_truth_demo_001.csv`
4. Compare predictions vs ground truth per frame
5. Print a summary table to stdout
6. Save detailed JSON to `eval_results/demo_001_eval.json`

## Example Output

```
======================================================================
EVALUATION SUMMARY: demo_001
======================================================================
Frames evaluated:     47
Contact Accuracy:     85.11% (40/47)
Contact Precision:    88.89% (TP=32, FP=4)
Contact Recall:       86.49% (TP=32, FN=5)
Grasp Accuracy:       72.34% (34/47)
----------------------------------------------------------------------
Segment Boundary Errors:
  pick_up: start_err=0.12s, end_err=0.33s
  manipulate: start_err=0.45s, end_err=0.21s
  place: start_err=0.08s, end_err=0.15s
======================================================================
```

## JSON Report Structure

The saved JSON contains:
- `video_id`, `video_path`, `timestamp`
- `num_frames_evaluated`
- `contact_accuracy`, `contact_precision`, `contact_recall`, `grasp_accuracy`
- `frame_results`: List of per-frame results with predictions and correctness
- `segment_results`: List of segment boundary errors in seconds
- `notes`: Config info

## Adding More Videos

1. Place video in `data/raw/`
2. Create `tests/eval/ground_truth_<video_id>.csv`
3. Run `python scripts/eval_pipeline.py data/raw/<video_id>.mp4`

## Notes

- The evaluator is robust to missing ground truth files (prints warning, continues)
- Frame indices must match the pipeline's sampled frames (default 30 FPS)
- Grasp evaluation only counts frames where pipeline detected a hand
- Segment matching uses object+hand grouping from ground truth as proxy for segment boundaries