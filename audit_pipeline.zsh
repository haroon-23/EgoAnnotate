#!/bin/zsh
# audit_pipeline.zsh  -  "what pipeline is actually in use?"
# Run from the repo root:   zsh audit_pipeline.zsh [PACK_DIR]      (default PACK_DIR=audit_pack)
# Optional env:  PYTHON=/path/to/python   RUN_TESTS=1 (runs the full pytest suite, ~3+ min)
# READ-ONLY on your code. Writes only into PACK_DIR/pipeline/.
setopt null_glob pipefail

PACK="${1:-audit_pack}"
P="$PACK/pipeline"
mkdir -p "$P/repo_files"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY="$(command -v python3)"; fi
fi
echo "[audit_pipeline] repo=$PWD python=$PY pack=$PACK"

# cap <outfile> <command...>   -> runs command, stores "$ cmd", output and exit code
cap()   { local f="$1"; shift; { print -r -- "\$ $*"; "$@"; print -r -- "[exit=$?]"; } > "$P/$f" 2>&1 }
# capsh <outfile> "<shell string>"  -> same, for pipelines/greps
capsh() { local f="$1"; shift; { print -r -- "\$ $*"; eval "$*"; print -r -- "[exit=$?]"; } > "$P/$f" 2>&1 }

# ---------- 1. machine ----------
{
  date -u +"utc: %Y-%m-%dT%H:%M:%SZ"
  sw_vers 2>/dev/null
  uname -m
  sysctl -n machdep.cpu.brand_string 2>/dev/null
  sysctl -n hw.ncpu 2>/dev/null
  echo "ram_bytes: $(sysctl -n hw.memsize 2>/dev/null)"
  "$PY" --version 2>&1
  command -v ffmpeg ffprobe 2>/dev/null
  ffmpeg -version 2>/dev/null | head -1
} > "$P/00_machine.txt" 2>&1

# ---------- 2. code state (git) ----------
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  capsh 01_git.txt 'git rev-parse HEAD; git branch --show-current; echo; git log --oneline -n 25; echo; echo "--- uncommitted files:"; git status --porcelain | head -80; echo; echo "--- diff stat vs HEAD:"; git diff --stat HEAD | tail -40'
else
  echo "NOT A GIT REPO - code state cannot be fingerprinted by commit. Consider 'git init && git add -A && git commit'." > "$P/01_git.txt"
fi

# ---------- 3. gates integrity (has the agent touched the tests that judge it?) ----------
BASE="$HOME/.ego_audit_baseline"; mkdir -p "$BASE"
if [[ -d gates ]]; then
  {
    echo "--- permissions:"; ls -ld gates; ls -l gates | head -30
    echo; echo "--- sha256 of gates/*.py:"
    find gates -type f -name '*.py' | sort | xargs shasum -a 256
  } > "$P/02_gates_integrity.txt" 2>&1
  find gates -type f -name '*.py' | sort | xargs shasum -a 256 > "$P/gates.sha256.now"
  if [[ -f "$BASE/gates.sha256" ]]; then
    { echo; echo "--- DIFF vs trusted baseline ($BASE/gates.sha256):"; diff "$BASE/gates.sha256" "$P/gates.sha256.now" && echo "IDENTICAL: gates unchanged since baseline"; } >> "$P/02_gates_integrity.txt" 2>&1
  else
    cp "$P/gates.sha256.now" "$BASE/gates.sha256"
    echo "\nBASELINE CREATED NOW from the current gates/. It is only trustworthy if you had not let the agent edit gates/ before this run." >> "$P/02_gates_integrity.txt"
  fi
  rm -f "$P/gates.sha256.now"
else
  echo "gates/ directory MISSING" > "$P/02_gates_integrity.txt"
fi

# ---------- 3b. what changed since the previous audit run? ----------
find src scripts tests configs docs -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.md' \) 2>/dev/null | sort | xargs shasum -a 256 > "$P/src_hashes.txt" 2>/dev/null
if [[ -f "$BASE/src_hashes.txt" ]]; then
  { echo "files changed/added/removed since the previous audit run:"; diff <(awk '{print $2" "$1}' "$BASE/src_hashes.txt") <(awk '{print $2" "$1}' "$P/src_hashes.txt") | grep -E '^[<>]' | awk '{print $1" "$2}' | sort -k2 | uniq; } > "$P/03b_changed_since_last_audit.txt" 2>&1
else
  echo "first run with change tracking: baseline stored; the next run will list changed files." > "$P/03b_changed_since_last_audit.txt"
fi
cp "$P/src_hashes.txt" "$BASE/src_hashes.txt"

# ---------- 4. dependencies + which backends can actually be imported ----------
cap 03_pip_freeze.txt "$PY" -m pip freeze
"$PY" - > "$P/04_backends_and_packages.txt" 2>&1 <<'PYEOF'
import importlib, importlib.metadata as md, json, os, sys
pk = ["numpy","torch","torchvision","opencv-python","opencv-python-headless","mediapipe","transformers","lerobot",
      "pybullet","mujoco","mink","h5py","pyarrow","pandas","tensorflow","tensorflow-datasets","google-genai",
      "google-generativeai","onnxruntime","scipy","pytest","PyYAML"]
print("== package versions ==")
for p in pk:
    try: print(f"{p:28s} {md.version(p)}")
    except Exception: print(f"{p:28s} -")
print("\n== optional perception backends: importable? (importable != verified) ==")
for m in ["wilor","hamer","hawor","lib_hawor","droid_slam","droid","sam2","depth_anything_v2","unidepth","ultralytics",
          "groundingdino","segment_anything","colmap","pycolmap","tensorflow_datasets","onnxruntime"]:
    try: importlib.import_module(m); print(f"{m:22s} IMPORTABLE")
    except Exception as e: print(f"{m:22s} no ({type(e).__name__})")
try:
    import torch; print("\ntorch cuda:", torch.cuda.is_available(), "mps:", torch.backends.mps.is_available())
except Exception as e: print("torch import failed", e)
print("\n== configs/backends.yaml ==")
for cand in ["configs/backends.yaml","configs/default.yaml","configs/quality.yaml"]:
    if os.path.exists(cand):
        print(f"--- {cand}"); print(open(cand).read()[:6000])
    else:
        print(f"--- {cand}  MISSING")
PYEOF

# ---------- 5. cached model weights (what models are really on disk) ----------
capsh 05_model_weights.txt 'ls -1 "$HOME/.cache/huggingface/hub" 2>/dev/null; echo; du -sh "$HOME/.cache/huggingface" 2>/dev/null; echo; ls -1 "$HOME/.cache/torch/hub/checkpoints" 2>/dev/null'

# ---------- 6. claims + config + docs snapshot ----------
ITEMS=()
for x in configs requirements*.txt pyproject.toml uv.lock Dockerfile* docker-compose* .github README* HANDOVER* DATASET_CARD* docs/*.md; do
  [[ -e "$x" ]] && ITEMS+=("$x")
done
(( ${#ITEMS} )) && tar cf - "${ITEMS[@]}" 2>/dev/null | tar xf - -C "$P/repo_files" 2>/dev/null
# the source is NOT copied wholesale; only line-level evidence below

# ---------- 7. static evidence scans (file:line, capped) ----------
SRC=(src scripts)
scan() { local out="$1" pat="$2"; shift 2; capsh "$out" "grep -rnE --include='*.py' --include='*.yaml' --include='*.md' '$pat' $* 2>/dev/null | head -80"; }
scan 10_scan_hardcoded_verdicts.txt 'PRODUCTION_READY|CERTIFIED|GOLD|99\.8|hardware_readiness_score *=|self_collision_events *= *0' src scripts
scan 11_scan_claims.txt 'OpenVLA|Octo|RLDS|production[- ]ready|all robots|any robot|any egocentric|universal' README.md HANDOVER.md DATASET_CARD.md docs src
scan 12_scan_todo_mock.txt 'TODO|FIXME|HACK|XXX|mock|placeholder|dummy|stub|NotImplemented' src scripts
scan 13_scan_resize_aspect.txt 'cv2\.resize|\.resize\(|224' src scripts
scan 14_scan_timestamps.txt 'timestamp *=|/ *fps|pts_time|CAP_PROP_POS_MSEC' src scripts
scan 15_scan_chirality.txt 'classification\[0\]\.label|mirrored_input|\.x *< *0\.5|handedness|HandIdentityTracker' src configs
scan 16_scan_intrinsics.txt '0\.85 *\*|fx *=|intrinsics|distortion|dist_coeffs' src configs
scan 17_scan_zero_fill.txt 'np\.zeros\(|np\.eye\(|fillna|nan_to_num' src
scan 18_scan_privacy.txt 'blur|anonym|consent|face_detect|-an|audio' src scripts configs
capsh 19_loc_per_module.txt 'find src scripts gates tests -name "*.py" | xargs wc -l | sort -n | tail -40'

# ---------- 8. entrypoints + tests ----------
capsh 20_run_help.txt "$PY scripts/run.py --help 2>&1 | head -80"
capsh 21_pytest_collect.txt "$PY -m pytest --collect-only -q 2>&1 | tail -5"
if [[ "${RUN_TESTS:-0}" == "1" ]]; then
  capsh 22_pytest_full.txt "$PY -m pytest -q --tb=line 2>&1 | tail -60"
else
  echo "pytest full run skipped (set RUN_TESTS=1 to include; takes several minutes on CPU)" > "$P/22_pytest_full.txt"
fi

# ---------- 9. docs the pipeline claims about itself ----------
for f in docs/STATUS.md docs/CONVENTIONS.md docs/PHYSICS_FINDINGS.md docs/PERF_BASELINE.md; do
  [[ -f "$f" ]] && echo "present: $f" || echo "MISSING: $f"
done > "$P/23_expected_docs.txt"

echo "[audit_pipeline] done -> $P  ($(ls "$P" | wc -l | tr -d ' ') items)"
echo "Next:  zsh audit_data.zsh $PACK"
