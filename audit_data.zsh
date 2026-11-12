#!/bin/zsh
# audit_data.zsh  -  "what did the pipeline actually produce, for every video?"
# Run from the repo root AFTER audit_pipeline.zsh:   zsh audit_data.zsh [PACK_DIR]
setopt null_glob pipefail

PACK="${1:-audit_pack}"
OUT_ROOT="${OUT_ROOT:-data/output}"
RAW_ROOT="${RAW_ROOT:-data/raw_videos}"
FRAMES="${FRAMES:-3}"
mkdir -p "$PACK/gates"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY="$(command -v python3)"; fi
fi
export PY ONLY
echo "[audit_data] repo=$PWD python=$PY out=$OUT_ROOT raw=$RAW_ROOT pack=$PACK"
[[ -d "$OUT_ROOT" ]] || { echo "ERROR: $OUT_ROOT not found. Run from the repo root or set OUT_ROOT."; exit 2; }

cat > "$PACK/_analyze.py" <<'PYEOF'
#!/usr/bin/env python3
import sys, os, json, re, subprocess, shutil, csv, time
from pathlib import Path
from collections import Counter
import numpy as np

PACK, REPO, OUT_ROOT, RAW_ROOT = [Path(x) for x in sys.argv[1:5]]
NFR = int(sys.argv[5]) if len(sys.argv) > 5 else 3
ONLY = os.environ.get("ONLY", "")
PYBIN = os.environ.get("PY", sys.executable)
EPS = PACK / "episodes"; EPS.mkdir(parents=True, exist_ok=True)

TH = dict(min_short_side_px=720, min_fps=29.0, ts_uniformity=0.01, vfr_rel=0.005, max_template_frac=0.20,
          min_segment_s=0.5, min_valid_hand_frac=0.80, max_dead_dim_frac=0.0, max_json_copy_mb=3.0)

def sh(cmd, timeout=600):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return 99, "", repr(e)

def jload(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None

def rate(s):
    try:
        a, b = str(s).split("/"); return float(a) / float(b) if float(b) else 0.0
    except Exception: return 0.0

def vinfo(path):
    rc, o, e = sh(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], 120)
    if rc: return {"error": "ffprobe failed", "path": str(path)}
    try: j = json.loads(o)
    except Exception: return {"error": "ffprobe json parse failed", "path": str(path)}
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    aud = [s for s in j.get("streams", []) if s.get("codec_type") == "audio"]
    rot = (v.get("tags") or {}).get("rotate")
    for sd in (v.get("side_data_list") or []):
        if "rotation" in sd: rot = sd["rotation"]
    rot = int(float(rot)) if rot is not None else 0
    w, h = int(v.get("width", 0) or 0), int(v.get("height", 0) or 0)
    dw, dh = (h, w) if abs(rot) % 180 == 90 else (w, h)
    fmt = j.get("format") or {}
    return dict(width=w, height=h, display_w=dw, display_h=dh, rotation=rot, codec=v.get("codec_name"),
                r_fps=rate(v.get("r_frame_rate")), avg_fps=rate(v.get("avg_frame_rate")),
                nb_frames=v.get("nb_frames"), duration=float(fmt.get("duration") or v.get("duration") or 0),
                bit_rate=fmt.get("bit_rate"), audio_streams=len(aud))

def find_source(ed, man):
    cands = []
    if man and man.get("video_path"): cands.append(Path(man["video_path"]))
    for stem in {ed.name, re.sub(r"_\d{6}$", "", ed.name)}:
        for ext in (".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV"): cands.append(RAW_ROOT / f"{stem}{ext}")
    for c in cands:
        p = c if c.is_absolute() else REPO / c
        if p.exists(): return p
    return None

def lerobot_root(ed):
    for c in (ed / "lerobot_v3", ed):
        if (c / "meta" / "info.json").exists(): return c
    return None

def analyze_lerobot(root, outdir):
    import pyarrow as pa, pyarrow.parquet as pq
    r = {}
    info = jload(root / "meta" / "info.json")
    if not info: return {"error": "info.json unreadable"}
    feats = info.get("features", {})
    r["ego_annotate_block"] = info.get("ego_annotate")
    r.update(codebase_version=info.get("codebase_version"), fps=info.get("fps"), total_frames_info=info.get("total_frames"),
             feature_keys=list(feats.keys()), ego_annotate_keys=sorted((info.get("ego_annotate") or {}).keys()),
             v2_leftovers=[n for n in ("episodes.jsonl", "tasks.jsonl") if (root / "meta" / n).exists()])
    files = sorted((root / "data").rglob("*.parquet"))
    r["parquet_files"] = len(files)
    if not files: r["error"] = "no parquet files"; return r
    tables = [pq.read_table(f) for f in files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    n = table.num_rows; r["rows"] = n
    schema = {f.name: str(f.type) for f in table.schema}
    mism = []
    for k, v in feats.items():
        if v.get("dtype") in ("float32", "float64") and k in schema:
            want = "float" if v["dtype"] == "float32" else "double"
            if want not in schema[k] or (want == "float" and "double" in schema[k]):
                mism.append(f"{k}: info={v['dtype']} parquet={schema[k]}")
    r["dtype_mismatches"] = mism
    if "timestamp" in schema:
        ts = np.asarray(table.column("timestamp").to_pylist(), float).ravel()
        epi = np.asarray(table.column("episode_index").to_pylist()).ravel() if "episode_index" in schema else np.zeros(n)
        worst_u, mono, ends = 0.0, True, []
        for e in np.unique(epi):
            t = ts[epi == e]; dt = np.diff(t)
            if len(dt) == 0: continue
            mono &= bool((dt > 0).all()); worst_u = max(worst_u, float(dt.std() / dt.mean()))
            ends.append(float(t[-1] + dt.mean()))
            r["ts_fps_measured"] = float(1 / dt.mean())
        r.update(ts_monotonic=mono, ts_uniformity_std_over_mean=worst_u, ts_end_s=ends[:5], n_episodes=int(len(np.unique(epi))))
    cols = {}; head = {}
    for key in ("observation.state", "action"):
        if key in schema and "list" in schema[key]:
            X = np.asarray(table.column(key).to_pylist(), float)
            names = (feats.get(key) or {}).get("names") or []
            names = [str(x) for x in names] + [f"d{i}" for i in range(len(names), X.shape[1])]
            std = X.std(0); dead = [names[i] for i in np.where(std < 1e-9)[0]]
            cols[key] = dict(dim=int(X.shape[1]), nan=int(np.isnan(X).sum()), inf=int(np.isinf(X).sum()),
                             n_dead=len(dead), frac_dead=len(dead) / X.shape[1], dead_dims=dead,
                             all_zero_rows_frac=float((np.abs(X).sum(1) == 0).mean()),
                             names=names, min=X.min(0).round(5).tolist(), max=X.max(0).round(5).tolist(), std=std.round(5).tolist())
            head[key] = X[:30].round(5)
    r["columns"] = cols
    st = cols.get("observation.state"); ac = cols.get("action")
    if st and ac:
        r["layout"] = dict(state_dim=st["dim"], action_dim=ac["dim"], dims_equal=st["dim"] == ac["dim"],
                           first_names_state=st["names"][:12], first_names_action=ac["names"][:12])
    if st:
        for i, nm in enumerate(st["names"]):
            if nm.endswith("wrist_z") or "gripper" in nm or nm.endswith("wrist_x") or nm.endswith("wrist_y"):
                r.setdefault("key_state_range", {})[nm] = dict(std=st["std"][i], min=st["min"][i], max=st["max"][i])
    try:
        vk = next((k for k in schema if "valid" in k.lower()), None)
        if st and vk:
            Xs = np.asarray(table.column("observation.state").to_pylist(), float)
            V = np.asarray(table.column(vk).to_pylist(), float); V = (V > 0.5); V = V.any(axis=1) if V.ndim > 1 else V
            idx = [i for i, nm in enumerate(st["names"]) if nm in ("wrist_x", "wrist_y", "wrist_z")]
            if len(idx) == 3 and len(Xs) > 4:
                P = Xs[:, idx]; both = V[1:] & V[:-1]; dP = np.abs(np.diff(P, axis=0)).max(1)
                hold = (dP < 1e-9) & both
                tri = V[2:] & V[1:-1] & V[:-2]; d2 = np.abs(P[2:] - 2 * P[1:-1] + P[:-2]).max(1)
                interp = (d2 < 1e-7) & tri & ~(hold[1:] & hold[:-1])
                run = best = 0
                for h_ in hold:
                    run = run + 1 if h_ else 0; best = max(best, run)
                r["valid_signal_provenance"] = dict(valid_frac=float(V.mean()), held_frac_of_valid_pairs=float(hold.sum() / max(both.sum(), 1)),
                    linear_interp_frac_of_valid_triples=float(interp.sum() / max(tri.sum(), 1)), longest_zero_motion_run_frames=int(best))
    except Exception as e:
        r["valid_signal_provenance"] = {"error": repr(e)}
    for k in schema:
        if "valid" in k.lower():
            v = np.asarray(table.column(k).to_pylist(), float)
            r.setdefault("valid_cols", {})[k] = dict(shape=list(v.shape), frac_true=float((v > 0.5).mean()))
    r["videos"] = []
    for v in sorted((root / "videos").rglob("*.mp4")):
        vi = vinfo(v)
        rc, o, e = sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                       "stream=nb_read_packets", "-of", "csv=p=0", str(v)], 180)
        try: vi["packets"] = int(o.strip().split(",")[0])
        except Exception: vi["packets"] = None
        vi["file"] = str(v.relative_to(root)); r["videos"].append(vi)
        if len(r["videos"]) == 1 and vi.get("duration"): grab(v, vi["duration"], outdir, max(NFR, 8), "export")
    r["video_packets_total"] = sum((x.get("packets") or 0) for x in r["videos"])
    try:
        with open(outdir / "data_head.csv", "w", newline="") as f:
            w = csv.writer(f)
            for key, X in head.items():
                w.writerow([f"# {key}"] + cols[key]["names"])
                for row in X: w.writerow(["", *row.tolist()])
    except Exception: pass
    return r

def analyze_h5(ed):
    out = []
    try: import h5py
    except ImportError: return [{"error": "h5py not installed"}]
    for p in sorted(list(ed.glob("*.hdf5")) + list(ed.glob("*.h5"))):
        d = {"file": p.name, "size_mb": round(p.stat().st_size / 1e6, 2), "datasets": {}}
        try:
            with h5py.File(p, "r") as f:
                def v(n, o):
                    if isinstance(o, h5py.Dataset):
                        d["datasets"][n] = dict(shape=list(o.shape), dtype=str(o.dtype), compression=o.compression)
                f.visititems(v)
        except Exception as e: d["error"] = repr(e)
        imgs = [(n, x) for n, x in d["datasets"].items() if n.endswith("image") and len(x["shape"]) == 4]
        d["image_aspects"] = {n: round(x["shape"][2] / max(x["shape"][1], 1), 4) for n, x in imgs}
        d["uncompressed_images"] = any(x["compression"] is None for _, x in imgs)
        d["name_claims_rlds"] = "rlds" in p.name.lower()
        out.append(d)
    return out

def find_lists(obj, pred, maxdepth=6):
    found = []
    def walk(o, d):
        if d > maxdepth: return
        if isinstance(o, list) and o and isinstance(o[0], dict) and pred(o[0]): found.append(o)
        if isinstance(o, dict):
            for v in o.values(): walk(v, d + 1)
        elif isinstance(o, list):
            for v in o[:3]: walk(v, d + 1)
    walk(obj, 0); return found

def seg_times(d):
    for a, b in (("start", "end"), ("start_time", "end_time"), ("start_s", "end_s"), ("t_start", "t_end")):
        if a in d and b in d:
            try: return float(d[a]), float(d[b])
            except Exception: return None
    return None

def analyze_annotations(ed):
    p = ed / "annotations.json"
    if not p.exists():
        c = sorted(ed.rglob("annotations*.json"))
        p = c[0] if c else None
    if not p: return {"error": "annotations.json not found"}, None
    a = jload(p)
    if a is None: return {"error": "annotations unreadable", "file": str(p)}, None
    res = {"file": str(p.relative_to(ed)), "size_mb": round(p.stat().st_size / 1e6, 2)}
    segl = find_lists(a, lambda d: seg_times(d) is not None)
    if segl:
        seg = max(segl, key=len); durs = np.array([max(seg_times(s)[1] - seg_times(s)[0], 0) for s in seg])
        src = Counter(str(s.get("caption_source", "MISSING")) for s in seg)
        caps = [str(s.get("description") or s.get("caption") or s.get("text") or "") for s in seg]
        labels = Counter(str(s.get("label") or s.get("action") or s.get("type") or "?") for s in seg)
        res["segments"] = dict(n=len(seg), min_s=float(durs.min()), mean_s=float(durs.mean()), max_s=float(durs.max()),
                               frac_shorter_than_min=float((durs < TH["min_segment_s"]).mean()),
                               caption_source=dict(src), template_frac=float(src.get("template", 0) / len(seg)),
                               caption_source_missing=bool(src.get("MISSING", 0) == len(seg)),
                               distinct_caption_ratio=float(len(set(caps)) / max(len(caps), 1)),
                               labels=dict(labels), sample_captions=caps[:6])
    else: res["segments"] = "SCHEMA_UNKNOWN"
    fl = find_lists(a, lambda d: ("left_hand" in d) or ("right_hand" in d))
    hand_valid = None
    if fl:
        fr = max(fl, key=len); L = np.array([bool(f.get("left_hand")) for f in fr]); R = np.array([bool(f.get("right_hand")) for f in fr])
        hand_valid = float((L | R).mean())
        sample = next((f.get("right_hand") or f.get("left_hand") for f in fr if (f.get("right_hand") or f.get("left_hand"))), None)
        prov = Counter()
        for f in fr:
            for sd in ("left_hand", "right_hand"):
                hh = f.get(sd)
                if isinstance(hh, dict):
                    for kk, vv in hh.items():
                        if re.search(r"interp|rescue|source|held|synth|fill|pass|conf", kk.lower()) and not isinstance(vv, (list, dict)):
                            prov[f"{kk}={vv if not isinstance(vv, float) else round(vv, 1)}"] += 1
        res["hands_extra"] = dict(hand_keys=sorted(sample.keys()) if isinstance(sample, dict) else str(type(sample)), provenance=dict(prov.most_common(15)))
        res["hands"] = dict(frames=len(fr), left_frac=float(L.mean()), right_frac=float(R.mean()), either_frac=hand_valid,
                            both_frac=float((L & R).mean()), left_minus_right=float(L.mean() - R.mean()))
    else: res["hands"] = "SCHEMA_UNKNOWN"
    objs = find_lists(a, lambda d: "label" in d and ("bbox" in d or "box" in d))
    if objs:
        res["objects"] = dict(n_detections=len(max(objs, key=len)),
                              unique_labels=sorted({str(o.get("label")) for o in max(objs, key=len)})[:30])
    return res, hand_valid

def analyze_npz(ed, hand_valid, fps):
    out = []
    for p in sorted(ed.rglob("*.npz")):
        if "wam" in p.parts and "traj" not in p.name.lower(): continue
        d = {"file": str(p.relative_to(ed))}
        try: z = np.load(p, allow_pickle=False)
        except Exception as e: d["error"] = repr(e); out.append(d); continue
        d["keys"] = {k: list(z[k].shape) for k in z.files}
        if "q" in z.files and z["q"].ndim == 2 and len(z["q"]) > 4 and fps:
            q = z["q"].astype(float); dt = 1.0 / float(fps)
            qd = np.diff(q, axis=0) / dt; qdd = np.diff(qd, axis=0) / dt; qddd = np.diff(qdd, axis=0) / dt
            d.update(max_joint_vel=float(np.abs(qd).max()), max_joint_jerk=float(np.abs(qddd).max()),
                     max_frame_jump_deg=float(np.degrees(np.abs(np.diff(q, axis=0)).max())),
                     frac_static_rows=float((np.abs(qd).sum(1) < 1e-9).mean()))
        for k in z.files:
            a = z[k]
            if a.dtype == object or a.ndim != 1 or a.size == 0: continue
            kl = k.lower()
            if any(t in kl for t in ("err", "resid")):
                d.setdefault("error_arrays", {})[k] = dict(mean=float(a.mean()), std=float(a.std()), min=float(a.min()), max=float(a.max()), constant=bool(a.std() < 1e-12))
            if any(t in kl for t in ("valid", "reach", "converg", "success")):
                fr = float((a.astype(float) > 0.5).mean()); d.setdefault("bool_arrays", {})[k] = fr
                if hand_valid is not None and abs(fr - hand_valid) < 1e-3:
                    d.setdefault("suspicious", []).append(f"{k} fraction == hand-valid fraction ({fr:.4f}): looks like tracking validity relabelled as solver success")
        d["has_target_ee"] = any("target_ee" in k for k in z.files)
        out.append(d)
    return out

def scrape(obj, keys, out, depth=0):
    if depth > 6: return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and not isinstance(v, (dict, list)): out.setdefault(k, v)
            else: scrape(v, keys, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:5]: scrape(v, keys, out, depth + 1)

def copy_reports(ed, outdir):
    rep = {}; rd = outdir / "reports"; rd.mkdir(exist_ok=True)
    for p in sorted(ed.rglob("*.json")):
        if "cache" in p.parts: continue
        nm = p.name.lower()
        if p.stat().st_size > TH["max_json_copy_mb"] * 1e6: continue
        if p.name == "annotations.json": continue
        if re.search(r"(verif|readiness|physics|quality|report|manifest|consent|capture_meta)", nm) or p.name in ("info.json", "stats.json"):
            shutil.copy2(p, rd / (str(p.relative_to(ed)).replace("/", "__")))
            j = jload(p)
            if j is not None:
                s = {}; scrape(j, {"passed", "verdict", "hardware_readiness_score", "reachable_pct", "reachable", "self_collision_frames",
                                   "self_collision_events", "weld_slip_mm", "overall_status", "accept"}, s)
                if s: rep[str(p.relative_to(ed))] = s
    return rep

def find_first(ed, pattern):
    c = sorted(p for p in ed.rglob("*.json") if re.search(pattern, p.name.lower()) and "cache" not in p.parts)
    return c[0] if c else None

def load_named(ed):
    out = {}
    p = find_first(ed, r"hardware_readiness")
    if p:
        j = jload(p) or {}; hw = {}
        for robot, v in j.items():
            a = (v or {}).get("hardware_audit") or {}
            hw[robot] = dict(status=(v or {}).get("capability_status"), dof=(v or {}).get("dof"),
                             verdict=a.get("verdict"), score=a.get("hardware_readiness_score"), hand_valid_pct=a.get("hand_valid_pct"),
                             ik_convergence_pct=a.get("ik_convergence_pct"), mean_pos_mm=a.get("mean_ik_residual_mm"), p95_pos_mm=a.get("p95_ik_residual_mm"),
                             p95_ori_deg=a.get("p95_ori_residual_deg"), max_vel=a.get("max_joint_velocity_rad_s"), max_jerk=a.get("max_joint_jerk_rad_s3"),
                             clamp=a.get("clamp_events"), selfcol=a.get("self_collision_frames"))
        out["hw"] = hw
    p = find_first(ed, r"physics_verification")
    if p: out["phys"] = jload(p)
    p = find_first(ed, r"roboverify|certificate")
    if p:
        j = jload(p) or {}
        l1 = next((v for k, v in j.items() if k.startswith("layer_1") and isinstance(v, dict)), {})
        out["cert"] = dict(verdict=j.get("overall_qc_verdict"), untruncated_hand_pct=l1.get("untruncated_hand_visibility_pct"), layer_status={k: v.get("status") for k, v in j.items() if isinstance(v, dict) and "status" in v})
    p = find_first(ed, r"consent")
    if p: out["consent"] = jload(p)
    p = ed / "quality.json"
    if p.exists(): out["quality_json"] = jload(p)
    return out

def flat(prefix, o, out):
    if isinstance(o, dict):
        for k, v in o.items(): flat(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(o, bool): out[prefix] = o
    elif isinstance(o, (int, float)): out[prefix] = round(float(o), 6)
    elif isinstance(o, str) and re.search(r"verdict|accept|status|passed", prefix.lower()): out[prefix] = o

def grab(src, dur, outdir, n, prefix):
    if not dur: return 0
    k = 0
    for i in range(n):
        t = dur * (i + 1) / (n + 1)
        rc, _, _ = sh(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
                       "-vf", "scale=640:-2", "-q:v", "5", str(outdir / f"{prefix}_{i+1}.jpg")], 120)
        k += (rc == 0)
    return k

def make_flags(s):
    F = []
    add = lambda c, d: F.append(f"{c}: {d}")
    if s.get("manifest_status") == "MISSING": add("MANIFEST_MISSING", "")
    if s.get("manifest_status") == "present_unknown_schema": add("MANIFEST_SCHEMA_UNKNOWN", ",".join(s.get("manifest_keys") or []))
    src = s.get("source") or {}
    if src and not src.get("error"):
        if min(src["display_w"], src["display_h"]) < TH["min_short_side_px"]:
            add("LOW_RES", f"{src['display_w']}x{src['display_h']} (suggested short side >= {TH['min_short_side_px']})")
        if src["r_fps"] and abs(src["r_fps"] - src["avg_fps"]) / src["r_fps"] > TH["vfr_rel"]:
            add("VFR_SOURCE", f"r_fps={src['r_fps']:.3f} avg_fps={src['avg_fps']:.3f}: index/fps timestamps are wrong; PTS required")
        if src["avg_fps"] and src["avg_fps"] < TH["min_fps"]: add("LOW_FPS", f"{src['avg_fps']:.2f}")
    elif not src: add("SOURCE_NOT_FOUND", "cannot cross-check timing/aspect against the raw video")
    lr = s.get("lerobot") or {}
    if not lr: add("NO_LEROBOT_EXPORT", "")
    elif lr.get("error"): add("LEROBOT_ERROR", lr["error"])
    else:
        if lr.get("v2_leftovers"): add("V2_LEFTOVERS", ",".join(lr["v2_leftovers"]))
        if lr.get("dtype_mismatches"): add("DTYPE_MISMATCH", f"{len(lr['dtype_mismatches'])} features (e.g. {lr['dtype_mismatches'][0]})")
        if not lr.get("ego_annotate_keys"): add("NO_EGO_ANNOTATE_BLOCK", "info.json has no units/frame/scale contract")
        if lr.get("ts_monotonic") is False: add("TS_NOT_MONOTONIC", "")
        if (lr.get("ts_uniformity_std_over_mean") or 0) > TH["ts_uniformity"]: add("TS_NOT_UNIFORM", f"{lr['ts_uniformity_std_over_mean']:.4f}")
        if lr.get("video_packets_total") and lr.get("rows") and lr["video_packets_total"] != lr["rows"]:
            add("VIDEO_ROWS_MISMATCH", f"packets={lr['video_packets_total']} rows={lr['rows']}")
        for v in lr.get("videos", []):
            if v.get("audio_streams"): add("EXPORT_HAS_AUDIO", v.get("file", ""))
            if src and not src.get("error") and v.get("display_h"):
                sa, ea = src["display_w"] / src["display_h"], v["display_w"] / v["display_h"]
                if abs(ea - sa) / sa > 0.02: add("EXPORT_ASPECT_DISTORTED", f"{ea:.3f} vs source {sa:.3f}")
        for key, c in (lr.get("columns") or {}).items():
            if c["nan"] or c["inf"]: add("NAN_INF", f"{key}: nan={c['nan']} inf={c['inf']}")
            if c["n_dead"]: add("DEAD_DIMS", f"{key}: {c['n_dead']}/{c['dim']} constant dims ({', '.join(c['dead_dims'][:8])}{'...' if c['n_dead'] > 8 else ''})")
        if not lr.get("valid_cols"): add("NO_VALIDITY_MASK", "no column containing 'valid': missing-hand frames cannot be distinguished from real zeros")
        lay = lr.get("layout")
        if lay and not lay["dims_equal"]: add("STATE_ACTION_DIM_DIFFER", f"state={lay['state_dim']} action={lay['action_dim']} (fine only if documented)")
    for h in s.get("hdf5", []):
        if h.get("error"): add("HDF5_ERROR", h["error"]); continue
        if h.get("name_claims_rlds"): add("HDF5_CALLED_RLDS", h["file"])
        if h.get("uncompressed_images"): add("HDF5_UNCOMPRESSED_IMAGES", h["file"])
        if src and not src.get("error") and src["display_h"]:
            sa = src["display_w"] / src["display_h"]
            for n, a in (h.get("image_aspects") or {}).items():
                if abs(a - sa) / sa > 0.02: add("HDF5_IMAGE_ASPECT_DISTORTED", f"{a:.3f} vs source {sa:.3f}")
    an = s.get("annotations") or {}
    sg = an.get("segments")
    if isinstance(sg, dict):
        if sg["frac_shorter_than_min"] > 0: add("SHORT_SEGMENTS", f"{sg['frac_shorter_than_min']*100:.0f}% shorter than {TH['min_segment_s']}s (min {sg['min_s']:.3f}s)")
        if sg["caption_source_missing"]: add("NO_CAPTION_SOURCE", "cannot tell VLM captions from templates")
        if sg["template_frac"] > TH["max_template_frac"]: add("TEMPLATE_CAPTIONS", f"{sg['template_frac']*100:.0f}% template")
    hd = an.get("hands")
    if isinstance(hd, dict):
        if hd["either_frac"] < TH["min_valid_hand_frac"]: add("LOW_HAND_VALIDITY", f"either-hand valid {hd['either_frac']*100:.1f}%")
        if hd["left_minus_right"] > 0.02: add("LEFT_EXCEEDS_RIGHT", f"left {hd['left_frac']*100:.1f}% vs right {hd['right_frac']*100:.1f}% (interpolation or chirality artefact?)")
    for t in s.get("retarget", []):
        if t.get("error"): add("NPZ_UNREADABLE", t["file"]); continue
        for sus in t.get("suspicious", []): add("RETARGET_SUSPICIOUS", f"{t['file']}: {sus}")
        for k, e in (t.get("error_arrays") or {}).items():
            if e["constant"]: add("CONSTANT_METRIC", f"{t['file']}:{k} identical on every frame")
        if not t.get("has_target_ee") and "q" in (t.get("keys") or {}): add("NO_TARGET_EE_SAVED", t["file"] + " (cannot independently recompute residuals)")
    q = s.get("quality_metrics") or {}; hw = s.get("hw") or {}; cert = s.get("cert") or {}
    hv = {}
    if q.get("hand_valid_ratio") is not None: hv["manifest"] = float(q["hand_valid_ratio"])
    if isinstance(an.get("hands"), dict): hv["annotations"] = an["hands"]["either_frac"]
    for k, v in (lr.get("valid_cols") or {}).items(): hv["lerobot_valid_col"] = v["frac_true"]; break
    for r_, v in hw.items():
        if v.get("hand_valid_pct") is not None: hv["hardware_report"] = v["hand_valid_pct"] / 100.0; break
    s["hand_valid_estimates"] = hv
    if len(hv) > 1 and max(hv.values()) - min(hv.values()) > 0.05:
        add("HAND_VALID_DISAGREES", ", ".join(f"{k}={v*100:.1f}%" for k, v in hv.items()))
    ikm = q.get("ik_convergence_pct"); ikh = next((v.get("ik_convergence_pct") for v in hw.values() if v.get("ik_convergence_pct") is not None), None)
    if ikm is not None and ikh is not None and abs(ikm - ikh) > 5: add("IK_DISAGREES", f"manifest={ikm} hardware_report={ikh}")
    lowest = min(hv.values()) if hv else None
    if s.get("accept_episode") is True and lowest is not None and lowest < TH["min_valid_hand_frac"]:
        add("ACCEPTED_DESPITE_LOW_HAND_VALIDITY", f"accept_episode=true with hand-valid min {lowest*100:.1f}% (failed_quality_checks={s.get('failed_quality_checks')})")
    if s.get("accept_episode") is True and ikm is not None and ikm < 50: add("ACCEPTED_DESPITE_LOW_IK", f"ik_convergence_pct={ikm}")
    if cert.get("verdict") and re.search(r"CERTIFIED|GOLD|PRODUCTION", str(cert["verdict"])) and lowest is not None and lowest < TH["min_valid_hand_frac"]:
        add("POSITIVE_VERDICT_LOW_QUALITY", f"{cert['verdict']} with hand-valid {lowest*100:.1f}%, untruncated {cert.get('untruncated_hand_pct')}%")
    if len(hw) > 1:
        sig = {}
        for r_, v in hw.items(): sig.setdefault((v.get("mean_pos_mm"), v.get("p95_pos_mm"), v.get("score")), []).append(r_)
        for k_, rs in sig.items():
            if len(rs) > 1 and k_[0] is not None: add("HW_IDENTICAL_ACROSS_ROBOTS", f"{rs} share pos residual {k_[0]}/{k_[1]} and score {k_[2]} (copied, not measured?)")
        for r_, v in hw.items():
            if (v.get("max_vel") or 0) > 10 or (v.get("max_jerk") or 0) > 1e4:
                add("HW_IMPLAUSIBLE_DYNAMICS", f"{r_}: max_vel={v.get('max_vel')} rad/s max_jerk={v.get('max_jerk')} verdict={v.get('verdict')} (check vs robot limits)")
    if lr.get("rows") and lr.get("fps") and src and not src.get("error") and src.get("duration"):
        if abs(lr["rows"] / lr["fps"] - src["duration"]) > 1.0:
            add("SOURCE_DURATION_MISMATCH", f"export {lr['rows']/lr['fps']:.1f}s vs matched source {src['duration']:.1f}s (wrong source match, or truncated/padded export?)")
    for f, rr in (s.get("reports") or {}).items():
        if rr.get("passed") is False: add("REPORT_SAYS_FAILED", f"{f}: {rr}")
    return F

def main():
    rows, flags_all = [], {}
    if not OUT_ROOT.exists(): print("OUT_ROOT missing"); return
    eps = []
    for d in sorted(OUT_ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith("."): continue
        if ONLY and not re.search(ONLY, d.name): continue
        if any((d / x).exists() for x in ("manifest.json", "lerobot_v3", "annotations.json", "meta/info.json")): eps.append(d)
    print(f"[analyze] {len(eps)} episode folders")
    for ed in eps:
        t0 = time.time(); od = EPS / ed.name; od.mkdir(parents=True, exist_ok=True)
        man = jload(ed / "manifest.json"); s = {"episode": ed.name, "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ed.stat().st_mtime))}
        q = (man or {}).get("quality_metrics") or {}
        s["manifest_present"] = man is not None
        s["manifest_keys"] = sorted(man.keys()) if man else None
        s["accept_episode"] = (man or {}).get("accept_episode")
        s["failed_quality_checks"] = (man or {}).get("failed_quality_checks")
        s["quality_metrics"] = q
        if not man: s["manifest_status"] = "MISSING"
        elif (man or {}).get("overall_status"): s["manifest_status"] = man["overall_status"]
        elif s["accept_episode"] is True: s["manifest_status"] = "accepted"
        elif s["accept_episode"] is False: s["manifest_status"] = "rejected"
        else: s["manifest_status"] = "present_unknown_schema"
        s["manifest_stages"] = {k: (v.get("status"), v.get("duration_s")) for k, v in ((man or {}).get("stages") or {}).items()} if man else {}
        s["elapsed_s"] = (man or {}).get("elapsed_s") or q.get("elapsed_s") or (man or {}).get("wall_seconds")
        s["manifest_metrics"] = (man or {}).get("metrics")
        s["quality_report"] = (man or {}).get("quality") or (man or {}).get("quality_report")
        src = find_source(ed, man)
        s["source_path"] = str(src) if src else None
        s["source"] = vinfo(src) if src else None
        if src and s["source"] and not s["source"].get("error"):
            s["frames_source"] = grab(src, s["source"]["duration"], od, NFR, "src")
        try:
            lr = lerobot_root(ed); s["lerobot"] = analyze_lerobot(lr, od) if lr else None
        except Exception as e: s["lerobot"] = {"error": f"analyzer crashed: {e!r}"}
        s["hdf5"] = analyze_h5(ed)
        try: s["annotations"], hv = analyze_annotations(ed)
        except Exception as e: s["annotations"], hv = {"error": repr(e)}, None
        lr_valid = None
        for k, v in ((s.get("lerobot") or {}).get("valid_cols") or {}).items(): lr_valid = v["frac_true"]; break
        hv = hv if hv is not None else lr_valid
        fps = (s.get("lerobot") or {}).get("fps") or 30
        try: s["retarget"] = analyze_npz(ed, hv, fps)
        except Exception as e: s["retarget"] = [{"error": repr(e), "file": "?"}]
        s["reports"] = copy_reports(ed, od)
        s.update(load_named(ed))
        for ov in list(ed.rglob("*overlay*.mp4"))[:1]:
            vi = vinfo(ov); s["overlay"] = dict(file=str(ov.relative_to(ed)), **{k: vi.get(k) for k in ("width", "height", "duration")})
            grab(ov, vi.get("duration"), od, NFR, "overlay")
        ap = ed / "annotations.json"
        if ap.exists() and ap.stat().st_size <= TH["max_json_copy_mb"] * 1e6: shutil.copy2(ap, od / "annotations.json")
        g13 = REPO / "gates" / "g13_integration_invariants.py"
        if g13.exists() and src:
            rc, o, e = sh([PYBIN, str(g13), str(ed), str(src)], 300)
            (od / "g13_output.txt").write_text(f"exit={rc}\n{o}\n{e}"); s["g13_exit"] = rc
        s["flags"] = make_flags(s); flags_all[ed.name] = s["flags"]
        s["analysis_seconds"] = round(time.time() - t0, 1)
        (od / "summary.json").write_text(json.dumps(s, indent=2, default=str))
        lr = s.get("lerobot") or {}; cols = lr.get("columns") or {}; sg = (s.get("annotations") or {}).get("segments"); hd = (s.get("annotations") or {}).get("hands")
        dur = (s.get("source") or {}).get("duration") or 0
        rows.append(dict(episode=ed.name, mtime=s["mtime"], status=s["manifest_status"], elapsed_s=s["elapsed_s"],
            compute_s_per_video_s=round(s["elapsed_s"] / dur, 2) if s["elapsed_s"] and dur else "",
            src_res=f"{(s.get('source') or {}).get('display_w','')}x{(s.get('source') or {}).get('display_h','')}",
            src_fps=(s.get("source") or {}).get("avg_fps", ""), duration_s=round(dur, 2), rows=lr.get("rows", ""),
            dead_action_dims=(cols.get("action") or {}).get("n_dead", ""), dead_state_dims=(cols.get("observation.state") or {}).get("n_dead", ""),
            hand_either_valid=hd.get("either_frac", "") if isinstance(hd, dict) else "",
            template_frac=sg.get("template_frac", "") if isinstance(sg, dict) else "",
            short_seg_frac=sg.get("frac_shorter_than_min", "") if isinstance(sg, dict) else "", n_flags=len(s["flags"])))
        print(f"  {ed.name}: {len(s['flags'])} flags ({s['analysis_seconds']}s)")
    gl = []
    series = {}; ep_ids = []
    for ed in eps:
        sj = jload(EPS / ed.name / "summary.json") or {}
        if not sj.get("manifest_present"): continue
        ep_ids.append(ed.name); flatd = {}
        flat("quality", sj.get("quality_metrics") or {}, flatd); flat("phys", sj.get("phys") or {}, flatd)
        flat("cert", (sj.get("cert") or {}).get("verdict") and {"verdict": sj["cert"]["verdict"]} or {}, flatd)
        for r_, v in (sj.get("hw") or {}).items(): flat(f"hw.{r_}", {k: x for k, x in v.items() if x is not None}, flatd)
        flat("consent", sj.get("consent") or {}, flatd)
        flatd["manifest.accept_episode"] = sj.get("accept_episode")
        for k, v in flatd.items(): series.setdefault(k, []).append(v)
    benign = ("min_segment_duration", "schema", "license", "purpose", "retention", "lawful", "dof")
    for k, vals in sorted(series.items()):
        if len(vals) >= 3 and len(set(map(str, vals))) == 1 and not any(b in k.lower() for b in benign):
            gl.append(f"CROSS_EPISODE_CONSTANT: {k} = {vals[0]} in all {len(vals)} episodes (placeholder, default or copied value?)")
    for k in ("consent.collector_id", "consent.collection_date", "consent.subject_acknowledged"):
        if k in series and len(series[k]) >= 3 and len(set(map(str, series[k]))) == 1:
            gl.append(f"CONSENT_LOOKS_AUTOGENERATED: {k} = {series[k][0]} identical across {len(series[k])} different videos")
    if rows:
        with open(PACK / "episodes_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    L = ["# PRECHECK (measured facts + threshold flags)", "",
         "This file is NOT a readiness verdict. Thresholds are suggestions, shown below; the verdict comes from the audit.", "",
         "Thresholds: " + ", ".join(f"{k}={v}" for k, v in TH.items()), ""]
    allc = Counter(code for fl in flags_all.values() for code in {c.split(":")[0] for c in fl})
    L += ["## Flag frequency across episodes", ""] + [f"- {k}: {v}/{len(flags_all)}" for k, v in allc.most_common()] + [""]
    L += ["## Cross-episode findings", ""] + ([f"- {g}" for g in gl] if gl else ["- none"]) + [""]
    for ep, fl in flags_all.items(): L += [f"## {ep}"] + ([f"- {c}" for c in fl] if fl else ["- no flags"]) + [""]
    (PACK / "PRECHECK.md").write_text("\n".join(L))
    print("[analyze] wrote PRECHECK.md and episodes_summary.csv")

main()
PYEOF

"$PY" "$PACK/_analyze.py" "$PACK" "$PWD" "$OUT_ROOT" "$RAW_ROOT" "$FRAMES" || echo "[audit_data] analyzer exited non-zero (partial output may exist)"

if [[ "${RUN_GATES:-1}" == "1" && -f gates/run_all.py ]]; then
  LATEST=""
  for m in $(ls -t "$OUT_ROOT"/*/manifest.json 2>/dev/null); do LATEST="${m:h}"; break; done
  if [[ -n "$LATEST" ]]; then
    echo "[audit_data] running gates on $LATEST (includes pytest; this can take several minutes)"
    "$PY" gates/run_all.py "$LATEST" --report-out "$PACK/gates/gates_report_latest.json" > "$PACK/gates/run_all_latest.txt" 2>&1
    echo "[exit=$?]" >> "$PACK/gates/run_all_latest.txt"
  else
    echo "no manifest.json found; gates not run" > "$PACK/gates/run_all_latest.txt"
  fi
else
  echo "gates skipped (RUN_GATES=0 or gates/run_all.py missing)" > "$PACK/gates/run_all_latest.txt"
fi
[[ -f data/chirality_labels.csv ]] && { cp data/chirality_labels.csv "$PACK/gates/"; echo "chirality labels rows: $(($(wc -l < data/chirality_labels.csv)-1))" >> "$PACK/gates/run_all_latest.txt"; }

rm -f "$PACK/_analyze.py"
rm -f "$PACK.zip"
zip -rq "$PACK.zip" "$PACK" -x '*.mp4' '*.mov' '*.hdf5' '*.h5' '*.parquet' '*.npz' '*.pt' '*.onnx'
SIZE=$(du -h "$PACK.zip" | cut -f1)
echo ""
echo "================================================================"
echo " DONE.  Upload this one file to Claude:  $PACK.zip   ($SIZE)"
echo " Quick look: cat $PACK/PRECHECK.md | head -60"
echo "================================================================"
