"""g17_metadata_flag_parity.py

Every capability claimed in lerobot_v3/meta/info.json -> "ego_annotate" must be backed by data.
Usage:  python gates/g17_metadata_flag_parity.py <episode_dir> [--json-out PATH]       Exit: 0 PASS, 1 FAIL.
"""
import sys, json, re, argparse
from pathlib import Path
import numpy as np

GATE = "g17_metadata_flag_parity"
POSE_RE = re.compile(r"(camera|cam|ego).*pose", re.I)
BENIGN_BOOL_KEYS = {"calibrated", "is_metric"}
METRIC_SCALE_SOURCES = {"calibrated_depth", "lidar", "arkit", "arcore", "marker"}


class R:
    def __init__(s): s.fails, s.warns, s.info = [], [], {}
    def check(s, name, ok, detail="", warn=False):
        print(f"  [{'PASS' if ok else ('WARN' if warn else 'FAIL')}] {name}: {detail}")
        if not ok: (s.warns if warn else s.fails).append(f"{name}: {detail}")
    def finish(s, out=None):
        st = "FAIL" if s.fails else "PASS"; print(f"G17 VERDICT: {st}")
        if out: Path(out).write_text(json.dumps(dict(gate=GATE, status=st, failures=s.fails, warnings=s.warns, info=s.info), indent=1, default=str))
        return 1 if s.fails else 0


def jload(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def load_columns(ep):
    import pyarrow.parquet as pq, pyarrow as pa
    files = sorted((ep / "lerobot_v3" / "data").rglob("*.parquet"))
    if not files: return None
    t = [pq.read_table(f) for f in files]; t = pa.concat_tables(t) if len(t) > 1 else t[0]
    cols = {}
    for name in t.column_names:
        try:
            arr = np.asarray(t.column(name).to_pylist(), dtype=float); cols[name] = arr
        except Exception: pass
    return cols


def width(arr):
    return 1 if arr.ndim == 1 else int(np.prod(arr.shape[1:]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episode_dir"); ap.add_argument("--json-out"); a = ap.parse_args()
    ep = Path(a.episode_dir); r = R(); print(f"=== G17: METADATA FLAGS vs DATA ({ep}) ===")
    info = jload(ep / "lerobot_v3" / "meta" / "info.json"); q = jload(ep / "quality.json") or {}
    r.check("info.json readable", info is not None, "")
    r.check("quality.json readable", bool(q), "")
    if info is None: return r.finish(a.json_out)
    ea = info.get("ego_annotate") or {}; feats = info.get("features") or {}
    cols = load_columns(ep)
    r.check("parquet columns readable", cols is not None, "")
    if cols is None: return r.finish(a.json_out)
    cam = q.get("camera") or {}
    for k in ea:
        if "egoverse" in k.lower() or "egoverse" in str(ea[k]).lower():
            r.check(f"unverifiable tier label '{k}'", False, f"{k}={ea[k]!r}: no evidence file defines this tier")
    handled = set()

    for k in list(ea):
        kl = k.lower()
        if "pose" in kl and ea.get(k) is True:
            handled.add(k)
            cand = [(n, c) for n, c in cols.items() if POSE_RE.search(n)]
            good = [(n, c) for n, c in cand if width(c) in (6, 7, 16) and np.isfinite(c).all() and np.ptp(c, axis=0).max() > 1e-9]
            r.check(f"{k}=true has a camera-pose column", bool(good), f"candidates={[n for n,_ in cand]} usable={[n for n,_ in good]}")
            r.check(f"{k}=true consistent with quality.camera.pose_source", str(cam.get("pose_source", "none")).lower() not in ("none", ""), f"pose_source={cam.get('pose_source')}")
        if kl.startswith("bimanual") and ea.get(k) is True:
            handled.add(k)
            names = " ".join([n.lower() for n in feats] + [str(x).lower() for f in feats.values() for x in (f.get("names") or [])]
                             ) if isinstance(feats, dict) else ""
            both = "left" in names and "right" in names
            r.check(f"{k}=true exposes both hands in features", both, "no feature or name mentions both 'left' and 'right'" if not both else "left+right found")
        if ("21kp" in kl or "keypoint" in kl) and ea.get(k) is True:
            handled.add(k)
            kp = [n for n, c in cols.items() if width(c) in (63, 126)]
            r.check(f"{k}=true has a 21-keypoint column", bool(kp), f"columns with 63/126 floats per row: {kp}")
        if k == "multi_embodiment_retargeted" and isinstance(ea[k], list):
            handled.add(k)
            for rb in ea[k]:
                r.check(f"retargeted robot '{rb}' has trajectory.npz", (ep / "embodiments" / rb / "trajectory.npz").exists(), "")
                r.check(f"retargeted robot '{rb}' in quality.robots", rb in (q.get("robots") or {}), "")
    if "calibrated" in ea:
        handled.add("calibrated")
        if ea["calibrated"] is True:
            cal = cam.get("calibration") or {}
            ok = cam.get("intrinsics_source") == "calibrated" and "file_sha256" in cal and cal.get("rms_reproj_px", 99) <= 1.0
            r.check("calibrated=true backed by a calibration record", ok, f"intrinsics_source={cam.get('intrinsics_source')} calibration={cal}")
        else:
            r.check("calibrated=false: no calibrated distortion model claimed", str(cam.get("intrinsics_source")) != "calibrated", f"intrinsics_source={cam.get('intrinsics_source')}")
    if "is_metric" in ea:
        handled.add("is_metric")
        units = (ea.get("units") or {}).get("wrist_pos", "")
        if ea["is_metric"] is True:
            r.check("is_metric=true backed by a metric scale source", cam.get("scale_source") in METRIC_SCALE_SOURCES, f"scale_source={cam.get('scale_source')}")
        else:
            r.check("is_metric=false: wrist units must not claim meters", str(units).lower() not in ("m", "meter", "meters", "metres"), f"units.wrist_pos={units!r} (use 'assumed_m' or 'relative')")
    for k, v in ea.items():
        if isinstance(v, bool) and v and k not in handled and k not in BENIGN_BOOL_KEYS:
            r.check(f"claim flag '{k}' is verifiable", False, "no rule in g17 can verify this true flag: remove it or add evidence + rule")
    ab = ea.get("absent_dims") or []
    st = cols.get("observation.state"); ac = cols.get("action")
    names = (feats.get("observation.state") or {}).get("names") or []
    for d in ab:
        if st is not None and d in names:
            i = names.index(d); r.check(f"declared absent dim '{d}' is constant", float(np.ptp(st[:, i])) < 1e-9, "")
    r.info["ego_annotate_keys"] = sorted(ea.keys())
    return r.finish(a.json_out)


if __name__ == "__main__":
    sys.exit(main())
