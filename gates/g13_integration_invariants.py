import sys, json, subprocess
from pathlib import Path
import numpy as np

fails = []
def check(name, ok, detail, warn=False):
    print(f"[{'PASS' if ok else ('WARN' if warn else 'FAIL')}] {name}: {detail}")
    if not ok and not warn: fails.append(name)

def probe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
                        "-show_format", "-of", "json", str(p)], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr)
    j = json.loads(r.stdout); return j["streams"][0], j["format"]

def rate(s):
    a, b = s.split("/"); return float(a) / float(b) if float(b) else 0.0

def display_dims(s):
    w, h = int(s["width"]), int(s["height"]); rot = (s.get("tags") or {}).get("rotate")
    for sd in (s.get("side_data_list") or []):
        if "rotation" in sd: rot = sd["rotation"]
    rot = int(float(rot)) if rot is not None else 0
    return ((h, w) if abs(rot) % 180 == 90 else (w, h)), rot

def content_aspect(img):
    m = img.max(axis=2) > 8
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    if len(rows) == 0 or len(cols) == 0: return None
    return (cols[-1] - cols[0] + 1) / (rows[-1] - rows[0] + 1)

def main():
    ep, src = Path(sys.argv[1]), Path(sys.argv[2])
    s, f = probe(src); (w, h), rot = display_dims(s); aspect = w / h; dur = float(f["duration"])
    r, a = rate(s["r_frame_rate"]), rate(s["avg_frame_rate"])
    check("source is constant frame rate", r > 0 and abs(r - a) / r < 0.005,
          f"r={r:.3f} avg={a:.3f} (VFR means index/fps timestamps are wrong; use PTS)")
    vids = sorted(ep.glob("lerobot_v3/videos/**/*.mp4"))
    check("exported video present", bool(vids), f"{len(vids)} found")
    for v in vids:
        vs, _ = probe(v); (vw, vh), _ = display_dims(vs)
        check(f"video aspect {v.name}", abs(vw / vh - aspect) / aspect < 0.02,
              f"export {vw}x{vh} vs source display {w}x{h} rot={rot}")
    try:
        import h5py
        for p in list(ep.glob("*.hdf5")) + list(ep.glob("*.h5")):
            hits = []
            with h5py.File(p, "r") as hf:
                hf.visititems(lambda n, o: hits.append(n) if isinstance(o, h5py.Dataset)
                              and n.endswith("image") and o.ndim == 4 else None)
                check(f"hdf5 image present {p.name}", bool(hits), f"{len(hits)} image datasets")
                for n in hits:
                    ds = hf[n]; sh = ds.shape; frame = np.asarray(ds[sh[0] // 2])
                    arr_aspect = sh[2] / sh[1]
                    if abs(arr_aspect - aspect) / aspect < 0.02:
                        check(f"hdf5 aspect {p.name}:{n}", True, f"native aspect {arr_aspect:.3f}")
                    else:
                        ca = content_aspect(frame)
                        ok = ca is not None and abs(ca - aspect) / aspect < 0.03
                        check(f"hdf5 aspect {p.name}:{n}", ok,
                              f"array {sh[2]}x{sh[1]}; content aspect {ca if ca is None else round(ca,3)} vs source {aspect:.3f} "
                              f"({'letterboxed OK' if ok else 'SQUASHED/CROPPED: content aspect differs from source'})")
    except ImportError:
        check("h5py available", False, "not installed", warn=True)
    import pyarrow.parquet as pq
    files = sorted(ep.glob("lerobot_v3/data/**/*.parquet"))
    check("parquet present", bool(files), f"{len(files)} files")
    if files:
        ts = np.concatenate([pq.read_table(x, columns=["timestamp"]).column(0).to_numpy() for x in 
files])
        dt = np.diff(ts); m = dt.mean()
        check("timestamps strictly increasing", bool((dt > 0).all()), f"min dt={dt.min():.5f}")
        check("timestamps uniform", dt.std() / m < 0.01, f"std/mean={dt.std()/m:.4f}")
        check("export duration matches source", abs(ts[-1] + m - dur) < 0.1,
              f"export {ts[-1]+m:.3f}s vs source {dur:.3f}s (pass the CORRECT source video for this episode)")
        info = json.loads((ep / "lerobot_v3/meta/info.json").read_text())
        check("info.json fps matches data", abs(info.get("fps", 0) * m - 1) < 0.01,
              f"info fps={info.get('fps')} measured={1/m:.3f}")
    print("\nFAILS:", fails if fails else "none"); sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
