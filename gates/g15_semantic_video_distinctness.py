"""g15_semantic_video_distinctness.py

Every deliverable video must be decodable, non-empty, and *different in content in the way its name promises*.
Usage:  python gates/g15_semantic_video_distinctness.py <episode_dir> [--frames 12] [--json-out PATH]
"""
import sys, json, hashlib, argparse
from pathlib import Path
import numpy as np

GATE = "g15_semantic_video_distinctness"
PIXEL_DELTA = 25
MIN_CHANGED_OVERLAY = 0.002
MIN_CHANGED_PANEL = 0.30
MIN_CHANGED_BETWEEN = 0.05
WIDE_FACTOR = 1.4
MAX_DURATION_DIFF_S = 1.0


class R:
    def __init__(s): s.fails, s.warns, s.info = [], [], {}
    def check(s, name, ok, detail="", warn=False):
        print(f"  [{'PASS' if ok else ('WARN' if warn else 'FAIL')}] {name}: {detail}")
        if not ok: (s.warns if warn else s.fails).append(f"{name}: {detail}")
    def finish(s, out=None):
        st = "FAIL" if s.fails else "PASS"; print(f"G15 VERDICT: {st}")
        if out: Path(out).write_text(json.dumps(dict(gate=GATE, status=st, failures=s.fails, warnings=s.warns, info=s.info), indent=1, default=str))
        return 1 if s.fails else 0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def sample(path, n, size=(256, 144)):
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = []
    if total > 0:
        for i in np.linspace(0, max(total - 1, 0), n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
            if ok: frames.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA))
    cap.release()
    return dict(frames=frames, total=total, fps=fps, w=w, h=h, dur=(total / fps if fps else 0))


def changed_fraction(a, b):
    m = min(len(a["frames"]), len(b["frames"]))
    if m == 0: return None
    vals = [float((np.abs(a["frames"][i].astype(int) - b["frames"][i].astype(int)) > PIXEL_DELTA).mean()) for i in range(m)]
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episode_dir"); ap.add_argument("--frames", type=int, default=12); ap.add_argument("--json-out")
    a = ap.parse_args(); ep = Path(a.episode_dir); r = R()
    print(f"=== G15: SEMANTIC VIDEO DISTINCTNESS ({ep}) ===")
    try: import cv2
    except ImportError: r.check("opencv available", False, "pip install opencv-python"); return r.finish(a.json_out)
    train = sorted((ep / "lerobot_v3" / "videos").rglob("*.mp4"))
    others = {p.name: p for p in sorted(ep.glob("*.mp4"))}
    r.check("training export video present", bool(train), f"{len(train)} found")
    if not train: return r.finish(a.json_out)
    vids = {"training": train[0]}; vids.update({n: p for n, p in others.items()})
    data, hashes = {}, {}
    for name, p in vids.items():
        d = sample(p, a.frames); data[name] = d; hashes[name] = sha(p)
        black = bool(d["frames"]) and (np.mean([f.mean() for f in d["frames"]]) < 5 and np.mean([f.std() for f in d["frames"]]) < 3)
        r.check(f"decodable & non-empty: {name}", len(d["frames"]) >= max(2, a.frames // 2) and not black,
                f"frames={d['total']} {d['w']}x{d['h']} fps={d['fps']:.2f} sampled={len(d['frames'])} black={black}")
    names = list(vids)
    dup = [(x, y) for i, x in enumerate(names) for y in names[i + 1:] if hashes[x] == hashes[y]]
    r.check("no two deliverables are byte-identical", not dup, f"identical pairs: {dup}" if dup else "all distinct")
    t = data["training"]
    for name in names:
        if name == "training": continue
        d = data[name]; low = name.lower(); cf = changed_fraction(t, d)
        wide = t["w"] > 0 and d["w"] >= WIDE_FACTOR * t["w"]
        r.info[name] = dict(changed_fraction_vs_training=cf, width=d["w"], wide_vs_training=wide)
        if cf is None: r.check(f"{name} comparable with training", False, "no frames"); continue
        if "overlay" in low:
            r.check(f"{name} differs from training (annotations drawn)", cf >= MIN_CHANGED_OVERLAY, f"changed pixels={cf*100:.3f}% (min {MIN_CHANGED_OVERLAY*100:.1f}%)")
        elif "side_by_side" in low or "multi_embodiment" in low:
            r.check(f"{name} is a real panel video (wider or strongly different)", wide or cf >= MIN_CHANGED_PANEL,
                    f"width {d['w']} vs training {t['w']} (x{d['w']/max(t['w'],1):.2f}); changed pixels={cf*100:.1f}%")
        else:
            r.check(f"{name} differs from training", cf >= MIN_CHANGED_OVERLAY, f"changed pixels={cf*100:.3f}%")
        r.check(f"{name} duration matches training", abs(d["dur"] - t["dur"]) <= MAX_DURATION_DIFF_S, f"{d['dur']:.2f}s vs {t['dur']:.2f}s")
    sb = [n for n in names if "side_by_side" in n.lower()]; me = [n for n in names if "multi_embodiment" in n.lower()]
    if sb and me:
        cf = changed_fraction(data[sb[0]], data[me[0]])
        r.check("side_by_side and multi_embodiment_proof show different content", cf is not None and cf >= MIN_CHANGED_BETWEEN,
                f"changed pixels between them={None if cf is None else round(cf*100,2)}% (min {MIN_CHANGED_BETWEEN*100:.0f}%)")
    for want in ("overlay_annotated.mp4", "side_by_side.mp4", "multi_embodiment_proof.mp4"):
        r.check(f"deliverable present: {want}", want in others, "" if want in others else "missing", warn=True)
    return r.finish(a.json_out)


if __name__ == "__main__":
    sys.exit(main())
