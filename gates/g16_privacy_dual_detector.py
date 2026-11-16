"""g16_privacy_dual_detector.py

Privacy check on EVERY exported video of an episode (training export, overlay, side_by_side, multi_embodiment_proof, ...).
Usage:  python gates/g16_privacy_dual_detector.py <episode_dir> [--source VIDEO] [--stride 15] [--dump 100]
                                                  [--out-dir DIR] [--allow-single-detector] [--json-out PATH]
"""
import sys, json, argparse, random
from pathlib import Path
import numpy as np

GATE = "g16_privacy_dual_detector"
DETECT_WIDTH = 960
MAX_FRAMES_PER_VIDEO = 240
MAX_SUSPECT_RATE = 0.01
MAX_MEAN_BLUR_DELTA = 0.10
MAX_PEAK_BLUR_DELTA = 0.40
LOWDETAIL_VAR = 20.0


class R:
    def __init__(s): s.fails, s.warns, s.info = [], [], {}
    def check(s, name, ok, detail="", warn=False):
        print(f"  [{'PASS' if ok else ('WARN' if warn else 'FAIL')}] {name}: {detail}")
        if not ok: (s.warns if warn else s.fails).append(f"{name}: {detail}")
    def finish(s, out=None):
        st = "FAIL" if s.fails else "PASS"; print(f"G16 VERDICT: {st}")
        if out: Path(out).write_text(json.dumps(dict(gate=GATE, status=st, failures=s.fails, warnings=s.warns, info=s.info), indent=1, default=str))
        return 1 if s.fails else 0


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0); ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def agree(a, b):
    if iou(a, b) >= 0.2: return True
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2); cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    inside = lambda c, bx: bx[0] <= c[0] <= bx[2] and bx[1] <= c[1] <= bx[3]
    return inside(ca, b) or inside(cb, a)


class Detectors:
    def __init__(self, allow_single):
        import cv2
        self.cv2 = cv2; self.mp = None; self.msg = []
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self.mp = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            else: self.msg.append("mediapipe legacy solutions API not available in this version")
        except Exception as e: self.msg.append(f"mediapipe unavailable: {e!r}")
        self.front = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.prof = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        self.haar_ok = not self.front.empty() and not self.prof.empty()
        if not self.haar_ok: self.msg.append("Haar cascades missing")
        self.ok = (self.mp is not None and self.haar_ok) or (allow_single and (self.mp is not None or self.haar_ok))

    def detect(self, bgr):
        cv2 = self.cv2; h, w = bgr.shape[:2]; sc = DETECT_WIDTH / w if w > DETECT_WIDTH else 1.0
        im = cv2.resize(bgr, (int(w * sc), int(h * sc))) if sc != 1.0 else bgr; H, W = im.shape[:2]
        mpb, hb = [], []
        if self.mp is not None:
            res = self.mp.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            for d in (res.detections or []):
                bb = d.location_data.relative_bounding_box
                mpb.append((bb.xmin * W, bb.ymin * H, (bb.xmin + bb.width) * W, (bb.ymin + bb.height) * H))
        if self.haar_ok:
            g = cv2.equalizeHist(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            for casc in (self.front, self.prof):
                for (x, y, ww, hh) in casc.detectMultiScale(g, 1.1, 5, minSize=(40, 40)):
                    hb.append((x, y, x + ww, y + hh))
            gf = cv2.flip(g, 1)
            for (x, y, ww, hh) in self.prof.detectMultiScale(gf, 1.1, 5, minSize=(40, 40)):
                hb.append((W - (x + ww), y, W - x, y + hh))
        confirmed, suspects = [], []
        for m in mpb:
            (confirmed if any(agree(m, h_) for h_ in hb) else suspects).append(m)
        for h_ in hb:
            if not any(agree(h_, m) for m in mpb): suspects.append(h_)
        return confirmed, suspects, sc


def lowdetail_fraction(bgr, b=32):
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); g = cv2.resize(g, (640, max(int(g.shape[0] * 640 / g.shape[1]), b)))
    L = cv2.Laplacian(g, cv2.CV_64F); n = m = 0
    for y in range(0, g.shape[0] - b + 1, b):
        for x in range(0, g.shape[1] - b + 1, b):
            n += 1; m += L[y:y + b, x:x + b].var() < LOWDETAIL_VAR
    return m / max(n, 1)


def frame_at(cap, idx):
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, fr = cap.read(); return fr if ok else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episode_dir"); ap.add_argument("--source"); ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--dump", type=int, default=100); ap.add_argument("--out-dir"); ap.add_argument("--allow-single-detector", action="store_true")
    ap.add_argument("--max-suspect-rate", type=float, default=MAX_SUSPECT_RATE); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--json-out")
    a = ap.parse_args(); ep = Path(a.episode_dir); r = R(); random.seed(a.seed)
    print(f"=== G16: PRIVACY, DUAL DETECTOR ({ep}) ===")
    try: import cv2
    except ImportError: r.check("opencv available", False, "pip install opencv-python"); return r.finish(a.json_out)
    det = Detectors(a.allow_single_detector)
    r.check("two independent detectors available", det.ok, "; ".join(det.msg) or "mediapipe + haar ready")
    if not det.ok: return r.finish(a.json_out)
    videos = sorted(set(list((ep / "lerobot_v3" / "videos").rglob("*.mp4")) + list(ep.glob("*.mp4"))))
    r.check("exported videos found", bool(videos), f"{len(videos)} videos")
    if not videos: return r.finish(a.json_out)
    out_dir = Path(a.out_dir) if a.out_dir else ep / "g16_review_frames"; out_dir.mkdir(parents=True, exist_ok=True)
    pool = []; per_video = {}
    for v in videos:
        cap = cv2.VideoCapture(str(v)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idxs = list(range(0, total, max(a.stride, 1)))
        if len(idxs) > MAX_FRAMES_PER_VIDEO: idxs = [int(i) for i in np.linspace(0, total - 1, MAX_FRAMES_PER_VIDEO)]
        conf_n = sus_n = 0; flagged = []
        for i in idxs:
            fr = frame_at(cap, i)
            if fr is None: continue
            c, s, sc = det.detect(fr)
            conf_n += len(c); sus_n += (1 if s else 0)
            pool.append((v.name, i, fr, c, s, sc))
            if c: flagged.append(i)
        cap.release()
        rate = sus_n / max(len(idxs), 1)
        per_video[v.name] = dict(frames_sampled=len(idxs), confirmed_faces=conf_n, suspect_frames=sus_n, suspect_rate=rate, confirmed_frames=flagged[:20])
        r.check(f"no confirmed (two-detector) faces: {v.name}", conf_n == 0, f"sampled={len(idxs)} confirmed={conf_n} frames={flagged[:8]}")
        r.check(f"suspect rate <= {a.max_suspect_rate:.0%}: {v.name}", rate <= a.max_suspect_rate, f"suspect frames={sus_n}/{len(idxs)} ({rate:.1%})")
    r.info["per_video"] = per_video
    src = Path(a.source) if a.source else None
    if src is None:
        mj = None
        try: mj = json.loads((ep / "manifest.json").read_text()).get("video_path")
        except Exception: pass
        if mj:
            for base in (Path.cwd(), ep, ep.parent.parent):
                if (base / mj).exists(): src = base / mj; break
    train = sorted((ep / "lerobot_v3" / "videos").rglob("*.mp4"))
    if src is not None and src.exists() and train:
        cs, ct = cv2.VideoCapture(str(src)), cv2.VideoCapture(str(train[0]))
        ns, nt = int(cs.get(cv2.CAP_PROP_FRAME_COUNT) or 0), int(ct.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        deltas = []
        for k in np.linspace(0, 1, 16):
            fs, ft = frame_at(cs, int(k * max(ns - 1, 0))), frame_at(ct, int(k * max(nt - 1, 0)))
            if fs is not None and ft is not None: deltas.append(lowdetail_fraction(ft) - lowdetail_fraction(fs))
        cs.release(); ct.release()
        if deltas:
            r.info["blur_delta"] = dict(mean=float(np.mean(deltas)), max=float(np.max(deltas)))
            r.check("over-blur: mean added low-detail area", np.mean(deltas) <= MAX_MEAN_BLUR_DELTA, f"mean +{np.mean(deltas)*100:.1f} pts (limit {MAX_MEAN_BLUR_DELTA*100:.0f})")
            r.check("over-blur: worst frame added low-detail area", np.max(deltas) <= MAX_PEAK_BLUR_DELTA, f"max +{np.max(deltas)*100:.1f} pts (limit {MAX_PEAK_BLUR_DELTA*100:.0f})")
    else:
        r.check("source video found for over-blur check", False, "pass --source or provide manifest.json video_path")
    random.shuffle(pool); sel = pool[:a.dump]; thumbs = []
    for k, (name, i, fr, c, s, sc) in enumerate(sel):
        img = fr.copy(); inv = 1.0 / sc if sc else 1.0
        for bx, col in [(b, (0, 255, 0)) for b in c] + [(b, (0, 0, 255)) for b in s]:
            cv2.rectangle(img, (int(bx[0] * inv), int(bx[1] * inv)), (int(bx[2] * inv), int(bx[3] * inv)), col, 3)
        h, w = img.shape[:2]; img = cv2.resize(img, (640, int(h * 640 / w)))
        cv2.putText(img, f"{name}:{i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"review_{k:03d}_{Path(name).stem}_{i:05d}.jpg"), img); thumbs.append(cv2.resize(img, (192, 108)))
    if thumbs:
        cols = 10; rows = (len(thumbs) + cols - 1) // cols; sheet = np.zeros((rows * 108, cols * 192, 3), np.uint8)
        for k, t in enumerate(thumbs): sheet[(k // cols) * 108:(k // cols + 1) * 108, (k % cols) * 192:(k % cols + 1) * 192] = t
        cv2.imwrite(str(out_dir / "review_sheet.jpg"), sheet)
    r.check("review frames dumped", len(thumbs) >= min(a.dump, len(pool)) and len(thumbs) > 0, f"{len(thumbs)} frames -> {out_dir}")
    print(f"  >>> HUMAN REVIEW REQUIRED: open {out_dir}/review_sheet.jpg and the review_*.jpg frames. "
          f"Green boxes = both detectors agree (face), red = single-detector suspect. Faces the detectors miss are only visible to a human.")
    return r.finish(a.json_out)


if __name__ == "__main__":
    sys.exit(main())
