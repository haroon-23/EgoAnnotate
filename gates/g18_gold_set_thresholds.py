"""g18_gold_set_thresholds.py

Measures the pipeline against a HUMAN-labelled gold set (gold_set.json, schema in GATES_SCHEMA.md) and enforces thresholds.
Pipeline side is read from <episode_dir>/pipeline_eval_view.json (schema in GATES_SCHEMA.md).

Usage:  python gates/g18_gold_set_thresholds.py <episode_dir> [--gold PATH] [--thresholds JSON] [--min-counts JSON]
                                               [--allow-missing-episode] [--json-out PATH]        Exit: 0 PASS, 1 FAIL.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

GATE = "g18_gold_set_thresholds"
THRESH = dict(hand_recall=0.85, wrist_pck=0.85, chirality_acc=0.95, hand_precision=0.90, contact_precision=0.80, contact_recall=0.80,
              boundary_f1=0.70, caption_correct=0.80, head_coverage=0.95, match_dist=0.10, pck_dist=0.03, boundary_tol_s=0.5, head_lap_var=25.0)
MIN_COUNTS = dict(hands=200, contact=100, boundary_clips=20, caption_ratings=200, head_frames=200)


class R:
    def __init__(s): s.fails, s.warns, s.info = [], [], {}
    def check(s, name, ok, detail="", warn=False):
        print(f"  [{'PASS' if ok else ('WARN' if warn else 'FAIL')}] {name}: {detail}")
        if not ok: (s.warns if warn else s.fails).append(f"{name}: {detail}")
    def finish(s, out=None):
        st = "FAIL" if s.fails else "PASS"; print(f"G18 VERDICT: {st}")
        if out: Path(out).write_text(json.dumps(dict(gate=GATE, status=st, failures=s.fails, warnings=s.warns, info=s.info), indent=1, default=str))
        return 1 if s.fails else 0


def jload(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def count_gold(gold):
    c = dict(hands=0, contact=0, boundary_clips=0, caption_ratings=0, head_frames=0)
    for e in (gold.get("episodes") or {}).values():
        c["hands"] += sum(1 for f in e.get("hand_frames", []) for h in f.get("hands", []) if h.get("wearer"))
        c["contact"] += len(e.get("contact_frames", [])); c["boundary_clips"] += 1 if len(e.get("segment_boundaries_s", [])) >= 3 else 0
        c["caption_ratings"] += len(e.get("caption_ratings", [])); c["head_frames"] += len(e.get("head_frames", []))
    return c


def dist(a, b, size):
    return float(np.hypot((a[0] - b[0]) * size[0], (a[1] - b[1]) * size[1]) / max(size))


def eval_hands(g, view, size, T):
    by_frame = {}
    for f in view.get("frames", []): by_frame[int(f["frame_idx"])] = [h for h in f.get("hands", []) if h.get("valid")]
    gw = matched = pck = chir_ok = chir_n = pipe_total = pipe_good = 0
    for fr in g.get("hand_frames", []):
        idx = int(fr["frame_idx"]); gold = fr.get("hands", []); pipe = list(by_frame.get(idx, [])); pipe_total += len(pipe); used = set()
        pairs = sorted([(dist(gh["wrist_xy_norm"], ph["wrist_xy_norm"], size), gi, pi) for gi, gh in enumerate(gold) for pi, ph in enumerate(pipe)])
        gmatch = {}
        for d, gi, pi in pairs:
            if d <= T["match_dist"] and gi not in gmatch and pi not in used: gmatch[gi] = (pi, d); used.add(pi)
        for gi, gh in enumerate(gold):
            if gh.get("wearer"): gw += 1
            if gi in gmatch and gh.get("wearer"):
                pi, d = gmatch[gi]; matched += 1; pck += d <= T["pck_dist"]; chir_n += 1; chir_ok += (pipe[pi].get("side") == gh.get("side"))
        pipe_good += sum(1 for gi, (pi, d) in gmatch.items() if gold[gi].get("wearer"))
    return dict(wearer_hands=gw, recall=matched / gw if gw else None, pck=pck / gw if gw else None,
                chirality=chir_ok / chir_n if chir_n else None, precision=pipe_good / pipe_total if pipe_total else None, pipeline_hands=pipe_total)


def eval_contact(g, view):
    idx = {}
    for f in view.get("frames", []):
        for h in f.get("hands", []): idx[(int(f["frame_idx"]), h.get("side"))] = bool(h.get("contact"))
    tp = fp = fn = 0
    for c in g.get("contact_frames", []):
        pred = idx.get((int(c["frame_idx"]), c["hand"]), False); gold = bool(c["contact"])
        tp += gold and pred; fp += (not gold) and pred; fn += gold and not pred
    return dict(precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None, n=len(g.get("contact_frames", [])))


def eval_boundaries(g, view, tol):
    gb = sorted(g.get("segment_boundaries_s", [])); pb = sorted(s["start_s"] for s in view.get("segments", []))[1:]
    if not gb: return dict(f1=None)
    used = set(); tp = 0
    for x in gb:
        best = min(((abs(x - y), j) for j, y in enumerate(pb) if j not in used), default=None)
        if best and best[0] <= tol: tp += 1; used.add(best[1])
    p = tp / len(pb) if pb else 0.0; rc = tp / len(gb); return dict(f1=(2 * p * rc / (p + rc)) if p + rc else 0.0, precision=p, recall=rc)


def eval_captions(g, view):
    segs = {int(s["index"]): s for s in view.get("segments", [])}; ok = n = 0; stale = []
    for c in g.get("caption_ratings", []):
        s = segs.get(int(c["segment_index"]))
        if s is None or str(s.get("caption", "")).strip() != str(c.get("caption", "")).strip(): stale.append(c["segment_index"]); continue
        n += 1; ok += bool(c["correct"])
    return dict(correct=ok / n if n else None, rated=n, stale=stale)


def eval_heads(g, ep, T):
    import cv2
    vids = sorted((ep / "lerobot_v3" / "videos").rglob("*.mp4"))
    if not vids or not g.get("head_frames"): return dict(coverage=None, boxes=0)
    cap = cv2.VideoCapture(str(vids[0])); cov = tot = 0
    for hf in g["head_frames"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(hf["frame_idx"])); ok, fr = cap.read()
        if not ok: continue
        H, W = fr.shape[:2]
        for b in hf.get("heads_xyxy_norm", []):
            x0, y0, x1, y1 = int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)
            crop = fr[max(y0, 0):max(y1, 1), max(x0, 0):max(x1, 1)]
            if crop.size == 0: continue
            g64 = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64)); tot += 1; cov += cv2.Laplacian(g64, cv2.CV_64F).var() < T["head_lap_var"]
    cap.release()
    return dict(coverage=cov / tot if tot else None, boxes=tot)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episode_dir"); ap.add_argument("--gold", default="data/gold_set.json")
    ap.add_argument("--thresholds"); ap.add_argument("--min-counts"); ap.add_argument("--allow-missing-episode", action="store_true"); ap.add_argument("--json-out")
    a = ap.parse_args(); ep = Path(a.episode_dir); r = R(); T = dict(THRESH); M = dict(MIN_COUNTS)
    if a.thresholds: T.update(json.loads(a.thresholds))
    if a.min_counts: M.update(json.loads(a.min_counts))
    print(f"=== G18: GOLD-SET THRESHOLDS ({ep}) ===")
    gold = jload(a.gold); view = jload(ep / "pipeline_eval_view.json")
    r.check(f"gold set readable ({a.gold})", gold is not None, "")
    r.check("pipeline_eval_view.json readable", view is not None, "")
    if gold is None or view is None: return r.finish(a.json_out)
    r.check("gold set declares blind labelling", gold.get("blind") is True and gold.get("schema_version") == "ego_gold_v1", f"blind={gold.get('blind')} schema={gold.get('schema_version')}")
    cnt = count_gold(gold); r.info["gold_counts"] = cnt
    for k, need in M.items(): r.check(f"gold set size: {k}", cnt[k] >= need, f"{cnt[k]} (need >= {need})")
    eid = view.get("episode_id") or ep.name
    g = (gold.get("episodes") or {}).get(eid) or (gold.get("episodes") or {}).get(ep.name)
    if g is None:
        r.check(f"episode '{eid}' present in gold set", False, "episode not measured", warn=a.allow_missing_episode); return r.finish(a.json_out)
    size = g.get("image_size") or view.get("image_size"); r.check("image_size known", bool(size), str(size))
    if not size: return r.finish(a.json_out)
    h = eval_hands(g, view, size, T); c = eval_contact(g, view); b = eval_boundaries(g, view, T["boundary_tol_s"]); cap = eval_captions(g, view); hd = eval_heads(g, ep, T)
    r.info.update(hands=h, contact=c, boundaries=b, captions=cap, heads=hd)

    def chk(label, val, thr, n_ok=True):
        r.check(label, val is not None and val >= thr and n_ok, f"{'n/a (no labels)' if val is None else f'{val:.3f}'} (need >= {thr})")
    chk("wearer-hand recall", h["recall"], T["hand_recall"]); chk("wrist PCK", h["pck"], T["wrist_pck"])
    chk("chirality accuracy", h["chirality"], T["chirality_acc"]); chk("hand precision (bystander FPs)", h["precision"], T["hand_precision"])
    chk("contact precision", c["precision"], T["contact_precision"]); chk("contact recall", c["recall"], T["contact_recall"])
    chk("segment-boundary F1", b["f1"], T["boundary_f1"]); chk("caption correctness", cap["correct"], T["caption_correct"])
    r.check("caption ratings refer to current captions", not cap["stale"], f"stale ratings for segments {cap['stale']}" if cap["stale"] else "all current")
    chk("head blur coverage", hd["coverage"], T["head_coverage"])
    return r.finish(a.json_out)


if __name__ == "__main__":
    sys.exit(main())
