"""g19_stale_artifact_chain.py

Detects stale or mismatched outputs. Every derived artifact must carry a provenance block proving it was computed from the
CURRENT annotations.json and the CURRENT code tree.
Usage:  python gates/g19_stale_artifact_chain.py <episode_dir> [--repo .] [--json-out PATH]       Exit: 0 PASS, 1 FAIL.
"""
import sys, json, hashlib, argparse, datetime
from pathlib import Path
import numpy as np

GATE = "g19_stale_artifact_chain"
SKEW_S = 2.0


class R:
    def __init__(s): s.fails, s.warns, s.info = [], [], {}
    def check(s, name, ok, detail="", warn=False):
        print(f"  [{'PASS' if ok else ('WARN' if warn else 'FAIL')}] {name}: {detail}")
        if not ok: (s.warns if warn else s.fails).append(f"{name}: {detail}")
    def finish(s, out=None):
        st = "FAIL" if s.fails else "PASS"; print(f"G19 VERDICT: {st}")
        if out: Path(out).write_text(json.dumps(dict(gate=GATE, status=st, failures=s.fails, warnings=s.warns, info=s.info), indent=1, default=str))
        return 1 if s.fails else 0


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def code_tree_sha256(repo):
    repo = Path(repo); files = []
    for top in ("src", "configs"):
        base = repo / top
        if not base.exists(): continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json") and "__pycache__" not in p.parts:
                files.append(p)
    d = hashlib.sha256()
    for p in sorted(files, key=lambda x: x.relative_to(repo).as_posix()):
        d.update(p.relative_to(repo).as_posix().encode() + b"\0" + file_sha(p).encode() + b"\n")
    return d.hexdigest()


def jload(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def parse_ts(s):
    try:
        t = datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if t.tzinfo is None: t = t.replace(tzinfo=datetime.timezone.utc)
        return t.timestamp()
    except Exception: return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episode_dir", nargs="?"); ap.add_argument("--repo", default="."); ap.add_argument("--print-code-hash", action="store_true")
    ap.add_argument("--json-out"); a = ap.parse_args()
    if a.print_code_hash: print(code_tree_sha256(a.repo)); return 0
    if not a.episode_dir: ap.error("episode_dir required")
    ep = Path(a.episode_dir); r = R(); print(f"=== G19: STALE-ARTIFACT CHAIN ({ep}) ===")
    ann = ep / "annotations.json"
    r.check("annotations.json present", ann.exists(), "")
    if not ann.exists(): return r.finish(a.json_out)
    ann_sha = file_sha(ann); ann_mtime = ann.stat().st_mtime; code_sha = code_tree_sha256(a.repo); now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    r.info.update(annotations_sha256=ann_sha, code_tree_sha256=code_sha)
    r.check("code tree found (src/ or configs/ under --repo)", code_sha != hashlib.sha256().hexdigest(), f"repo={Path(a.repo).resolve()}")

    def verify(label, prov):
        if not isinstance(prov, dict):
            r.check(f"{label}: provenance block", False, "missing"); return
        for k, want in (("annotations_sha256", ann_sha), ("code_tree_sha256", code_sha)):
            r.check(f"{label}: {k}", prov.get(k) == want, "matches current" if prov.get(k) == want else f"stale/missing (artifact={str(prov.get(k))[:12]} current={want[:12]})")
        ts = parse_ts(prov.get("generated_at_utc"))
        r.check(f"{label}: generated_at_utc valid and not before annotations.json", ts is not None and ts >= ann_mtime - SKEW_S and ts <= now + 60,
                f"generated={prov.get('generated_at_utc')} annotations_mtime={datetime.datetime.fromtimestamp(ann_mtime, datetime.timezone.utc).isoformat()}")

    q = jload(ep / "quality.json"); man = jload(ep / "manifest.json"); view = jload(ep / "pipeline_eval_view.json")
    cert_p = ep / "wam" / "roboverify_certificate.json"
    if not cert_p.exists():
        c = sorted(ep.rglob("*certificate*.json")); cert_p = c[0] if c else cert_p
    cert = jload(cert_p); hw = jload(ep / "hardware_readiness_report.json"); info = jload(ep / "lerobot_v3" / "meta" / "info.json")
    for label, obj, prov in [("quality.json", q, (q or {}).get("provenance")), ("manifest.json", man, (man or {}).get("provenance")),
                             ("certificate", cert, (cert or {}).get("provenance")), ("hardware_readiness_report.json", hw, (hw or {}).get("_provenance")),
                             ("pipeline_eval_view.json", view, (view or {}).get("provenance")),
                             ("lerobot info.json ego_annotate", info, ((info or {}).get("ego_annotate") or {}).get("provenance"))]:
        r.check(f"{label} present", obj is not None, "")
        if obj is not None: verify(label, prov)
    for tp in sorted((ep / "embodiments").glob("*/trajectory.npz")):
        try:
            z = np.load(tp, allow_pickle=False); prov = {k: str(z[k]) for k in ("annotations_sha256", "code_tree_sha256", "generated_at_utc") if k in z.files}
        except Exception as e: prov = None
        verify(f"trajectory {tp.parent.name}", prov)
    r.check("at least one trajectory.npz", bool(list((ep / "embodiments").glob("*/trajectory.npz"))), "")
    stale = []
    for p in list((ep / "lerobot_v3").rglob("*.parquet")) + list((ep / "lerobot_v3").rglob("*.mp4")) + list(ep.glob("*.mp4")):
        if p.stat().st_mtime < ann_mtime - SKEW_S: stale.append(p.name)
    r.check("no export older than annotations.json", not stale, f"stale files: {stale}" if stale else "all exports newer")
    return r.finish(a.json_out)


if __name__ == "__main__":
    sys.exit(main())
