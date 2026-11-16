"""g14_cross_report_parity.py

Every report an episode ships must agree with quality.json (the single source of truth).
Compares quality.json against: manifest.json, hardware_readiness_report.json,
wam/roboverify_certificate.json, lerobot_v3/meta/info.json (ego_annotate block),
consent_manifest.json. Also validates the quality.json schema (see GATES_SCHEMA.md).

Usage:  python gates/g14_cross_report_parity.py <episode_dir> [--json-out PATH]
Exit:   0 PASS, 1 FAIL.    Install into gates/ yourself; the coding agent must not edit it.
"""
import sys, json, re, argparse
from pathlib import Path

GATE = "g14_cross_report_parity"
TOL = dict(ratio=5e-4, pct=0.05, mm=0.005, deg=0.005, score=0.05, rel=0.01)
DEFAULT_ROBOTS = ["franka_panda", "unitree_g1_upper", "unitree_h1_upper", "allegro_hand_right"]


class R:
    def __init__(s): s.fails, s.warns, s.info = [], [], {}
    def check(s, name, ok, detail="", warn=False):
        print(f"  [{'PASS' if ok else ('WARN' if warn else 'FAIL')}] {name}: {detail}")
        if not ok: (s.warns if warn else s.fails).append(f"{name}: {detail}")
    def finish(s, out=None):
        st = "FAIL" if s.fails else "PASS"; print(f"G14 VERDICT: {st}")
        if out: Path(out).write_text(json.dumps(dict(gate=GATE, status=st, failures=s.fails, warnings=s.warns, info=s.info), indent=1, default=str))
        return 1 if s.fails else 0


def jload(p):
    try: return json.loads(Path(p).read_text())
    except Exception as e: return None


def dig(d, path):
    """dotted-path lookup; returns (found, value)"""
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur: cur = cur[k]
        else: return False, None
    return True, cur


def close(a, b, tol_kind):
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return a == b
    t = TOL[tol_kind]; d = abs(float(a) - float(b))
    if tol_kind in ("mm", "deg"): return d <= max(t, TOL["rel"] * max(abs(float(a)), abs(float(b))))
    return d <= t


QUALITY_REQUIRED = [
    ("schema_version", str), ("episode_id", str), ("provenance", dict),
    ("source_video", dict), ("hands", dict), ("segments", dict), ("privacy", dict),
    ("camera", dict), ("robots", dict), ("decision", dict), ("consent", dict),
]
HANDS_KEYS = ["frames", "detected_ratio", "rescued_ratio", "interpolated_ratio", "hand_valid_ratio", "left_ratio",
              "right_ratio", "untruncated_ratio", "either_present_ratio"]
ROBOT_KEYS = ["status", "ik_convergence_pct", "ik_convergence_def", "fk_pos_residual_mean_mm", "fk_pos_residual_p95_mm",
              "fk_ori_residual_p95_deg", "clamp_events", "self_collision_frames", "max_joint_vel_rad_s", "max_joint_jerk_rad_s3"]
DECISION_KEYS = ["accept_episode", "quality_tier", "overall_status", "failed_quality_checks"]
CAMERA_KEYS = ["device_id", "calibrated", "intrinsics_source", "distortion_model", "pose_source", "pose_tracked", "scale_source", "is_metric"]


def validate_quality(q, r, robots):
    for k, t in QUALITY_REQUIRED:
        ok = k in q and isinstance(q[k], t)
        r.check(f"quality.json has '{k}'", ok, "" if ok else f"missing or wrong type (need {t.__name__})")
    if not all(k in q for k, _ in QUALITY_REQUIRED): return False
    r.check("quality.json schema_version", q["schema_version"] == "ego_quality_v1", str(q["schema_version"]))
    for sec, keys in (("hands", HANDS_KEYS), ("decision", DECISION_KEYS), ("camera", CAMERA_KEYS)):
        miss = [k for k in keys if k not in q[sec]]
        r.check(f"quality.{sec} keys", not miss, f"missing {miss}" if miss else "complete")
    for rb in robots:
        if rb not in q["robots"]:
            r.check(f"quality.robots has {rb}", False, "missing"); continue
        miss = [k for k in ROBOT_KEYS if k not in q["robots"][rb]]
        r.check(f"quality.robots.{rb} keys", not miss, f"missing {miss}" if miss else "complete")
    for k in ("annotations_sha256", "code_tree_sha256", "generated_at_utc"):
        r.check(f"quality.provenance.{k}", k in q["provenance"], "")
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episode_dir"); ap.add_argument("--json-out")
    ap.add_argument("--robots", default=",".join(DEFAULT_ROBOTS)); a = ap.parse_args()
    ep = Path(a.episode_dir); robots = [x for x in a.robots.split(",") if x]
    r = R(); print(f"=== G14: CROSS-REPORT PARITY ({ep}) ===")
    q = jload(ep / "quality.json")
    if q is None:
        r.check("quality.json present and valid JSON", False, "missing: every report must derive from quality.json"); return r.finish(a.json_out)
    if not validate_quality(q, r, robots): return r.finish(a.json_out)

    man = jload(ep / "manifest.json"); hw = jload(ep / "hardware_readiness_report.json")
    cert_p = ep / "wam" / "roboverify_certificate.json"
    if not cert_p.exists():
        c = sorted(ep.rglob("*certificate*.json")); cert_p = c[0] if c else cert_p
    cert = jload(cert_p); info = jload(ep / "lerobot_v3" / "meta" / "info.json"); consent = jload(ep / "consent_manifest.json")
    for nm, obj in (("manifest.json", man), ("hardware_readiness_report.json", hw), ("certificate", cert), ("lerobot_v3/meta/info.json", info)):
        r.check(f"{nm} present", obj is not None, "")

    def cmp(label, qpath, other, opath, kind):
        fq, vq = dig(q, qpath)
        fo, vo = dig(other, opath) if other is not None else (False, None)
        if not fq or not fo:
            r.check(label, False, f"missing field (quality.{qpath}: {fq}, report.{opath}: {fo})"); return
        r.check(label, close(vq, vo, kind), f"quality={vq} report={vo}")

    primary = q.get("primary_robot", "franka_panda")
    if man is not None:
        for lab, qp, mp, kind, scale in [
            ("manifest hand_valid_ratio", "hands.hand_valid_ratio", "quality_metrics.hand_valid_ratio", "ratio", 1),
            ("manifest detected_hand_ratio", "hands.detected_ratio", "quality_metrics.detected_hand_ratio", "ratio", 1),
            ("manifest rescued_hand_ratio", "hands.rescued_ratio", "quality_metrics.rescued_hand_ratio", "ratio", 1),
            ("manifest interpolated_hand_ratio", "hands.interpolated_ratio", "quality_metrics.interpolated_hand_ratio", "ratio", 1),
            ("manifest left_hand_ratio", "hands.left_ratio", "quality_metrics.left_hand_ratio", "ratio", 1),
            ("manifest right_hand_ratio", "hands.right_ratio", "quality_metrics.right_hand_ratio", "ratio", 1),
            ("manifest mean_blurred_area_frac", "privacy.mean_blurred_area_frac", "quality_metrics.mean_blurred_area_frac", "ratio", 1),
            ("manifest face_detections_redacted", "privacy.face_detections_redacted", "quality_metrics.face_detections_redacted", "pct", 1),
        ]:
            cmp(lab, qp, man, mp, kind)
        fq, vq = dig(q, "hands.untruncated_ratio"); fo, vo = dig(man, "quality_metrics.untruncated_hand_visibility_pct")
        r.check("manifest untruncated_hand_visibility_pct", fq and fo and abs(vq * 100 - vo) <= TOL["pct"], f"quality={vq*100 if fq else None} report={vo}")
        for lab, rk, mp, kind in [("manifest ik_convergence_pct", "ik_convergence_pct", "quality_metrics.ik_convergence_pct", "pct"),
                                  ("manifest fk_pos_residual_p95_mm", "fk_pos_residual_p95_mm", "quality_metrics.fk_pos_residual_p95_mm", "mm"),
                                  ("manifest fk_ori_residual_p95_deg", "fk_ori_residual_p95_deg", "quality_metrics.fk_ori_residual_p95_deg", "deg")]:
            cmp(lab + f" ({primary})", f"robots.{primary}.{rk}", man, mp, kind)
        cmp("manifest accept_episode", "decision.accept_episode", man, "accept_episode", "pct")
        cmp("manifest overall_status", "decision.overall_status", man, "overall_status", "pct")
        cmp("manifest quality_tier", "decision.quality_tier", man, "quality_metrics.quality_tier", "pct")
        cmp("manifest failed_quality_checks", "decision.failed_quality_checks", man, "failed_quality_checks", "pct")
    if hw is not None:
        hw_robots = [k for k in hw if not k.startswith("_")]
        extra = sorted(set(hw_robots) - set(q["robots"])); lack = sorted(set(q["robots"]) - set(hw_robots))
        r.check("hardware report robot set == quality.robots", not extra and not lack, f"extra={extra} missing={lack}")
        for rb in q["robots"]:
            audit = (hw.get(rb) or {}).get("hardware_audit") or {}
            hwr = hw.get(rb) or {}
            for lab, rk, hk, kind in [("ik_convergence_pct", "ik_convergence_pct", "ik_convergence_pct", "pct"),
                                      ("fk_pos_residual_mean_mm", "fk_pos_residual_mean_mm", "mean_ik_residual_mm", "mm"),
                                      ("fk_pos_residual_p95_mm", "fk_pos_residual_p95_mm", "p95_ik_residual_mm", "mm"),
                                      ("fk_ori_residual_p95_deg", "fk_ori_residual_p95_deg", "p95_ori_residual_deg", "deg"),
                                      ("clamp_events", "clamp_events", "clamp_events", "pct"),
                                      ("self_collision_frames", "self_collision_frames", "self_collision_frames", "pct"),
                                      ("max_joint_vel_rad_s", "max_joint_vel_rad_s", "max_joint_velocity_rad_s", "mm"),
                                      ("max_joint_jerk_rad_s3", "max_joint_jerk_rad_s3", "max_joint_jerk_rad_s3", "mm")]:
                cmp(f"hw[{rb}] {lab}", f"robots.{rb}.{rk}", audit, hk, kind)
            vq = q["robots"][rb].get("status"); vstr = str(audit.get("verdict", ""))
            r.check(f"hw[{rb}] verdict starts with quality status", bool(vq) and vstr.startswith(str(vq)),
                    f"quality.status={vq} hw.verdict={vstr[:48]}")
            fo2, vo2 = "hand_valid_pct" in audit, audit.get("hand_valid_pct")
            r.check(f"hw[{rb}] hand_valid_pct", fo2 and abs(vo2 / 100.0 - q["hands"]["hand_valid_ratio"]) <= TOL["ratio"] * 2,
                    f"quality={q['hands']['hand_valid_ratio']*100:.3f}% hw={vo2}")
    if cert is not None:
        l1 = next((v for k, v in cert.items() if k.startswith("layer_1") and isinstance(v, dict)), {})
        v = l1.get("untruncated_hand_visibility_pct")
        r.check("certificate untruncated_hand_visibility_pct", v is not None and abs(q["hands"]["untruncated_ratio"] * 100 - v) <= TOL["pct"],
                f"quality={q['hands']['untruncated_ratio']*100:.3f} cert={v}")
        verdict = str(cert.get("overall_qc_verdict", "")); acc = q["decision"]["accept_episode"]
        positive = bool(re.search(r"VERIFIED|CERTIFIED|PASS|OK", verdict)) and not re.search(r"REJECT|REVIEW|LOW_RES|FAIL", verdict)
        r.check("certificate verdict consistent with decision", positive == bool(acc), f"accept={acc} verdict={verdict[:60]}")
    if info is not None:
        ea = info.get("ego_annotate") or {}
        for lab, qp, ik in [("device_id", "camera.device_id", "device_id"), ("calibrated", "camera.calibrated", "calibrated"),
                            ("is_metric", "camera.is_metric", "is_metric"), ("scale_source", "camera.scale_source", "scale_source"),
                            ("consent_status", "consent.consent_status", "consent_status")]:
            cmp(f"info.json ego_annotate.{lab}", qp, ea, ik, "pct")
    if consent is not None:
        cmp("consent_manifest consent_status", "consent.consent_status", consent, "consent_status", "pct")
    d = q["decision"]; tier, acc, st, fl = d["quality_tier"], d["accept_episode"], d["overall_status"], d["failed_quality_checks"]
    exp = {"ACCEPT": (True, "accepted"), "REVIEW": (False, "review_required"), "REJECT": (False, "rejected")}
    ok = tier in exp and acc == exp[tier][0] and st == exp[tier][1]
    r.check("decision fields mutually consistent", ok, f"tier={tier} accept={acc} status={st}")
    r.check("failed_quality_checks empty iff accepted", (len(fl) == 0) == bool(acc), f"accept={acc} failed={fl}")
    return r.finish(a.json_out)


if __name__ == "__main__":
    sys.exit(main())
