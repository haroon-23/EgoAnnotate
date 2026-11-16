"""g21_gate_selftest.py

A gate that cannot fail is not a gate. This self-test builds a synthetic, fully consistent fixture (3 episodes, 4 robots, videos,
parquet, reports, gold set), proves g14-g20 PASS on the pristine fixture, then applies deliberately broken mutations and proves the
targeted gate FAILS for the EXPECTED reason. It also checks the FK engine against an analytic forward-kinematics formula.

Usage:  python gates/g21_gate_selftest.py [--only g14,g17] [--keep] [--json-out PATH]
Needs:  ffmpeg on PATH, opencv-python, numpy, pyarrow, pybullet, scikit-image (face fixture for g16; install with: pip install scikit-image).
        A mutation that cannot be run (missing dependency) counts as FAIL: the gate is then unverified.
Exit:   0 = every gate passed the good fixture and failed every mutation, 1 otherwise.
"""
import sys, os, json, shutil, subprocess, tempfile, argparse, datetime, importlib.util
from pathlib import Path
import numpy as np

GATE = "g21_gate_selftest"
HERE = Path(__file__).resolve().parent
ROBOTS = ["franka_panda", "unitree_g1_upper", "unitree_h1_upper", "allegro_hand_right"]
LINKS = {"franka_panda": (0.30, 0.30, 0.12), "unitree_g1_upper": (0.28, 0.25, 0.10), "unitree_h1_upper": (0.33, 0.27, 0.11), "allegro_hand_right": (0.12, 0.10, 0.08)}
SIGMA_MM = {"franka_panda": 0.8, "unitree_g1_upper": 1.0, "unitree_h1_upper": 1.3, "allegro_hand_right": 1.6}
T, FPS, W, H = 90, 30.0, 320, 180
STATUS = {"franka_panda": "VERIFIED_KINEMATIC"}


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def ffmpeg(args):
    subprocess.run(["ffmpeg", "-v", "error", "-y"] + args, check=True)


def qmul(a, b):
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def urdf_text(name, L):
    link = lambda n: f'<link name="{n}"><inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>'
    jt = lambda n, par, ch, xyz, ax: (f'<joint name="{n}" type="revolute"><parent link="{par}"/><child link="{ch}"/><origin xyz="{xyz}"/><axis xyz="{ax}"/>'
                                      f'<limit lower="-2.8" upper="2.8" effort="10" velocity="3.0"/></joint>')
    return (f'<robot name="{name}">' + "".join(link(n) for n in ("base", "l1", "l2", "l3", "tip")) + jt("j1", "base", "l1", "0 0 0.1", "0 0 1")
            + jt("j2", "l1", "l2", f"{L[0]} 0 0", "0 1 0") + jt("j3", "l2", "l3", f"{L[1]} 0 0", "0 1 0")
            + f'<joint name="jt" type="fixed"><parent link="l3"/><child link="tip"/><origin xyz="{L[2]} 0 0"/></joint></robot>')


def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def analytic_fk(q, L):
    R = Rz(q[0]); p = np.array([0, 0, 0.1]); p = p + R @ np.array([L[0], 0, 0]); R = R @ Ry(q[1]); p = p + R @ np.array([L[1], 0, 0]); R = R @ Ry(q[2]); p = p + R @ np.array([L[2], 0, 0]); return p, R


def mat_angle_to_quat_deg(R, quat_xyzw):
    x, y, z, w = quat_xyzw
    Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return float(np.degrees(np.arccos(np.clip((np.trace(R.T @ Rq) - 1) / 2, -1, 1))))


def now_iso(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_episode(root, name, seed, g19, g20):
    import pyarrow as pa, pyarrow.parquet as pq
    rng = np.random.default_rng(seed); repo = root / "repo"; ep = root / "out" / name; ep.mkdir(parents=True, exist_ok=True)
    (repo / "data" / "raw_videos").mkdir(parents=True, exist_ok=True)
    src = repo / "data" / "raw_videos" / f"{name}.mp4"
    ffmpeg(["-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=30:duration=3", "-pix_fmt", "yuv420p", "-c:v", "libx264", str(src)])
    ann = ep / "annotations.json"; ann.write_text(json.dumps({"episode_id": name, "frames": [{"frame_idx": i} for i in range(T)]}))
    sha_a = g19.file_sha(ann); sha_c = g19.code_tree_sha256(repo); prov = lambda: dict(annotations_sha256=sha_a, code_tree_sha256=sha_c, generated_at_utc=now_iso())
    # videos
    tdir = ep / "lerobot_v3" / "videos" / "observation.images.ego" / "chunk-000"; tdir.mkdir(parents=True, exist_ok=True); train = tdir / "file-000.mp4"
    ffmpeg(["-i", str(src), "-filter_complex", "[0:v]split[a][b];[b]crop=60:50:20:20,boxblur=10:2[c];[a][c]overlay=20:20", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-an", str(train)])
    ffmpeg(["-i", str(train), "-vf", "drawbox=x=150:y=10:w=120:h=60:color=red:t=fill", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-an", str(ep / "overlay_annotated.mp4")])
    ffmpeg(["-i", str(train), "-i", str(train), "-filter_complex", "[1:v]negate[n];[0:v][n]hstack", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-an", str(ep / "side_by_side.mp4")])
    ffmpeg(["-i", str(train), "-f", "lavfi", "-i", f"smptebars=size={W}x{H}:rate=30:duration=3", "-filter_complex", "[0:v][1:v]hstack", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-an", str(ep / "multi_embodiment_proof.mp4")])
    # robots
    (repo / "urdfs").mkdir(parents=True, exist_ok=True); robots_q = {}; hw = {}
    t = np.arange(T) / (T - 1)
    for k, rb in enumerate(ROBOTS):
        up = repo / "urdfs" / f"{rb}.urdf"; up.write_text(urdf_text(rb, LINKS[rb]))
        ph = rng.uniform(0, 6.28, 3); q = np.stack([0.7 * np.sin(2 * np.pi * t + ph[0]), 0.5 * np.sin(2 * np.pi * t + ph[1]), 0.4 * np.sin(2 * np.pi * t + ph[2])], 1)
        base_pos, base_q = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])
        pos, quat, _, _ = g20.fk_traj(up, ["j1", "j2", "j3"], "tip", base_pos, base_q, q)
        sig = SIGMA_MM[rb] * rng.uniform(0.7, 1.4) / 1000.0
        tpos = pos + rng.normal(0, sig, pos.shape); tq = []
        for i in range(T):
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax); ang = np.radians(rng.normal(0, 0.4)); dq = np.r_[ax * np.sin(ang / 2), np.cos(ang / 2)]
            tq.append(qmul(dq, quat[i]))
        valid = rng.random(T) < 0.9
        epd = ep / "embodiments" / rb; epd.mkdir(parents=True, exist_ok=True); pv = prov()
        np.savez(epd / "trajectory.npz", q=q, joint_names=np.array(["j1", "j2", "j3"]), valid_mask=valid, target_ee_pos=tpos, target_ee_quat_xyzw=np.array(tq),
                 urdf_path=np.array(f"urdfs/{rb}.urdf"), ee_link=np.array("tip"), base_pos=base_pos, base_quat_xyzw=base_q, fps=np.array(FPS),
                 annotations_sha256=np.array(pv["annotations_sha256"]), code_tree_sha256=np.array(pv["code_tree_sha256"]), generated_at_utc=np.array(pv["generated_at_utc"]))
        m = g20.recompute(np.load(epd / "trajectory.npz"), repo); df = dict(pos_tol_mm=4.0, ori_tol_deg=2.0)
        conv = 100.0 * float(np.mean((m["pos_err"] <= df["pos_tol_mm"]) & (m["ori_err"] <= df["ori_tol_deg"])))
        robots_q[rb] = dict(status=STATUS.get(rb, "EXPERIMENTAL_UNVERIFIED"), ik_convergence_pct=conv, ik_convergence_def=df, fk_pos_residual_mean_mm=m["pos_mean"],
                            fk_pos_residual_p95_mm=m["pos_p95"], fk_ori_residual_p95_deg=m["ori_p95"], clamp_events=0, self_collision_frames=0,
                            max_joint_vel_rad_s=m["max_vel"], max_joint_jerk_rad_s3=m["max_jerk"])
    hands = dict(frames=T, detected_ratio=0.80, rescued_ratio=0.05, interpolated_ratio=0.03, hand_valid_ratio=0.85, left_ratio=0.55, right_ratio=0.90, untruncated_ratio=0.88, either_present_ratio=0.88)
    privacy = dict(face_detections_redacted=12, mean_blurred_area_frac=0.02, max_blurred_area_frac=0.05, audio_streams_in_exports=0)
    cam = dict(device_id=f"assumed_pinhole_{W}x{H}", calibrated=False, intrinsics_source="assumed", distortion_model="none", pose_source="none", pose_tracked=False, scale_source="assumed", is_metric=False)
    dec = dict(accept_episode=True, quality_tier="ACCEPT", overall_status="accepted", failed_quality_checks=[])
    consent = dict(consent_status="UNSET_INTERNAL_TEST", distribution_restriction="NOT_FOR_DISTRIBUTION")
    quality = dict(schema_version="ego_quality_v1", episode_id=name, primary_robot="franka_panda", provenance=prov(),
                   source_video=dict(path=f"data/raw_videos/{name}.mp4", width=W, height=H, fps=FPS, duration_s=T / FPS), hands=hands,
                   segments=dict(count=4, min_s=0.75, max_s=0.75, vlm_fraction=1.0, template_fraction=0.0), privacy=privacy, camera=cam, robots=robots_q, decision=dec, consent=consent)
    (ep / "quality.json").write_text(json.dumps(quality, indent=1))
    fr = robots_q["franka_panda"]
    (ep / "manifest.json").write_text(json.dumps(dict(episode_id=name, overall_status="accepted", accept_episode=True, failed_quality_checks=[], video_path=f"data/raw_videos/{name}.mp4",
        provenance=prov(), quality_metrics=dict(hand_valid_ratio=0.85, detected_hand_ratio=0.80, rescued_hand_ratio=0.05, interpolated_hand_ratio=0.03, left_hand_ratio=0.55,
        right_hand_ratio=0.90, mean_blurred_area_frac=0.02, face_detections_redacted=12, untruncated_hand_visibility_pct=88.0, ik_convergence_pct=fr["ik_convergence_pct"],
        fk_pos_residual_p95_mm=fr["fk_pos_residual_p95_mm"], fk_ori_residual_p95_deg=fr["fk_ori_residual_p95_deg"], quality_tier="ACCEPT"))))
    hwr = {}
    for rb, v in robots_q.items():
        hwr[rb] = dict(capability_status=v["status"], dof=3, hardware_audit=dict(verdict=f"{v['status']} (fixture)", hardware_readiness_score=90.0, hand_valid_pct=85.0,
            ik_convergence_pct=v["ik_convergence_pct"], mean_ik_residual_mm=v["fk_pos_residual_mean_mm"], p95_ik_residual_mm=v["fk_pos_residual_p95_mm"],
            p95_ori_residual_deg=v["fk_ori_residual_p95_deg"], clamp_events=0, self_collision_frames=0, max_joint_velocity_rad_s=v["max_joint_vel_rad_s"],
            max_joint_jerk_rad_s3=v["max_joint_jerk_rad_s3"]))
    hwr["_provenance"] = prov(); (ep / "hardware_readiness_report.json").write_text(json.dumps(hwr))
    (ep / "wam").mkdir(exist_ok=True)
    (ep / "wam" / "roboverify_certificate.json").write_text(json.dumps(dict(overall_qc_verdict="VERIFIED_OK", provenance=prov(), layer_1_hand_qc=dict(untruncated_hand_visibility_pct=88.0, status="PASS"))))
    (ep / "consent_manifest.json").write_text(json.dumps(consent))
    # lerobot metadata + parquet
    st = rng.normal(size=(T, 24)).astype(np.float32); ac = rng.normal(size=(T, 24)).astype(np.float32)
    tbl = pa.table({"observation.state": pa.array(st.tolist(), pa.list_(pa.float32())), "action": pa.array(ac.tolist(), pa.list_(pa.float32())),
                    "timestamp": pa.array((np.arange(T) / FPS).astype(np.float32)), "frame_index": pa.array(np.arange(T)), "episode_index": pa.array(np.zeros(T, dtype=np.int64)),
                    "observation.hand_valid": pa.array(np.ones(T, dtype=np.int64))})
    (ep / "lerobot_v3" / "data" / "chunk-000").mkdir(parents=True, exist_ok=True); pq.write_table(tbl, ep / "lerobot_v3" / "data" / "chunk-000" / "file-000.parquet")
    (ep / "lerobot_v3" / "meta").mkdir(parents=True, exist_ok=True)
    (ep / "lerobot_v3" / "meta" / "info.json").write_text(json.dumps(dict(codebase_version="v3.0", fps=30, features={"observation.state": dict(dtype="float32", shape=[24], names=[f"s{i}" for i in range(24)]),
        "action": dict(dtype="float32", shape=[24], names=[f"a{i}" for i in range(24)])}, ego_annotate=dict(device_id=cam["device_id"], calibrated=False, is_metric=False, scale_source="assumed",
        consent_status=consent["consent_status"], units=dict(wrist_pos="assumed_m"), provenance=prov()))))
    # eval view + gold
    frames, gframes, contacts = [], [], []
    for i in range(T):
        lx, ly, rx, ry = 0.30 + 0.05 * np.sin(i / 10), 0.60, 0.70, 0.60 + 0.05 * np.cos(i / 10); cl, cr = (i // 10) % 2 == 0, (i // 15) % 2 == 0
        frames.append(dict(frame_idx=i, hands=[dict(side="left", valid=True, source="detected", wrist_xy_norm=[lx, ly], contact=bool(cl), grasp="none"),
                                              dict(side="right", valid=True, source="detected", wrist_xy_norm=[rx, ry], contact=bool(cr), grasp="none")]))
        hands_g = [dict(side="left", wearer=True, wrist_xy_norm=[lx + 0.004, ly]), dict(side="right", wearer=True, wrist_xy_norm=[rx, ry - 0.004])]
        if i % 3 == 0: hands_g.append(dict(side="left", wearer=False, wrist_xy_norm=[0.5, 0.15]))
        gframes.append(dict(frame_idx=i, hands=hands_g)); contacts += [dict(frame_idx=i, hand="left", contact=bool(cl)), dict(frame_idx=i, hand="right", contact=bool(cr))]
    segs = [dict(index=k, start_s=0.75 * k, end_s=0.75 * (k + 1), caption=f"person does step {k} of the task", caption_source="vlm") for k in range(4)]
    (ep / "pipeline_eval_view.json").write_text(json.dumps(dict(schema_version="ego_eval_view_v1", episode_id=name, image_size=[W, H], fps=FPS, provenance=prov(), frames=frames, segments=segs)))
    return dict(episode_id=name, image_size=[W, H], hand_frames=gframes, contact_frames=contacts, segment_boundaries_s=[0.75, 1.5, 2.25],
                caption_ratings=[dict(segment_index=s["index"], caption=s["caption"], correct=True) for s in segs],
                head_frames=[dict(frame_idx=i, heads_xyxy_norm=[[20 / W, 20 / H, 80 / W, 70 / H]]) for i in (5, 20, 40, 60, 80)])


def build_root(root, g19, g20):
    root.mkdir(parents=True, exist_ok=True); (root / "repo" / "src").mkdir(parents=True, exist_ok=True); (root / "repo" / "configs").mkdir(parents=True, exist_ok=True)
    (root / "repo" / "src" / "pipeline.py").write_text("VERSION = 1\n"); (root / "repo" / "configs" / "default.yaml").write_text("fps: 30\n")
    gold = dict(schema_version="ego_gold_v1", labeler="fixture", blind=True, created_utc=now_iso(), episodes={})
    for k, n in enumerate(("ep_a", "ep_b", "ep_c")): gold["episodes"][n] = build_episode(root, n, 100 + k, g19, g20)
    (root / "gold_set.json").write_text(json.dumps(gold))
    return root


def run_gate(gate_file, ep, root, extra=()):
    out = root / f"_{gate_file}.json"
    cmd = [sys.executable, str(HERE / gate_file), str(ep)] + list(extra) + ["--json-out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    fails = []
    if out.exists():
        try: fails = json.loads(out.read_text()).get("failures", [])
        except Exception: pass
    if p.returncode not in (0, 1) or not out.exists(): fails.append("CRASH: " + (p.stderr or p.stdout)[-400:])
    return p.returncode, fails


def jl(p): return json.loads(Path(p).read_text())
def jw(p, d): Path(p).write_text(json.dumps(d))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--keep", action="store_true"); ap.add_argument("--json-out"); a = ap.parse_args()
    only = {x.strip() for x in a.only.split(",") if x.strip()}; results = []; problems = []
    print("=== G21: GATE SELF-TEST ===")
    for tool in ("ffmpeg",):
        if shutil.which(tool) is None: print(f"  [FAIL] {tool} not found"); print("G21 VERDICT: FAIL"); return 1
    try: import pybullet, cv2, pyarrow  # noqa
    except ImportError as e: print(f"  [FAIL] missing dependency: {e}"); print("G21 VERDICT: FAIL"); return 1
    g19, g20 = load_mod("g19_stale_artifact_chain"), load_mod("g20_four_robot_independent")
    tmp = Path(tempfile.mkdtemp(prefix="g21_"))
    try:
        # FK engine vs analytic formula
        up = tmp / "fk.urdf"; up.write_text(urdf_text("fk", LINKS["franka_panda"])); rngx = np.random.default_rng(1); qq = rngx.uniform(-1.5, 1.5, (20, 3))
        pos, quat, _, _ = g20.fk_traj(up, ["j1", "j2", "j3"], "tip", [0, 0, 0], [0, 0, 0, 1], qq)
        dp = max(np.linalg.norm(pos[i] - analytic_fk(qq[i], LINKS["franka_panda"])[0]) for i in range(20)) * 1000
        do = max(mat_angle_to_quat_deg(analytic_fk(qq[i], LINKS["franka_panda"])[1], quat[i]) for i in range(20))
        ok = dp < 0.01 and do < 0.05; results.append(("fk-engine", "analytic FK agrees with pybullet", ok, f"max pos diff {dp:.5f} mm, max ori diff {do:.5f} deg"))
        print(f"  [{'PASS' if ok else 'FAIL'}] FK engine vs analytic formula: pos {dp:.5f} mm ori {do:.5f} deg")
        if not ok: problems.append("FK engine disagrees with analytic FK")
        base = build_root(tmp / "base", g19, g20); gold = base / "gold_set.json"
        mp_ok = False
        try:
            import mediapipe as mp; mp_ok = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection")
        except Exception: pass
        g16x = ["--stride", "10", "--dump", "12"] + ([] if mp_ok else ["--allow-single-detector"])
        if not mp_ok: print("  NOTE: mediapipe legacy face detection unavailable here: g16 self-test runs single-detector (Haar). Install mediapipe==0.10.21 for the full dual-detector test.")
        gx = lambda root: {"g14_cross_report_parity.py": [], "g15_semantic_video_distinctness.py": [], "g16_privacy_dual_detector.py": g16x + ["--repo", str(root / "repo")],
                           "g17_metadata_flag_parity.py": [], "g18_gold_set_thresholds.py": ["--gold", str(root / "gold_set.json"), "--min-counts", json.dumps(dict(hands=100, contact=100, boundary_clips=3, caption_ratings=12, head_frames=10))],
                           "g19_stale_artifact_chain.py": ["--repo", str(root / "repo")], "g20_four_robot_independent.py": ["--repo", str(root / "repo"), "--output-root", str(root / "out")]}
        gates = list(gx(base))
        def selected(g): return not only or g.split("_")[0] in only
        for g in gates:
            if not selected(g): continue
            rc, fails = run_gate(g, base / "out" / "ep_a", base, gx(base)[g]); ok = rc == 0
            results.append((g, "pristine fixture passes", ok, "; ".join(fails)[:200])); print(f"  [{'PASS' if ok else 'FAIL'}] {g}: pristine fixture {'passes' if ok else 'FAILS unexpectedly: ' + '; '.join(fails)[:240]}")
            if not ok: problems.append(f"{g} fails the pristine fixture")

        def mutate(tag, gate, expect_any, fn, positive=False):
            if not selected(gate): return
            root = tmp / f"mut_{tag}"; shutil.copytree(base, root)
            try: fn(root)
            except Exception as e:
                results.append((gate, tag, False, f"mutation could not be applied: {e!r}")); problems.append(f"{tag}: cannot apply ({e!r})"); print(f"  [FAIL] {tag}: mutation error {e!r}"); return
            rc, fails = run_gate(gate, root / "out" / "ep_a", root, gx(root)[gate])
            if positive:
                ok = rc == 0; detail = "good variant passes" if ok else "good variant FAILED: " + "; ".join(fails)[:200]
            else:
                hit = any(any(s in f for s in expect_any) for f in fails); ok = rc == 1 and hit
                detail = f"failed for expected reason ({[f[:60] for f in fails if any(s in f for s in expect_any)][:1]})" if ok else f"rc={rc}, expected one of {expect_any}, got {[f[:70] for f in fails][:3]}"
            results.append((gate, tag, ok, detail)); print(f"  [{'PASS' if ok else 'FAIL'}] {gate} / {tag}: {detail}")
            if not ok: problems.append(f"{gate}/{tag}: {detail}")

        ep = lambda root: root / "out" / "ep_a"
        def q_edit(f):
            def _(root): d = jl(ep(root) / "quality.json"); f(d); jw(ep(root) / "quality.json", d)
            return _
        def m_edit(file, f):
            def _(root): d = jl(ep(root) / file); f(d); jw(ep(root) / file, d)
            return _
        # g14
        mutate("g14_hand_ratio", "g14_cross_report_parity.py", ["manifest hand_valid_ratio"], q_edit(lambda d: d["hands"].__setitem__("hand_valid_ratio", 0.5)))
        mutate("g14_hw_ik", "g14_cross_report_parity.py", ["hw[unitree_g1_upper] ik_convergence_pct"], m_edit("hardware_readiness_report.json", lambda d: d["unitree_g1_upper"]["hardware_audit"].__setitem__("ik_convergence_pct", 12.3)))
        mutate("g14_cert", "g14_cross_report_parity.py", ["certificate untruncated"], m_edit("wam/roboverify_certificate.json", lambda d: d["layer_1_hand_qc"].__setitem__("untruncated_hand_visibility_pct", 22.63)))
        mutate("g14_tier", "g14_cross_report_parity.py", ["decision fields mutually consistent"], q_edit(lambda d: d["decision"].__setitem__("quality_tier", "REJECT")))
        # g15
        mutate("g15_identical", "g15_semantic_video_distinctness.py", ["byte-identical"], lambda root: shutil.copy(ep(root) / "overlay_annotated.mp4", ep(root) / "side_by_side.mp4"))
        mutate("g15_overlay_copy", "g15_semantic_video_distinctness.py", ["overlay_annotated.mp4 differs from training"],
               lambda root: ffmpeg(["-i", str(next((ep(root) / "lerobot_v3" / "videos").rglob("*.mp4"))), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(ep(root) / "overlay_annotated.mp4")]))
        # g16
        def faces(root):
            from skimage import data
            import cv2
            im = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)[37:37 + H, 63:63 + W]; cv2.imwrite(str(root / "face.png"), im)   # 93px face, no resampling
            v = next((ep(root) / "lerobot_v3" / "videos").rglob("*.mp4")); ffmpeg(["-loop", "1", "-i", str(root / "face.png"), "-t", "3", "-r", "30", "-pix_fmt", "yuv420p", "-c:v", "libx264", str(v)])
        mutate("g16_faces", "g16_privacy_dual_detector.py", ["no confirmed", "suspect rate"], faces)
        mutate("g16_overblur", "g16_privacy_dual_detector.py", ["over-blur"],
               lambda root: ffmpeg(["-i", str(root / "repo" / "data" / "raw_videos" / "ep_a.mp4"), "-vf", "boxblur=25:5", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-an", str(next((ep(root) / "lerobot_v3" / "videos").rglob("*.mp4")))]))
        # g17
        def info_edit(f):
            def _(root): d = jl(ep(root) / "lerobot_v3/meta/info.json"); f(d["ego_annotate"]); jw(ep(root) / "lerobot_v3/meta/info.json", d)
            return _
        mutate("g17_pose_claim", "g17_metadata_flag_parity.py", ["camera-pose column"], info_edit(lambda e: e.__setitem__("camera_pose_6dof_tracked", True)))
        mutate("g17_units", "g17_metadata_flag_parity.py", ["wrist units must not claim meters"], info_edit(lambda e: e["units"].__setitem__("wrist_pos", "m")))
        mutate("g17_tier", "g17_metadata_flag_parity.py", ["unverifiable tier label"], info_edit(lambda e: e.__setitem__("egoverse_tier", "EgoVerse-A")))
        mutate("g17_bimanual", "g17_metadata_flag_parity.py", ["both hands"], info_edit(lambda e: e.__setitem__("bimanual_state", True)))
        mutate("g17_unknown_flag", "g17_metadata_flag_parity.py", ["is verifiable"], info_edit(lambda e: e.__setitem__("super_quality_ok", True)))
        def good_pose(root):
            import pyarrow.parquet as pq, pyarrow as pa
            pth = next((ep(root) / "lerobot_v3" / "data").rglob("*.parquet")); t = pq.read_table(pth); pose = np.cumsum(np.random.default_rng(3).normal(size=(T, 7)), 0).astype(np.float32)
            t = t.append_column("observation.camera_pose", pa.array(pose.tolist(), pa.list_(pa.float32()))); pq.write_table(t, pth)
            info_edit(lambda e: e.__setitem__("camera_pose_6dof_tracked", True))(root); q_edit(lambda d: d["camera"].update(pose_source="slam", pose_tracked=True))(root)
        mutate("g17_honest_pose_claim", "g17_metadata_flag_parity.py", [], good_pose, positive=True)
        # g18
        mutate("g18_chirality", "g18_gold_set_thresholds.py", ["chirality accuracy"], m_edit("pipeline_eval_view.json", lambda d: [h.__setitem__("side", "right" if h["side"] == "left" else "left") for f in d["frames"] for h in f["hands"]]))
        mutate("g18_shifted", "g18_gold_set_thresholds.py", ["wearer-hand recall"], m_edit("pipeline_eval_view.json", lambda d: [h.__setitem__("wrist_xy_norm", [h["wrist_xy_norm"][0] + 0.2, h["wrist_xy_norm"][1]]) for f in d["frames"] for h in f["hands"]]))
        mutate("g18_bystander", "g18_gold_set_thresholds.py", ["hand precision"], m_edit("pipeline_eval_view.json", lambda d: [f["hands"].append(dict(side="left", valid=True, source="detected", wrist_xy_norm=[0.5, 0.15], contact=False, grasp="none")) for f in d["frames"]]))
        mutate("g18_stale_caption", "g18_gold_set_thresholds.py", ["refer to current captions"], m_edit("pipeline_eval_view.json", lambda d: d["segments"][0].__setitem__("caption", "something else entirely")))
        mutate("g18_not_blind", "g18_gold_set_thresholds.py", ["blind"], lambda root: jw(root / "gold_set.json", {**jl(root / "gold_set.json"), "blind": False}))
        # g19
        mutate("g19_annotations", "g19_stale_artifact_chain.py", ["annotations_sha256"], lambda root: (ep(root) / "annotations.json").write_text(json.dumps({"episode_id": "ep_a", "frames": [], "edited": True})))
        mutate("g19_code", "g19_stale_artifact_chain.py", ["code_tree_sha256"], lambda root: (root / "repo" / "src" / "pipeline.py").write_text("VERSION = 2\n"))
        def old_export(root):
            p = next((ep(root) / "lerobot_v3" / "data").rglob("*.parquet")); t = (ep(root) / "annotations.json").stat().st_mtime - 3600; os.utime(p, (t, t))
        mutate("g19_old_export", "g19_stale_artifact_chain.py", ["older than annotations.json"], old_export)
        mutate("g19_missing_prov", "g19_stale_artifact_chain.py", ["provenance"], m_edit("manifest.json", lambda d: d.pop("provenance")))
        # g20
        def perturb(root):
            z = dict(np.load(ep(root) / "embodiments/franka_panda/trajectory.npz")); z["target_ee_pos"] = z["target_ee_pos"] + np.array([0.02, 0, 0]); np.savez(ep(root) / "embodiments/franka_panda/trajectory.npz", **z)
        mutate("g20_perturbed_target", "g20_four_robot_independent.py", ["fk_pos_residual_mean_mm reproduced"], perturb)
        def derive(root):
            for n in ("ep_a", "ep_b", "ep_c"):
                p = root / "out" / n / "quality.json"; d = jl(p); f = d["robots"]["franka_panda"]
                for rb, (kp, km, off) in {"unitree_g1_upper": (1.2, 1.17, -2.7), "unitree_h1_upper": (1.28, 1.24, -4.05), "allegro_hand_right": (1.36, 1.31, -5.4)}.items():
                    d["robots"][rb].update(fk_pos_residual_p95_mm=f["fk_pos_residual_p95_mm"] * kp, fk_pos_residual_mean_mm=f["fk_pos_residual_mean_mm"] * km, ik_convergence_pct=f["ik_convergence_pct"] + off)
                jw(p, d)
        mutate("g20_derived_ratios", "g20_four_robot_independent.py", ["varies across episodes"], derive)
        mutate("g20_missing_keys", "g20_four_robot_independent.py", ["self-describing npz keys"],
               lambda root: np.savez(ep(root) / "embodiments/unitree_h1_upper/trajectory.npz", q=np.zeros((T, 3))))
    finally:
        if a.keep: print(f"  fixture kept at {tmp}")
        else: shutil.rmtree(tmp, ignore_errors=True)
    bad = [r_ for r_ in results if not r_[2]]
    print(f"\n  {len(results) - len(bad)}/{len(results)} self-test checks passed")
    for p_ in problems: print(f"  PROBLEM: {p_}")
    status = "FAIL" if bad or not results else "PASS"; print(f"G21 VERDICT: {status}")
    if a.json_out: Path(a.json_out).write_text(json.dumps(dict(gate=GATE, status=status, results=[dict(gate=g, case=c, ok=o, detail=d) for g, c, o, d in results], problems=problems), indent=1))
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
