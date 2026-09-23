#!/usr/bin/env python3
"""EgoAnnotate reference pipeline. Single file. No legacy wiring.
Stages: track -> detect -> contact/grasp -> retarget(mink) -> render -> export -> validate.
Usage: python scripts/run_reference_pipeline.py --video V --out OUT --urdf U"""
import argparse, json, os, sys, subprocess, shutil
import cv2, numpy as np, h5py
import pyarrow as pa, pyarrow.parquet as pq
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

FPS = 30.0
PANDA_LIM = [(-2.8973,2.8973),(-1.7628,1.7628),(-2.8973,2.8973),
             (-3.0718,-0.0698),(-2.8973,2.8973),(-0.0175,3.7525),(-2.8973,2.8973)]
PANDA_VEL = np.array([2.17,2.17,2.17,2.17,2.61,2.61,2.61])
VOCAB = ["bottle","container","case","mouse","keys","charger","cable","notebook","cup","phone"]
TIPS = (4,8,12,16,20)

def sanitize_trajectory(q, reach, reason, dt, vel_limits=PANDA_VEL, lim=PANDA_LIM, tol=1e-3):
    """Last word on feasibility. Operates ONLY on final arrays.
    Holds q[i]=q[i-1] and downgrades reach on any limit or per-frame velocity violation."""
    q = q.copy(); reach = reach.copy(); reason = list(reason)
    LO = np.array([l[0] for l in lim]); HI = np.array([l[1] for l in lim])
    budget = 0.9 * np.array(vel_limits) * dt
    if reach[0] and (np.any(q[0] < LO-tol) or np.any(q[0] > HI+tol)):
        reach[0] = False; reason[0] = "joint_limit"; q[0] = q[0]*0.0
    for i in range(1, len(q)):
        if not reach[i]:
            q[i] = q[i-1]; continue
        if np.any(q[i] < LO-tol) or np.any(q[i] > HI+tol):
            reach[i] = False; reason[i] = "joint_limit"; q[i] = q[i-1]; continue
        if np.any(np.abs(q[i]-q[i-1]) > budget):
            reach[i] = False; reason[i] = "joint_velocity"; q[i] = q[i-1]; continue
    return q, reach, reason

def canon(name):
    n = (name or "").strip().lower()
    for v in VOCAB:
        if v in n or n in v: return v
    return n if n in VOCAB else "object"

def iou(a,b):
    x1,y1 = max(a[0],b[0]), max(a[1],b[1]); x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0.,x2-x1)*max(0.,y2-y1)
    u = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/u if u>0 else 0.

def nms(dets, thr=0.5):
    dets = sorted(dets, key=lambda d: d["score"], reverse=True); keep=[]
    for d in dets:
        if all(iou(d["bbox"],k["bbox"])<thr for k in keep): keep.append(d)
    return keep

class Tracker:
    def __init__(self): self.tracks={}; self.nid=0
    def update(self, dets):
        for d in dets:
            bid,bs = None,0.3
            for tid,t in self.tracks.items():
                s=iou(d["bbox"],t["bbox"])
                if s>bs: bs,bid=s,tid
            if bid is None:
                bid=self.nid; self.nid+=1
                self.tracks[bid]={"bbox":d["bbox"],"name":d["name"],"age":0}
            t=self.tracks[bid]; t["bbox"]=d["bbox"]; t["age"]=0
        for t in self.tracks.values(): t["age"]+=1
        for tid in [t for t,v in self.tracks.items() if v["age"]>15]: del self.tracks[tid]
        return [{"id":tid,"bbox":tr["bbox"],"name":tr["name"] or "unknown"}
                for tid,tr in self.tracks.items() if tr["age"]<=3]

class Hands:
    def __init__(self):
        import mediapipe as mp; self.mp=mp
        self.video=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=2,
            min_detection_confidence=0.3,min_tracking_confidence=0.3,model_complexity=1)
        self.static=mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=2,
            min_detection_confidence=0.25,model_complexity=1)
    @staticmethod
    def _k(lm): return [c for p in lm.landmark for c in (p.x,p.y,p.z)]
    def _det(self,rgb,st):
        r=(self.static if st else self.video).process(rgb); o=[]
        if r.multi_hand_landmarks:
            for lm,h in zip(r.multi_hand_landmarks,r.multi_handedness):
                o.append((h.classification[0].label.lower(),self._k(lm)))
        return o
    def track(self,frames):
        n=len(frames); rec=[{"i":i,"lp":False,"rp":False,"li":False,"ri":False,
                             "lk":[0.]*63,"rk":[0.]*63} for i in range(n)]
        prev={"l":None,"r":None}
        for i,f in enumerate(frames):
            rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB); asg={}
            for lab,k in self._det(rgb,False):
                side="l" if lab=="left" else "r"
                if prev["l"] is not None and prev["r"] is not None:
                    w=np.array(k[0:2]); dl=np.linalg.norm(w-prev["l"]); dr=np.linalg.norm(w-prev["r"])
                    if min(dl,dr)<0.25: side="l" if dl<dr else "r"
                asg[side]=k
            for s,k in asg.items():
                rec[i][s+"p"]=True; rec[i][s+"k"]=k; prev[s]=np.array(k[0:2])
        for i in range(n):                                   # static rescue
            miss=[s for s in ("l","r") if not rec[i][s+"p"] and i>0 and rec[i-1][s+"p"]]
            if not miss: continue
            rgb=cv2.cvtColor(frames[i],cv2.COLOR_BGR2RGB)
            p={s:(np.array(rec[i-1][s+"k"][0:2]) if rec[i-1][s+"p"] else None) for s in ("l","r")}
            for lab,k in self._det(rgb,True):
                side="l" if lab=="left" else "r"
                if p["l"] is not None and p["r"] is not None:
                    w=np.array(k[0:2]); dl=np.linalg.norm(w-p["l"]); dr=np.linalg.norm(w-p["r"])
                    if min(dl,dr)<0.25: side="l" if dl<dr else "r"
                if side in miss:
                    rec[i][side+"p"]=True; rec[i][side+"k"]=k; miss.remove(side)
        for s in ("l","r"):                                  # <=5 gap interp
            i=0
            while i<n:
                if not rec[i][s+"p"]:
                    j=i
                    while j<n and not rec[j][s+"p"]: j+=1
                    if 0<i and j<n and j-i<=5:
                        a,b=rec[i-1][s+"k"],rec[j][s+"k"]
                        for k2 in range(i,j):
                            t=(k2-(i-1))/(j-(i-1))
                            rec[k2][s+"k"]=[xa+t*(xb-xa) for xa,xb in zip(a,b)]
                            rec[k2][s+"p"]=True; rec[k2][s+"i"]=True
                    i=j
                else: i+=1
        return rec

def grasp_openness(k):
    w=np.array(k[0:3]); mcp=np.array(k[27:30])
    palm=max(np.linalg.norm(w-mcp),1e-6)
    return float(np.mean([np.linalg.norm(np.array(k[i*3:i*3+3])-w) for i in (8,12,16,20)])/palm)

def majority(labels,present):
    out=list(labels); n=len(labels); i=0
    while i<n:
        if not present[i]: i+=1; continue
        j=i
        while j<n and present[j]: j+=1
        for k in range(i,j):
            lo,hi=max(i,k-2),min(j,k+3)
            vs=[labels[m] for m in range(lo,hi) if present[m] and labels[m]!="unknown"]
            if vs: out[k]=max(set(vs),key=vs.count)
        i=j
    return out

class CSM:
    def __init__(s): s.on=s.off=0; s.st=False
    def upd(s,ev):
        if ev: s.on,s.off=s.on+1,0
        else: s.off,s.on=s.off+1,0
        if not s.st and s.on>=3: s.st=True
        elif s.st and s.off>=3: s.st=False
        return s.st

def wrap_px(text,font,sc,th,maxw,ml=3):
    L,cur=[],""
    for w in text.split():
        t=(cur+" "+w).strip()
        if cv2.getTextSize(t,font,sc,th)[0][0]<=maxw or not cur: cur=t
        else: L.append(cur); cur=w
    if cur: L.append(cur)
    return L[:ml]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--video",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--urdf",required=True)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    cap=cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise RuntimeError("cannot open "+a.video)
    fps_src=cap.get(cv2.CAP_PROP_FPS) or 60.0
    SAMPLE_EVERY=2
    frames=[]; idx=0
    while True:
        ok,f=cap.read()
        if not ok: break
        if idx%SAMPLE_EVERY==0: frames.append(f)
        idx+=1
    cap.release()
    fps=fps_src/SAMPLE_EVERY
    n=len(frames); H,W=frames[0].shape[:2]
    print(f"[ref] frames={n} fps={fps:.2f} {W}x{H}")

    # ---- detection (Grounding DINO, constrained vocab) ----
    wpath="weights/groundingdino_swint_ogc.pth"
    if not os.path.exists(wpath):
        raise RuntimeError("MISSING WEIGHTS. Run:\nmkdir -p weights && curl -L -o "+wpath+
          " https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth")
    import groundingdino
    cfg=os.path.join(os.path.dirname(groundingdino.__file__),"config","GroundingDINO_SwinT_OGC.py")
    if not os.path.exists(cfg): raise RuntimeError("GDINO config missing: "+cfg)
    import torch, time
    torch.set_num_threads(max(1,(os.cpu_count() or 2)//2))
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
    from PIL import Image
    gd_transform=T.Compose([
        T.RandomResize([800],max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    model=load_model(cfg,wpath,device="cpu"); model.eval()
    MAXSIDE=400; KEY_EVERY=15; DIFF_THRESH=12.0
    sc=min(1.0, MAXSIDE/float(max(H,W)))
    def detect(i):
        fin=frames[i] if sc>=1.0 else cv2.resize(frames[i],(int(W*sc),int(H*sc)))
        rgb=cv2.cvtColor(fin,cv2.COLOR_BGR2RGB)
        t_img,_=gd_transform(Image.fromarray(rgb),None)
        with torch.no_grad():
            boxes,logits,phrases=predict(model,t_img,". ".join(VOCAB)+".",0.30,0.25,device="cpu")
        h2,w2=fin.shape[:2]; dets=[]
        for b,p,s in zip(boxes,phrases,logits):
            cx,cy,bw,bh=b.tolist()
            dets.append({"bbox":[(cx-bw/2)*w2/sc,(cy-bh/2)*h2/sc,
                                 (cx+bw/2)*w2/sc,(cy+bh/2)*h2/sc],
                         "name":canon(p),"score":float(s)})
        return nms(dets)
    cache_path=os.path.join(a.out,"gdino_cache.json")
    if os.path.exists(cache_path):
        print(f"[ref] loading cached GDINO detections from {cache_path}")
        with open(cache_path) as f: per_obj=json.load(f)
    else:
        key_idx=[]; prev_g=None
        for i,f in enumerate(frames):
            g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
            d=0.0 if prev_g is None else float(np.mean(cv2.absdiff(g,prev_g)))
            if prev_g is None or i%KEY_EVERY==0 or d>DIFF_THRESH: key_idx.append(i)
            prev_g=g
        print(f"[ref] keyframes={len(key_idx)} of {n} (GDINO calls)")
        trk=Tracker(); per_obj=[[] for _ in range(n)]; t0=time.time()
        for ki in key_idx:
            per_obj[ki]=trk.update(detect(ki))
            print(f"[ref] detect {ki}/{n} elapsed={time.time()-t0:.0f}s")
        for a2,b2 in zip(key_idx,key_idx[1:]):
            if b2-a2<=1: continue
            ia={o["id"]:o for o in per_obj[a2]}; ib={o["id"]:o for o in per_obj[b2]}
            for k in range(a2+1,b2):
                t=(k-a2)/float(b2-a2); out=[]
                for oid in set(ia)|set(ib):
                    if oid in ia and oid in ib:
                        A=ia[oid]["bbox"]; B=ib[oid]["bbox"]
                        bb=[xa+t*(xb-xa) for xa,xb in zip(A,B)]
                        pad=0.15*((bb[2]-bb[0])+(bb[3]-bb[1]))/2.0
                        out.append({"id":oid,"bbox":[bb[0]-pad,bb[1]-pad,bb[2]+pad,bb[3]+pad],
                                    "name":ia[oid]["name"]})
                    elif oid in ia:
                        out.append({"id":oid,"bbox":ia[oid]["bbox"],"name":ia[oid]["name"]})
                per_obj[k]=out
        if key_idx:
            for k in range(key_idx[-1]+1,n): per_obj[k]=per_obj[key_idx[-1]]
        os.makedirs(a.out,exist_ok=True)
        with open(cache_path,"w") as f: json.dump(per_obj,f)

    # ---- tracking + grasp + contact ----
    rec=Hands().track(frames)
    ds=[grasp_openness(rec[i][s+"k"]) for i in range(n) for s in ("l","r") if rec[i][s+"p"]]
    p30,p70=np.percentile(ds,[30,70])
    def glabel(d): return "open" if d>p70 else ("power_wrap" if d<p30 else "precision_pinch")
    gl={s:majority([glabel(grasp_openness(rec[i][s+"k"])) for i in range(n)],
                    [rec[i][s+"p"] for i in range(n)]) for s in ("l","r")}
    csm={}; contact={s:[False]*n for s in ("l","r")}; cobj={s:[None]*n for s in ("l","r")}
    for i in range(n):
        for s in ("l","r"):
            kps=rec[i][s+"k"]; tips=[(kps[t*3],kps[t*3+1]) for t in TIPS]
            for o in per_obj[i]:
                x1,y1,x2,y2 = o["bbox"][0]/W, o["bbox"][1]/H, o["bbox"][2]/W, o["bbox"][3]/H
                ev=rec[i][s+"p"] and sum(1 for x,y in tips if x1<=x<=x2 and y1<=y<=y2)>=2
                key=(s,o["id"]); csm.setdefault(key,CSM())
                if csm[key].upd(ev): contact[s][i]=True; cobj[s][i]=o["name"]

    # ---- retarget targets: object-centric, smoothed ----
    import mink, mujoco
    scale_file = os.path.join(a.out, "scale.json")
    if not os.path.exists(scale_file) and os.path.exists("scale.json"):
        scale_file = "scale.json"
    if os.path.exists(scale_file):
        with open(scale_file) as sf:
            sdata = json.load(sf)
        K = 1.0 / sdata["px_per_m"]
        scale_method = "a4_plane"
    else:
        K = 0.6
        scale_method = "anthropometry±15%"

    mj=mujoco.MjModel.from_xml_path(a.urdf)
    _c=["panda_link8","panda_hand","link8","panda_end_effector","ee_link","panda_link7"]
    ee=next((c for c in _c if mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_BODY,c)>=0),None)
    if ee is None:
        raise RuntimeError("EE body not found; bodies="+str(
            [mujoco.mj_id2name(mj,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(mj.nbody)]))
    cfgm=mink.Configuration(mj)
    task=mink.FrameTask(frame_name=ee,frame_type="body",position_cost=1.0,orientation_cost=1.0)
    post=mink.PostureTask(mj,cost=1e-3)
    lim=mink.ConfigurationLimit(mj)
    vel_lim=mink.VelocityLimit(mj, {f"panda_joint{j+1}": 0.899*PANDA_VEL[j] for j in range(7)})
    HOME=np.array([0.5,0.0,0.25])
    pos_t=[]; quat_t=[]; present=[]
    for i in range(n):
        s="r" if rec[i]["rp"] else ("l" if rec[i]["lp"] else None)
        present.append(s is not None)
        if s is None: pos_t.append(HOME.copy()); quat_t.append([1.,0.,0.,0.]); continue
        k=rec[i][s+"k"]; w=np.array(k[0:2])
        oc=np.array([0.5,0.6])
        if cobj[s][i]:
            bb=[o["bbox"] for o in per_obj[i] if o["name"]==cobj[s][i]]
            if bb: oc=np.array([(bb[0][0]+bb[0][2])/2/W,(bb[0][1]+bb[0][3])/2/H])
        rel=w-oc
        pos_t.append(HOME+np.array([rel[0]*K,-rel[1]*K,0.0]))
        x=np.array(k[15:18])-np.array(k[51:54]); z=np.array(k[0:3])-np.array(k[27:30])
        x=x/(np.linalg.norm(x)+1e-9); z=z/(np.linalg.norm(z)+1e-9); y=np.cross(z,x)
        R=np.stack([x,y,z],axis=1)
        quat_t.append(Rotation.from_matrix(R).as_quat())
    pos_t=np.array(pos_t); quat_t=np.array(quat_t); pos_s=pos_t.copy()
    i=0
    while i<n:
        if not present[i]: i+=1; continue
        j=i
        while j<n and present[j]: j+=1
        b=list(range(i,j)); m=len(b); wv=min(9,m if m%2 else m-1)
        if wv>=5:
            pos_s[b]=savgol_filter(pos_t[b],wv,2,axis=0)
            rv=Rotation.from_quat(quat_t[b]).as_rotvec()
            quat_t[b]=Rotation.from_rotvec(savgol_filter(rv,wv,2,axis=0)).as_quat()
        i=j

    # ---- mink IK with enforced limits + velocity gate ----
    HOME_Q=np.array([(lo+hi)/2.0 for lo,hi in PANDA_LIM])
    q_full=np.zeros(mj.nq); q_full[:7]=HOME_Q
    cfgm.update(q_full)
    q=np.zeros((n,7)); reach=np.zeros(n,bool); reason=["no_hand"]*n
    grip=np.zeros(n); method=["no_hand"]*n
    q_prev=HOME_Q.copy(); dt=1.0/fps
    VEL_LIM=0.9*PANDA_VEL
    LO=np.array([PANDA_LIM[j][0] for j in range(7)]); HI=np.array([PANDA_LIM[j][1] for j in range(7)])
    for i in range(n):
        s="r" if rec[i]["rp"] else ("l" if rec[i]["lp"] else None)
        d=grasp_openness(rec[i][s+"k"]) if s else 2.8
        grip[i]=min(max((d-1.7)/1.0,0.0),1.0)*0.08
        method[i]=gl[s][i] if s else "no_hand"
        if s is None:
            reason[i]="no_hand"; q[i]=q_prev.copy()
            q_full[:7]=q_prev; cfgm.update(q_full); continue
        if rec[i][s+"i"]:
            reason[i]="interpolated"; q[i]=q_prev.copy()
            q_full[:7]=q_prev; cfgm.update(q_full); continue
        try:
            task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(Rotation.from_quat(quat_t[i]).as_matrix()), pos_s[i]))
            qt=cfgm.q.copy(); qt[mj.nq-2:]=grip[i]/2.0
            post.set_target(qt)
            dq=mink.solve_ik(cfgm,[task,post],dt,solver="daqp",limits=[lim,vel_lim],damping=1e-6)
            cfgm.integrate_inplace(dq,dt)
            qc=np.array(cfgm.q[:7],float)
            qc=np.clip(qc,LO,HI)
            q_full[:7]=qc; cfgm.update(q_full)
        except Exception as e:
            print("[ref] mink error",i,repr(e)); sys.exit(2)
        if np.any(qc<LO-1e-4) or np.any(qc>HI+1e-4):
            reason[i]="joint_limit"; q[i]=q_prev.copy(); continue
        reach[i]=True; reason[i]="tracked"; q[i]=qc; q_prev=qc.copy()

    q, reach, reason = sanitize_trajectory(q, reach, reason, dt)

    # ---- HARD AUDIT: raises, so bad data cannot ship ----
    if np.any(q<LO-1e-4) or np.any(q>HI+1e-4):
        raise RuntimeError("EXPORT BLOCKED: joint limits violated")
    idx=np.where(reach)[0]
    for f1,f2 in zip(idx[:-1],idx[1:]):
        if f2-f1==1 and np.any(np.abs(q[f2]-q[f1])/dt > VEL_LIM+1e-6):
            raise RuntimeError(f"EXPORT BLOCKED: velocity violation frames {f1},{f2}")
    if int(np.sum(reach & np.array([rec[i]["li"] or rec[i]["ri"] for i in range(n)]))):
        raise RuntimeError("EXPORT BLOCKED: interpolated frame reachable")
    mov=int(np.sum(np.abs(np.diff(q,axis=0)).max(axis=1)>1e-6))
    if mov < 0.5*(n-1):
        raise RuntimeError(f"EXPORT BLOCKED: trajectory frozen (moving={mov})")
    print(f"[ref] audit ok: moving={mov}/{n-1} reachable={int(reach.sum())} reasons={ {r:int((np.array(reason)==r).sum()) for r in set(reason)} }")

    # ---- render side-by-side (table + proxy) ----
    import pybullet as pb, pybullet_data
    cli=pb.connect(pb.DIRECT); pb.setAdditionalSearchPath(pybullet_data.getDataPath(),physicsClientId=cli)
    pb.loadURDF("plane.urdf",physicsClientId=cli)
    tb=pb.createMultiBody(0,pb.createCollisionShape(pb.GEOM_BOX,halfExtents=[0.35,0.3,0.02],physicsClientId=cli),
        pb.createVisualShape(pb.GEOM_BOX,halfExtents=[0.35,0.3,0.02],rgbaColor=[0.45,0.3,0.15,1],physicsClientId=cli),
        basePosition=[0.5,0,0.23],physicsClientId=cli)
    cyl_id=pb.createMultiBody(0.05,pb.createCollisionShape(pb.GEOM_CYLINDER,radius=0.03,height=0.12,physicsClientId=cli),
        pb.createVisualShape(pb.GEOM_CYLINDER,radius=0.03,length=0.12,rgbaColor=[0.8,0.1,0.1,1],physicsClientId=cli),
        basePosition=[0.5,0,0.31],physicsClientId=cli)
    rob=pb.loadURDF(a.urdf,useFixedBase=True,physicsClientId=cli)
    num_j=pb.getNumJoints(rob,physicsClientId=cli)
    arm_idx=[j for j in range(num_j)
             if pb.getJointInfo(rob,j,physicsClientId=cli)[1].decode() in
             ("panda_joint%d"%k for k in range(1,8))] or list(range(7))
    vw=pb.computeViewMatrixFromYawPitchRoll([0.5,0,0.3],1.2,35,-25,0,2)
    pj=pb.computeProjectionMatrixFOV(60,4/3,0.01,5)
    sbs=a.out+"/side_by_side.mp4"; tmp=sbs+".tmp.mp4"
    wr=cv2.VideoWriter(tmp,cv2.VideoWriter_fourcc(*"mp4v"),FPS,(1120,550))
    for i in range(n):
        pb.resetBasePositionAndOrientation(cyl_id, [pos_s[i][0], pos_s[i][1], 0.31], [0,0,0,1], physicsClientId=cli)
        for jj,av in zip(arm_idx,q[i]): pb.resetJointState(rob,jj,av,physicsClientId=cli)
        for jj in range(num_j):
            if pb.getJointInfo(rob,jj,physicsClientId=cli)[1].decode().startswith("panda_finger"):
                pb.resetJointState(rob,jj,grip[i]/2,physicsClientId=cli)
        _,_,px,_,_=pb.getCameraImage(640,480,vw,pj,renderer=pb.ER_TINY_RENDERER,physicsClientId=cli)
        rgb=np.array(px,dtype=np.uint8).reshape(480,640,4)[:,:,:3]
        rf=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        hf=cv2.resize(frames[i],(int(W*480/H),480))
        lab=np.zeros((30,1120,3),np.uint8)
        cv2.putText(lab,"HUMAN DEMONSTRATION",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,255,200),1)
        cv2.putText(lab,"PANDA RETARGETED (mink/MuJoCo)",(max(hf.shape[1]+10,270),20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,255),1)
        comp=np.vstack([lab,np.hstack([hf,cv2.resize(rf,(1120-hf.shape[1],480))]),
                        np.zeros((40,1120,3),np.uint8)])
        col=(100,255,100) if reach[i] else (100,100,255)
        cv2.putText(comp,f"Frame {i:05d} t={i/FPS:.2f}s IK: {'VALID' if reach[i] else 'FALLBACK('+reason[i]+')'} "
                    f"Gripper={grip[i]*1000:.1f}mm Method={method[i]}",(10,535),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1,cv2.LINE_AA)
        wr.write(comp)
    wr.release(); pb.disconnect(cli)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg","-y","-i",tmp,"-c:v","libx264","-pix_fmt","yuv420p",
                        "-movflags","+faststart",sbs],check=True,capture_output=True); os.remove(tmp)
    else: os.replace(tmp,sbs)

    # ---- overlay ----
    ov=a.out+"/overlay_annotated.mp4"; tmp2=ov+".tmp.mp4"
    wr=cv2.VideoWriter(tmp2,cv2.VideoWriter_fourcc(*"mp4v"),FPS,(W,H))
    for i in range(n):
        f=frames[i].copy(); placed=[]; seen_lab=set()
        for o in per_obj[i]:
            if o["name"] in seen_lab: continue
            seen_lab.add(o["name"])
            x1,y1,x2,y2=[int(v) for v in o["bbox"]]
            cv2.rectangle(f,(x1,y1),(x2,y2),(160,160,160),1)
            t=o["name"]; (tw,th),_=cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,0.4,1)
            y=y1-6; r=[x1,y-th-4,x1+tw,y+4]
            for p in placed:
                if not(r[2]<p[0] or r[0]>p[2] or r[3]<p[1] or r[1]>p[3]): y=p[3]+th+6; r=[x1,y-th-4,x1+tw,y+4]
            placed.append(r); cv2.putText(f,t,(x1,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(220,220,220),1,cv2.LINE_AA)
        for s,col in (("l",(255,0,255)),("r",(0,255,0))):
            if rec[i][s+"p"]:
                c=(0,255,255) if rec[i][s+"i"] else col
                k=rec[i][s+"k"]; pts=[(int(k[j*3]*W),int(k[j*3+1]*H)) for j in range(21)]
                for a2,b2 in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),
                              (11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20)]:
                    cv2.line(f,pts[a2],pts[b2],c,1,cv2.LINE_AA)
                for p in pts: cv2.circle(f,p,2,c,-1,cv2.LINE_AA)
        pw=W//2-6
        for kk,s in enumerate(("l","r")):
            x=3+kk*(pw+3); cv2.rectangle(f,(x,0),(x+pw,84),(0,0,0),-1)
            cv2.rectangle(f,(x,0),(x+pw,84),(0,255,0) if s=="l" else (255,0,255),2)
            cv2.putText(f,"LEFT" if s=="l" else "RIGHT",(x+6,16),cv2.FONT_HERSHEY_SIMPLEX,0.45,
                        (0,255,0) if s=="l" else (255,0,255),1,cv2.LINE_AA)
            ik="IK: VALID" if reach[i] else "IK: FALLBACK"
            (bw2,bh2),_=cv2.getTextSize(ik,cv2.FONT_HERSHEY_SIMPLEX,0.4,1)
            cv2.putText(f,ik,(x+pw-bw2-8,16),cv2.FONT_HERSHEY_SIMPLEX,0.4,
                        (0,255,0) if reach[i] else (0,165,255),1,cv2.LINE_AA)
            obj=cobj[s][i]
            txt=(gl[s][i]+"; "+("in contact with "+obj if contact[s][i] else "no contact"))
            if rec[i][s+"i"]: txt="[interp] "+txt
            for li,ln in enumerate(wrap_px(txt,cv2.FONT_HERSHEY_SIMPLEX,0.38,1,pw-12)):
                cv2.putText(f,ln,(x+6,36+li*14),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1,cv2.LINE_AA)
        cv2.rectangle(f,(0,H-26),(W,H),(0,0,0),-1)
        cv2.line(f,(i,H-26),(i,H),(255,255,0),1)
        cv2.putText(f,f"{i}/{n} {i/FPS:.2f}s",(4,H-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv2.LINE_AA)
        wr.write(f)
    wr.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg","-y","-i",tmp2,"-c:v","libx264","-pix_fmt","yuv420p",
                        "-movflags","+faststart",ov],check=True,capture_output=True); os.remove(tmp2)
    else: os.replace(tmp2,ov)

    # ---- export ----
    vexp = np.abs(np.diff(q, axis=0))/dt
    if vexp.max() > 0.9*PANDA_VEL.max() + 1e-6:
        raise RuntimeError(f"EXPORT BLOCKED: exported speed {vexp.max():.2f} rad/s")
    st=np.zeros((n,24),np.float32)
    st[:,0:3]=pos_s; st[:,3:9]=Rotation.from_quat(quat_t).as_matrix()[:,:,:2].reshape(n,6)
    st[:,9]=grip/0.08
    act=np.zeros((n,24),np.float32)
    act[1:,0:3]=np.diff(pos_s,axis=0); act[1:,9:]=np.diff(np.stack([grip/0.08]*15,axis=1),axis=0)
    with h5py.File(a.out+"/episode_rlds.hdf5","w") as h:
        ep=h.create_group("whatsapp_video"); steps=ep.create_group("steps"); obs=steps.create_group("observation")
        obs.create_dataset("robot_joint_angles",data=q.astype(np.float32))
        obs.create_dataset("robot_gripper_opening_m",data=grip.astype(np.float32).reshape(n,1))
        steps.create_dataset("action",data=act)
        steps.create_dataset("robot_reachable",data=reach)
    os.makedirs(a.out+"/lerobot_v3/data/chunk-000",exist_ok=True)
    os.makedirs(a.out+"/lerobot_v3/meta",exist_ok=True)
    tbl=pa.table({"observation.state":pa.array(st.tolist(),type=pa.list_(pa.float32(),24)),
                  "action":pa.array(act.tolist(),type=pa.list_(pa.float32(),24)),
                  "observation.robot_joint_angles":pa.array(q.tolist(),type=pa.list_(pa.float32(),7)),
                  "observation.robot_gripper_opening_m":pa.array(grip.reshape(n,1).tolist(),type=pa.list_(pa.float32(),1)),
                  "robot_reachable":pa.array(reach.tolist(),type=pa.bool_()),
                  "frame_index":pa.array(list(range(n)),type=pa.int64())})
    pq.write_table(tbl,a.out+"/lerobot_v3/data/chunk-000/file-000.parquet")
    json.dump({"codebase_version":"v3.0","total_frames":n,"fps":FPS},
              open(a.out+"/lerobot_v3/meta/info.json","w"),indent=2)
    lost=sum(1 for i in range(n) if not(rec[i]["lp"] and rec[i]["rp"]))
    loss_active=100*sum(1 for i in range(n) if not(rec[i]["lp"] or rec[i]["rp"]))/n
    pin_rate=float(np.mean([1 if grip[i]<0.005 else 0 for i in range(n)]))

    task_desc = "Pick up object and place on target table"
    json.dump({"episode_id": os.path.basename(a.out), "video_path": os.path.abspath(a.video),
               "task_description": task_desc, "num_frames": n, "duration_seconds": n/FPS,
               "target_robot": "panda", "urdf": os.path.basename(a.urdf),
               "ik": "mink/MuJoCo (daqp)", "scale_method": scale_method},
              open(os.path.join(a.out, "metadata.json"), "w"), indent=2)

    json.dump({"total_frames":n,"reachable":int(reach.sum()),
               "pct_reachable":100*reach.sum()/n,"tracking_loss_pct":100*lost/n,
               "tracking_loss_pct_active":loss_active,
               "reason_histogram":{r:int((np.array(reason)==r).sum()) for r in set(reason)},
               "gripper_pin_rate":float(pin_rate),"sanitizer":"export-side v1",
               "max_joint_speed_rad_s":float(np.max(np.abs(np.diff(q,axis=0)))/dt),
               "joint_limits":"enforced_by_solver_and_audited","ik":"mink/MuJoCo (daqp)"},
              open(a.out+"/summary.json","w"),indent=2)
    print(f"[ref] DONE reachable={reach.sum()}/{n} loss={100*lost/n:.1f}%")

if __name__=="__main__": main()
