"""Exact metric scale from an A4 sheet (0.210 x 0.297 m) lying flat on the table
in ANY one frame. Writes scale.json {px_per_m, method:'a4_plane'}; reference
pipeline uses it when present, else anthropometry with disclosed caveat."""
import cv2, numpy as np, json, sys
A4 = (0.210, 0.297)
img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
quad = None
for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
    p = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
    if len(p) == 4 and cv2.contourArea(c) > 0.05*img.shape[0]*img.shape[1]:
        quad = p.reshape(4, 2).astype(np.float32); break
if quad is None:
    raise RuntimeError("A4 quad not found: place the sheet flat, unoccluded, well-lit")
d = [np.linalg.norm(quad[i]-quad[(i+1) % 4]) for i in range(4)]
px_per_m = float(np.mean([d[0]/A4[1], d[1]/A4[0], d[2]/A4[1], d[3]/A4[0]]))
json.dump({"px_per_m": px_per_m, "method": "a4_plane"}, open(sys.argv[2], "w"), indent=2)
print(f"px_per_m={px_per_m:.2f}")
