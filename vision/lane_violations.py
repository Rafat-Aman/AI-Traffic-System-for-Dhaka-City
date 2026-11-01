"""Lane-line violations (V5) on a recorded video with automatic lane-line detection, plus the violators' plates.

    python lane_violations.py --source video.mp4 --out lane_out

1. Lane lines are found in the picture itself. The road is locked to one view (--ref-at, default the middle of the
   video): frames spread over the whole video are matched to it (surveillance/camera_motion.py, vehicles masked out)
   and warped into it, i.e. stabilised on the road; their median is a picture of the empty road, and the painted
   lines are traced in it row by row (thin bright paint, curves allowed). Only lines running toward the road's
   vanishing point are kept (not people on the pavement). A stretch whose paint covers most of its length is SOLID,
   otherwise BROKEN (dashed). The outermost lines are the road edges when no asphalt is seen beyond them.
2. Zones: one road zone between the road edges (its width = lanes x --lane-width metres) and the solid lines between
   the edges as V5 lane lines (a curved line becomes a few short straight pieces). --include-broken also counts
   broken lines (any lane change), --include-edges the edge lines. Written to <out>/lanes.json (the format
   violation_tracker.py --zones reads) with a picture of what was found (<out>/lanes_detected.jpg).
3. The violation pipeline (surveillance/pipeline.py, V5 only) runs on every frame, every vehicle measured in the
   locked view (the empty road is the camera-movement reference); each violator's plate is read with the web
   service's reader (service/plates.py: BLPD plate detector + our character model, voted over the vehicle's 30
   largest views).
4. Frame-by-frame check of the lock: once per video second the lane lines are mapped into the actual frame and the
   share lying on paint seen in that frame is recorded (alignment.csv; each violation carries the latest value).

Output in <out>: violations.csv + snapshots/ (scene and close-up per violation), plates.csv (one row per violation:
plate read, status, plate image, lane alignment), plates/ (plate crops and "vehicle + plate" images), annotated.mp4,
alignment.csv, lanes.json, lanes_detected.jpg, background.jpg.

Plate detector: models/plate_models/LPD_Model.pt from https://github.com/prosensarker4/BLPD (not in this repo).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

VISION = Path(__file__).resolve().parent
sys.path.insert(0, str(VISION))
sys.path.insert(0, str(VISION / "pipelines"))

WORK_W = 1920   # lane detection resolution


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lane-line violations with automatic lane detection")
    p.add_argument("--source", required=True, help="Video file")
    p.add_argument("--out", default="lane_out", help="Output folder")
    p.add_argument("--lane-width", type=float, default=3.5, help="Lane width in metres (sets the road scale)")
    p.add_argument("--include-broken", action="store_true", help="Broken (dashed) lines count too: any lane change")
    p.add_argument("--include-edges", action="store_true", help="The two road-edge lines count too")
    p.add_argument("--margin", type=float, default=0.6,
                   help="V5: metres the vehicle's centre must be clear of the line on both sides (lower = stricter)")
    p.add_argument("--hold", type=float, default=0.5, help="V5: seconds on each side of the line")
    p.add_argument("--solid-coverage", type=float, default=0.75,
                   help="Share of a line's length that must be painted for it to be solid")
    p.add_argument("--sample-sec", type=float, default=0,
                   help="Lane detection: use the first N seconds (0 = the whole video)")
    p.add_argument("--sample-frames", type=int, default=40, help="Lane detection: frames in the median")
    p.add_argument("--ref-at", type=float, default=None,
                   help="Video time (s) of the view the road is locked to (default: the middle of the video). "
                        "Pick a moment showing the whole road: zoomed-in stretches are mapped into it")
    p.add_argument("--lanes-only", action="store_true", help="Detect the lanes, write lanes.json, then stop")
    p.add_argument("--zones", default=None,
                   help="Use this zone file instead of detecting lanes (e.g. an earlier run's lanes.json; its "
                        "background.jpg next to it is used for the road lock)")
    p.add_argument("--imgsz", type=int, default=960, help="Vehicle detector input size")
    p.add_argument("--analysis-fps", type=float, default=30, help="Frames analysed per second")
    p.add_argument("--end", type=float, default=None, help="Stop at this video time (s)")
    p.add_argument("--no-stabilize", action="store_true", help="Camera is fixed: do not follow camera movement")
    p.add_argument("--no-plates", action="store_true", help="Do not read number plates")
    p.add_argument("--no-video", action="store_true", help="Do not write annotated.mp4")
    p.add_argument("--video-width", type=int, default=1920, help="Width of annotated.mp4")
    return p.parse_args(argv)


# ---------------------------------------------------------------- 1. background (empty road)

def build_background(path, model_path, imgsz, sample_sec, n_frames, ref_at=None, log=print):
    """Median picture of the empty road, WORK_W wide, in the view of the reference frame (the frame at `ref_at`
    seconds; default: the middle of the sampled stretch). Frames spread over the first `sample_sec` seconds (0 = the
    whole video) are each matched to the reference (surveillance/camera_motion.py, vehicles masked out) and warped
    into its view, i.e. stabilised on the road; frames that cannot be matched are left out.
    Returns (background, reference frame at WORK_W, full frame size (W, H), reference time)."""
    from ultralytics import YOLO

    from surveillance.camera_motion import CameraMotion
    from surveillance.detector import pick_device

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    last = total if not sample_sec else min(total or int(sample_sec * fps), int(sample_sec * fps))
    ref_idx = int(last // 2 if ref_at is None else min(last - 1, max(0, ref_at * fps)))

    def read(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        return f if ok else None

    first = read(ref_idx)
    if first is None:
        raise SystemExit(f"Cannot read frame {ref_idx} of {path}")
    H0, W0 = first.shape[:2]
    k = WORK_W / W0

    def small(f):
        return cv2.resize(f, (WORK_W, round(f.shape[0] * k)), interpolation=cv2.INTER_AREA)

    device = pick_device()
    model = YOLO(str(model_path), task="detect")

    def vehicles(img):
        return model.predict(img, imgsz=imgsz, conf=0.15, verbose=False, device=device)[0].boxes.xyxy.tolist()

    ref = small(first)
    ref_boxes = vehicles(ref)
    # the reference's own vehicles must not be matched: CameraMotion leaves its "zone" polygons out of the features
    motion = CameraMotion(ref, [np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) for x1, y1, x2, y2 in ref_boxes])
    stack, masks, used = [], [], 0
    indices = sorted(set(np.linspace(0, max(0, last - 1), max(2, n_frames)).astype(int).tolist()) | {ref_idx})
    for i in indices:
        frame = ref if i == ref_idx else read(i)
        if frame is None:
            continue
        if frame.shape[1] != WORK_W:
            frame = small(frame)
        t = i / fps
        boxes = ref_boxes if i == ref_idx else vehicles(frame)
        motion.anchor_due = None   # frames seconds apart: always match to the reference itself
        motion.update(frame, t, boxes)
        if motion.last_ok_t != t:
            log(f"  {t:5.1f}s: could not be matched to the reference view, left out")
            continue
        veh = np.zeros(frame.shape[:2], np.uint8)
        for x1, y1, x2, y2 in boxes:
            pad = 0.04 * (x2 - x1)
            cv2.rectangle(veh, (int(x1 - pad), int(y1 - pad)), (int(x2 + pad), int(y2 + pad)), 255, -1)
        size = (frame.shape[1], frame.shape[0])
        stack.append(cv2.warpAffine(frame, motion.A, size, flags=cv2.INTER_LINEAR, borderValue=0))
        valid = cv2.warpAffine(np.full(frame.shape[:2], 255, np.uint8), motion.A, size, flags=cv2.INTER_NEAREST)
        masks.append((valid > 0) & (cv2.warpAffine(veh, motion.A, size, flags=cv2.INTER_NEAREST) == 0))
        used += 1
        log(f"  {t:5.1f}s: {len(boxes)} vehicles masked, view shift {motion.shift_px:.0f} px, zoom {motion.zoom:.2f}")
    cap.release()
    if not stack:
        raise SystemExit("No frames could be matched for lane detection")
    data = np.stack(stack).astype(np.float32)
    data[~np.stack(masks)] = np.nan
    bg = np.nanmedian(data, axis=0)
    # pixels covered by a vehicle (or outside the picture) in every frame: fall back to the plain median / reference
    holes = np.isnan(bg)
    if holes.any():
        bg[holes] = ref.astype(np.float32)[holes]
    log(f"  empty road from {used} of {len(indices)} frames, in the view at {ref_idx / fps:.1f}s")
    return np.clip(bg, 0, 255).astype(np.uint8), ref, (W0, H0), ref_idx / fps


# ---------------------------------------------------------------- 2. lane lines in the empty road

def paint_mask(bg):
    """Thin bright paint (white or yellow) on the road."""
    H, W = bg.shape[:2]
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    kw = max(15, int(0.025 * W)) | 1
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
    hls = cv2.cvtColor(bg, cv2.COLOR_BGR2HLS)
    hue, sat = hls[..., 0], hls[..., 2]
    white = sat < 90
    yellow = (hue >= 12) & (hue <= 40) & (sat >= 60)
    thr = max(25.0, float(np.percentile(tophat, 97)))
    mask = ((tophat > thr) & (white | yellow)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask, tophat


def trace_lines(mask, row_step=4, log=print):
    """Follows paint runs from the bottom of the picture up. Returns [{"pts": [(x, y)], "rows": n sampled rows
    spanned, "hits": rows with paint}]."""
    H, W = mask.shape
    max_run = int(0.03 * W)
    tracks, done = [], []
    for y in range(H - 1 - row_step // 2, -1, -row_step):
        row = mask[y] > 0
        d = np.diff(np.concatenate([[0], row.view(np.int8), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        runs = [((s + e - 1) / 2, e - s) for s, e in zip(starts, ends) if 2 <= e - s <= max_run]
        used = set()
        for tr in sorted(tracks, key=lambda t: -len(t["pts"])):
            pts = tr["pts"]
            if len(pts) >= 6:
                ys = np.float64([p[1] for p in pts[-12:]])
                xs = np.float64([p[0] for p in pts[-12:]])
                slope = np.polyfit(ys, xs, 1)[0] if np.ptp(ys) > 0 else 0.0
            else:
                slope = 0.0 if len(pts) < 2 else (pts[-1][0] - pts[0][0]) / min(-1e-6, pts[-1][1] - pts[0][1])
            gap = pts[-1][1] - y
            pred = pts[-1][0] + slope * (y - pts[-1][1])
            tol = 6 + 0.04 * gap + 0.004 * W
            best = None
            for i, (x, w) in enumerate(runs):
                if i in used:
                    continue
                if abs(x - pred) <= tol + w / 2 and (best is None or abs(x - pred) < abs(runs[best][0] - pred)):
                    best = i
            if best is not None:
                used.add(best)
                pts.append((runs[best][0], y))
                tr["hits"] += 1
                tr["miss"] = 0
            else:
                tr["miss"] += 1
        for i, (x, w) in enumerate(runs):
            if i not in used:
                tracks.append({"pts": [(x, y)], "hits": 1, "miss": 0, "y0": y})
        max_miss = int(0.22 * H / row_step)
        keep = []
        for tr in tracks:
            (keep if tr["miss"] <= max_miss else done).append(tr)
        tracks = keep
    done += tracks
    out = []
    for tr in done:
        y_hi, y_lo = tr["pts"][0][1], tr["pts"][-1][1]   # bottom, top of the traced line
        rows = int((y_hi - y_lo) / row_step) + 1
        if tr["hits"] < 8:
            continue
        out.append({"pts": tr["pts"], "rows": rows, "hits": tr["hits"], "y_bottom": y_hi, "y_top": y_lo})
    return out


def fit_lines(traced, shape, min_span=0.25, log=print):
    """Polynomial x(y) per traced line, long ones only, sorted left to right at the picture's lower part."""
    H, W = shape[:2]
    lines = []
    for tr in traced:
        span = (tr["y_bottom"] - tr["y_top"]) / H
        if span < min_span:
            continue
        pts = np.float64(tr["pts"])
        deg = 2 if span > 0.4 else 1
        coef = np.polyfit(pts[:, 1], pts[:, 0], deg)
        resid = np.abs(np.polyval(coef, pts[:, 1]) - pts[:, 0])
        if np.median(resid) > 0.004 * W:
            continue   # not a smooth line
        slope = np.polyval(np.polyder(coef), (tr["y_bottom"] + tr["y_top"]) / 2)
        if abs(slope) > 2.5:
            continue   # too flat: not a line along the road
        coverage = tr["hits"] / max(1, tr["rows"])
        lines.append({"coef": coef, "y_top": float(tr["y_top"]), "y_bottom": float(tr["y_bottom"]),
                      "coverage": round(float(coverage), 2), "span": round(float(span), 2)})
    return merge_lines(lines, W, H)


def merge_lines(lines, W, H):
    """One entry per painted line: pieces of the same line (a dashed line traced in parts, a solid line broken by a
    shadow) are joined when they continue each other."""
    lines = sorted(lines, key=lambda l: -l["span"])
    merged = []
    for ln in lines:
        for m in merged:
            lo, hi = max(ln["y_top"], m["y_top"]), min(ln["y_bottom"], m["y_bottom"])
            ys = np.linspace(lo, hi, 20) if hi > lo else np.float64([ln["y_top"], ln["y_bottom"]])
            if np.max(np.abs(np.polyval(ln["coef"], ys) - np.polyval(m["coef"], ys))) < 0.012 * W:
                rows_m = m["coverage"] * (m["y_bottom"] - m["y_top"])
                rows_l = ln["coverage"] * (ln["y_bottom"] - ln["y_top"])
                m["y_top"], m["y_bottom"] = min(m["y_top"], ln["y_top"]), max(m["y_bottom"], ln["y_bottom"])
                if hi <= lo:   # pieces one after the other: the coverage adds up
                    m["coverage"] = round(min(1.0, (rows_m + rows_l) / (m["y_bottom"] - m["y_top"])), 2)
                m["span"] = round((m["y_bottom"] - m["y_top"]) / H, 2)
                break
        else:
            merged.append(dict(ln))
    return merged




def paint_profile(ln, tophat, row_step=2):
    """Rows of a line and whether each has paint: the brightest top-hat value next to the curve, compared with the
    line's own typical paint strength (worn paint is fainter but still continuous)."""
    H, W = tophat.shape
    tol = max(4, int(0.006 * W))
    ys = np.arange(int(ln["y_top"]), int(ln["y_bottom"]) + 1, row_step)
    xs = np.round(np.polyval(ln["coef"], ys)).astype(int)
    inside = (xs - tol >= 0) & (xs + tol < W)
    ys, xs = ys[inside], xs[inside]
    if not len(ys):
        return ys, np.zeros(0, bool)
    v = np.float32([tophat[y, x - tol:x + tol + 1].max() for y, x in zip(ys, xs)])
    strength = float(np.percentile(v, 75))
    return ys, v > max(12.0, 0.4 * strength)


def classify_sections(lines, tophat, solid_coverage, row_step=2):
    """Splits each line into SOLID and BROKEN stretches. A gap in the paint longer than 3% of the picture height
    separates stretches; a stretch at least 20% of the picture high and painted over `solid_coverage` of it is solid."""
    H = tophat.shape[0]
    gap_rows = max(3, int(0.03 * H / row_step))
    for ln in lines:
        ys, painted = paint_profile(ln, tophat, row_step)
        ln["sections"] = []
        if not len(ys) or not painted.any():
            ln["coverage"] = 0.0
            continue
        idx = np.flatnonzero(painted)
        ys, painted = ys[idx[0]:idx[-1] + 1], painted[idx[0]:idx[-1] + 1]   # trim unpainted ends
        ln["y_top"], ln["y_bottom"] = float(ys[0]), float(ys[-1])
        ln["coverage"] = round(float(painted.mean()), 2)
        stretches, start, run = [], 0, 0
        for i, p in enumerate(painted):
            if p:
                if run >= gap_rows:
                    stretches.append((start, i - run))
                    start = i
                run = 0
            else:
                run += 1
        stretches.append((start, len(painted) - 1))
        for a, b in stretches:
            span = (ys[b] - ys[a]) / H
            cov = float(painted[a:b + 1].mean())
            kind = "solid" if span >= 0.2 and cov >= solid_coverage else "broken"
            sec = {"kind": kind, "y_top": float(ys[a]), "y_bottom": float(ys[b]), "coverage": round(cov, 2)}
            if ln["sections"] and ln["sections"][-1]["kind"] == kind == "broken":
                ln["sections"][-1]["y_bottom"] = sec["y_bottom"]   # dashes of one broken stretch
            else:
                ln["sections"].append(sec)
    return [ln for ln in lines if ln["sections"]]


def toward_vanishing_point(lines, shape, max_angle=15.0, log=print):
    """Keeps the lines that run toward the road's vanishing point (where the long lines' near-end directions meet):
    painted lane lines do, a row of people on the pavement or a pole does not."""
    import itertools
    import math

    H, W = shape[:2]

    def tangent(ln):
        y = ln["y_bottom"] - 0.25 * (ln["y_bottom"] - ln["y_top"])   # the near part of the line
        x = float(np.polyval(ln["coef"], y))
        dxdy = float(np.polyval(np.polyder(ln["coef"]), y))
        return np.array([x, y]), np.array([-dxdy, -1.0])               # pointing up the picture

    major = [ln for ln in lines if ln["y_bottom"] - ln["y_top"] >= 0.4 * H]
    meets = []
    for a, b in itertools.combinations(major, 2):
        (p1, d1), (p2, d2) = tangent(a), tangent(b)
        m = np.array([d1, -d2]).T
        if abs(np.linalg.det(m)) < 1e-6:
            continue
        s1, s2 = np.linalg.solve(m, p2 - p1)
        if s1 > 0 and s2 > 0:
            meets.append(p1 + s1 * d1)
    if not meets:
        return lines
    vp = np.median(np.float64(meets), axis=0)
    keep = []
    for ln in lines:
        p, d = tangent(ln)
        to = vp - p
        ang = math.degrees(math.acos(np.clip(d @ to / (np.linalg.norm(d) * np.linalg.norm(to) + 1e-9), -1, 1)))
        if ang <= max_angle:
            keep.append(ln)
        else:
            log(f"  left out a line at x={p[0]:.0f}, y={p[1]:.0f} (of {W}x{H}): it does not run along the road "
                f"({ang:.0f} deg off)")
    return keep


def x_at(ln, y):
    """x of a line at row y; beyond its traced rows the line continues straight along its end direction."""
    y = np.asarray(y, np.float64)
    out = np.polyval(ln["coef"], np.clip(y, ln["y_top"], ln["y_bottom"]))
    d = np.polyder(ln["coef"])
    for end in (ln["y_top"], ln["y_bottom"]):
        sel = (y < end) if end == ln["y_top"] else (y > end)
        out = np.where(sel, np.polyval(ln["coef"], end) + np.polyval(d, end) * (y - end), out)
    return out


def lane_spacing(lines, y):
    """Typical distance in pixels between neighbouring lines at row y."""
    xs = sorted(float(x_at(ln, y)) for ln in lines)
    gaps = np.diff(xs)
    return float(np.median(gaps)) if len(gaps) else None


def road_like(bg, lines, line, side, sp_fn):
    """Whether the band just beyond `line` (side -1 = left, +1 = right) looks like the lanes' asphalt."""
    H, W = bg.shape[:2]
    lab = cv2.cvtColor(bg, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[..., 0]
    tex = cv2.blur(np.abs(cv2.Laplacian(L, cv2.CV_32F)), (9, 9))
    lane_px, band_px = [], []
    ordered = sorted(lines, key=lambda l: float(x_at(l, 0.6 * H)))
    for y in np.linspace(line["y_top"], line["y_bottom"], 40).astype(int):
        sp = sp_fn(y)
        if not sp:
            continue
        for a, b in zip(ordered, ordered[1:]):   # lane centres
            xa, xb = float(x_at(a, y)), float(x_at(b, y))
            for x in np.linspace(xa + 0.3 * (xb - xa), xb - 0.3 * (xb - xa), 5).astype(int):
                if 0 <= x < W:
                    lane_px.append((L[y, x], tex[y, x], *lab[y, x, 1:]))
        x0 = float(x_at(line, y))
        for f in np.linspace(0.35, 0.8, 5):
            x = int(x0 + side * f * sp)
            if 0 <= x < W:
                band_px.append((L[y, x], tex[y, x], *lab[y, x, 1:]))
    if len(band_px) < 20 or len(lane_px) < 20:
        return None   # cannot tell (beyond the picture)
    lane, band = np.float32(lane_px), np.float32(band_px)
    dl = abs(np.median(band[:, 0]) - np.median(lane[:, 0]))
    dc = np.hypot(*(np.median(band[:, 2:], axis=0) - np.median(lane[:, 2:], axis=0)))
    rough = np.median(band[:, 1]) / max(1e-3, np.median(lane[:, 1]))
    return bool(dl < 18 and dc < 8 and rough < 1.8)


def chord_pieces(ln, y_top, y_bottom, sp_fn, H, tol_frac=0.05, min_rows=0.08, overlap=0.25):
    """Straight pieces along a curved line, each within tol_frac of a lane width of the curve, overlapping."""
    pieces = []
    yb = y_bottom
    while yb - y_top > min_rows * H * 0.5:
        best = max(y_top, yb - min_rows * H)
        y = best
        while y > y_top:
            y = max(y_top, y - 0.02 * H)
            ys = np.linspace(y, yb, 25)
            xs = x_at(ln, ys)
            chord = xs[0] + (xs[-1] - xs[0]) * (ys - ys[0]) / (ys[-1] - ys[0])
            sp = sp_fn((y + yb) / 2) or 0.15 * H
            if np.max(np.abs(xs - chord)) > tol_frac * sp:
                break
            best = y
        pieces.append((best, yb))
        if best <= y_top:
            break
        yb = best + overlap * (yb - best)
    return pieces


def build_zones(bg, lines, frame_size, lane_width, include_broken, include_edges, log=print):
    """Road zone + V5 lane lines (zone file dict) from the lines found in the work-resolution background."""
    H, W = bg.shape[:2]
    k = W / frame_size[0]   # work pixels per frame pixel
    lines.sort(key=lambda l: float(x_at(l, 0.6 * H)))   # in place: L1, L2, ... from left to right everywhere
    if len(lines) < 2:
        raise SystemExit(f"Only {len(lines)} lane line(s) found: cannot place a road zone. "
                         "Check lanes_detected.jpg; draw zones by hand (violation_tracker.py --calibrate).")
    major = [l for l in lines if (l["y_bottom"] - l["y_top"]) >= 0.3 * H] or lines

    def sp_fn(y):
        present = [l for l in lines if l["y_top"] - 0.05 * H <= y <= l["y_bottom"] + 0.05 * H]
        return lane_spacing(present if len(present) >= 2 else lines, y)

    left, right = lines[0], lines[-1]
    edges = {}
    for name, ln, side in (("left", left, -1), ("right", right, 1)):
        rl = road_like(bg, lines, ln, side, sp_fn)
        edges[name] = rl is False   # asphalt beyond it: a lane divider, the real edge is not in view / not found
        log(f"  {name}most line: " + ("road edge" if edges[name] else "lane divider (road continues beyond it)"
                                      if rl else "lane divider? (cannot see beyond it)"))
    y_far = max(0.0, min(l["y_top"] for l in major))
    y_near = float(H - 1)
    ext = {}
    for name in ("left", "right"):
        ext[name] = 0.35 if edges[name] else 1.0   # in lanes
    pts = []
    for y in (y_far, y_near):
        sp = sp_fn(y) or 0.15 * W
        xl = float(x_at(left, y)) - ext["left"] * sp
        xr = float(x_at(right, y)) + ext["right"] * sp
        pts.append((xl, xr))
    (fl, fr), (nl, nr) = pts
    poly = [[fl, y_far], [fr, y_far], [nr, y_near], [nl, y_near]]
    # width in lanes, measured at a row where most lines are present
    y_ref = float(np.median([(l["y_top"] + l["y_bottom"]) / 2 for l in major]))
    sp_ref = sp_fn(y_ref)
    xs = [float(x_at(l, y_ref)) for l in lines]
    lanes = sum(max(1, round((b - a) / sp_ref)) for a, b in zip(xs, xs[1:]))
    width_lanes = lanes + ext["left"] + ext["right"]
    width_m = round(width_lanes * lane_width, 2)
    # nominal length: lane widths shrink with distance (pinhole), the near distance assumes a ~50 degree lens
    sp_near, sp_far = sp_fn(y_near), sp_fn(y_far)
    view_w_m = W / sp_near * lane_width
    z_near = view_w_m / (2 * np.tan(np.radians(25)))
    length_m = round(float(max(10.0, z_near * (sp_near / max(1e-6, sp_far) - 1))), 1)
    zone = {"type": "road", "name": "Road (auto lanes)", "points": [[round(x / k, 1), round(y / k, 1)] for x, y in poly],
            "width_m": width_m, "length_m": length_m, "road_category": "city", "allowed_direction": "both",
            "speed_limit_kph": None, "banned_classes": [], "stopping_allowed": False, "scale_source": "auto",
            "lanes": lanes}
    log(f"  road zone: {lanes} lanes between the outer lines, {width_m} m wide (lane {lane_width} m), "
        f"nominal length {length_m} m")
    lane_lines, report = [], []
    for i, ln in enumerate(lines):
        is_edge = (ln is left and edges["left"]) or (ln is right and edges["right"])
        role = "edge" if is_edge else "divider"
        kinds = []
        for j, sec in enumerate(ln["sections"]):
            kinds.append(f"{sec['kind']} {sec['y_top'] / H:.0%}-{sec['y_bottom'] / H:.0%} of height "
                         f"(painted {sec['coverage']:.0%})")
            use = (sec["kind"] == "solid" or include_broken) and (role == "divider" or include_edges)
            if not use:
                continue
            for n, (ya, yb) in enumerate(chord_pieces(ln, sec["y_top"], sec["y_bottom"], sp_fn, H)):
                a, b = (float(x_at(ln, ya)), ya), (float(x_at(ln, yb)), yb)
                lane_lines.append({"name": f"L{i + 1}{'s' if sec['kind'] == 'solid' else 'b'}{j + 1}.{n + 1}",
                                   "points": [[round(a[0] / k, 1), round(a[1] / k, 1)],
                                              [round(b[0] / k, 1), round(b[1] / k, 1)]]})
        report.append({"line": f"L{i + 1}", "role": role, "sections": kinds})
        log(f"  L{i + 1} ({role}): " + "; ".join(kinds))
    cfg = {"frame_size": list(frame_size), "peak_hours": None, "rules": {}, "zones": [zone], "lane_lines": lane_lines,
           "auto_lanes": {"lane_width_m": lane_width, "lines": report,
                          "curves": [{"line": f"L{i + 1}", "coef": [float(c) for c in ln["coef"]],
                                      "y_top": ln["y_top"], "y_bottom": ln["y_bottom"], "work_scale": k}
                                     for i, ln in enumerate(lines)]}}
    return cfg


def draw_lanes(img_work, lines, cfg, k):
    """Picture of what was found: solid stretches red, broken blue, road zone cyan, V5 pieces white."""
    out = img_work.copy()
    H = out.shape[0]
    for i, ln in enumerate(lines):
        for sec in ln["sections"]:
            ys = np.arange(int(sec["y_top"]), int(sec["y_bottom"]) + 1, 3)
            xs = np.polyval(ln["coef"], ys)
            col = (0, 0, 255) if sec["kind"] == "solid" else (255, 140, 0)
            cv2.polylines(out, [np.int32(np.stack([xs, ys], 1))], False, col, 6, cv2.LINE_AA)
        y = int((ln["y_top"] + ln["y_bottom"]) / 2)
        cv2.putText(out, f"L{i + 1}", (int(x_at(ln, y)) + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    poly = np.float32(cfg["zones"][0]["points"]) * k
    cv2.polylines(out, [np.int32(poly)], True, (255, 255, 0), 3, cv2.LINE_AA)
    for ll in cfg["lane_lines"]:
        a, b = (tuple(int(v * k) for v in p) for p in ll["points"])
        cv2.line(out, a, b, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, a, 6, (255, 255, 255), -1)
    cv2.putText(out, "red = solid line, blue = broken, white = checked for crossing (V5), cyan = road zone",
                (20, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def detect_lanes(args, out, log=print):
    log("Lane detection: building a picture of the empty road...")
    bg, small_ref, (W, H), ref_t = build_background(args.source, VISION / "models" / "bd_vehicles_best.pt",
                                                     args.imgsz, args.sample_sec, args.sample_frames, args.ref_at,
                                                     log=log)
    cv2.imwrite(str(out / "background.jpg"), bg)
    mask, tophat = paint_mask(bg)
    lines = fit_lines(trace_lines(mask), bg.shape)
    lines = classify_sections(lines, tophat, args.solid_coverage)
    lines = toward_vanishing_point(lines, bg.shape, log=log)
    log(f"  {len(lines)} painted lines found")
    cfg = build_zones(bg, lines, (W, H), args.lane_width, args.include_broken, args.include_edges, log=log)
    cfg["auto_lanes"]["reference_s"] = round(ref_t, 2)
    (out / "lanes.json").write_text(json.dumps(cfg, indent=2))
    cv2.imwrite(str(out / "lanes_detected.jpg"), draw_lanes(small_ref, lines, cfg, bg.shape[1] / W))
    log(f"  {len(cfg['lane_lines'])} line pieces checked for crossing -> {out / 'lanes.json'}, "
        f"{out / 'lanes_detected.jpg'}")
    if not cfg["lane_lines"]:
        log("  No solid lane line between the road edges: nothing to check. "
            "(--include-broken counts broken lines, --include-edges the edge lines.)")
    # The camera-movement correction compares every frame with this reference. The empty road (no vehicles) is the
    # right one: matched against the first frame, the first frames (no vehicle mask yet) lock onto the vehicles
    # (on the test video: "zoom 1.085" while the camera moved ~70 px), and the lines then land beside the paint.
    return out / "lanes.json", cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------- 3. violations + plates

def plate_writer(out, plate_rows, lock, log=print):
    """WebPlateCollector's on_result: saves the plate crop and the "vehicle + plate" image, keeps the read."""
    def on_result(key, r, plate_img, evidence_img):
        row = {"plate": r.get("registration") or r.get("number") or "", "number": r.get("number", ""),
               "plate_status": r.get("status", ""), "plate_confidence": round(float(r.get("conf") or 0), 2),
               "agreeing_reads": r.get("agree", 0), "reads": r.get("reads", 0),
               "plate_px": r.get("plate_px", 0), "plate_image": "", "vehicle_plate_image": ""}
        if plate_img is not None and plate_img.size:
            row["plate_image"] = f"plates/{key}_plate.jpg"
            cv2.imwrite(str(out / row["plate_image"]), plate_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if evidence_img is not None and evidence_img.size:
            row["vehicle_plate_image"] = f"plates/{key}_vehicle_plate.jpg"
            cv2.imwrite(str(out / row["vehicle_plate_image"]), evidence_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with lock:
            plate_rows[key] = row
        log(f"\n[plate] {key}: {row['plate'] or '-'} ({row['plate_status']})")
    return on_result


def line_points(camera, motion=None, n=24):
    """Points along every checked lane line, in the current frame's pixels (zones are in the reference view)."""
    pts = np.concatenate([np.linspace(ln.pts[0], ln.pts[1], n) for ln in camera.lines]).astype(np.float32)
    return motion.to_frame(pts) if motion is not None else pts


def paint_at(frame, pts, tol=6):
    """Top-hat paint response next to each point (frame pixels; measured at WORK_W), NaN outside the picture."""
    H, W = frame.shape[:2]
    k = WORK_W / W
    small = cv2.resize(frame, (WORK_W, round(H * k)), interpolation=cv2.INTER_AREA)
    _, tophat = paint_mask(small)
    out = np.full(len(pts), np.nan, np.float32)
    for i, (x, y) in enumerate(pts * k):
        xi, yi = int(round(x)), int(round(y))
        if tol <= xi < small.shape[1] - tol and 0 <= yi < small.shape[0]:
            out[i] = tophat[yi, xi - tol:xi + tol + 1].max()
    return out


def lane_alignment(frame, camera, motion, boxes, strength):
    """Frame-by-frame check of the road lock: share of the lane-line points, mapped into this frame, that lie on
    paint seen in this frame (points under a vehicle or outside the picture left out). None: too few to judge."""
    pts = line_points(camera, motion)
    free = np.ones(len(pts), bool)
    for x1, y1, x2, y2 in boxes:
        free &= ~((pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2))
    v = paint_at(frame, pts)
    ok = free & ~np.isnan(v)
    if ok.sum() < 10:
        return None
    return round(float((v[ok] > max(12.0, 0.4 * strength)).mean()), 2)


def run(args, zones_path, ref, out, log=print):
    import threading

    from surveillance.detector import Detector
    from surveillance.pipeline import Pipeline
    from surveillance.sources import FileSource
    from violations import load_camera_config
    from violations.evidence import EvidenceWriter

    source = FileSource(args.source, args.analysis_fps, 0, False).open()
    camera = load_camera_config(zones_path, source.width, source.height)
    if not camera.lines:
        log("No lane lines to check: stopping.")
        source.close()
        return
    if ref is not None and ref.shape[:2] != (source.height, source.width):
        ref = None
    detector = Detector(str(VISION / "models" / "bd_vehicles_best.pt"), args.imgsz, "pytorch", tracker="bytetrack",
                        fps=source.fps)

    plates, lock, plate_rows = None, threading.Lock(), {}
    if not args.no_plates:
        from service import config as vconfig
        from service import plates as sp
        if not vconfig.PLATE_MODEL.exists():
            raise SystemExit(f"Plate detector missing: {vconfig.PLATE_MODEL}\n  wget -O {vconfig.PLATE_MODEL} "
                             "https://github.com/prosensarker4/BLPD/raw/main/LPD_Model.pt   (or use --no-plates)")
        sp.CLOCK_OVERLAY = (0.0, 0.0, 0.0, 0.0)   # the Kakoli camera's clock corner; a phone video has none
        (out / "plates").mkdir(exist_ok=True)

        plates = sp.WebPlateCollector(sp.PlateModels(), plate_writer(out, plate_rows, lock, log), live=False)

    # A zoomed-in stretch of a handheld video shows only part of the road the zones were drawn on: keep checking
    # the part in view instead of pausing (the frame-by-frame alignment check below watches the lock).
    import surveillance.camera_motion as camera_motion
    camera_motion.MIN_IN_VIEW = 0.2
    overrides = {"lane_line": {"margin_m": args.margin, "hold_sec": args.hold}}
    pipe = Pipeline(detector, camera, (source.width, source.height), None, False, None, overrides, ["V5"],
                    auto_roi=True, display_width=args.video_width, plates=plates, reference=ref,
                    stabilize=not args.no_stabilize)
    if not args.no_stabilize and ref is not None:
        # The road lock. The pipeline's own setup leaves the road zones out of the reference's features (for a fixed
        # camera they are full of traffic), but the reference here is the EMPTY road: its paint and asphalt are the
        # best features there are. Without them a handheld video's first 13 s could not be matched and were taken
        # as "unverified, assume unmoved" (lines 0.1-0.5 on the paint), so: road features in, an unverified view
        # is not trusted (a hand-held camera is never steady), and the clean reference is never swapped for a live
        # frame (re-anchoring).
        from concurrent.futures import ThreadPoolExecutor

        class RoadLock(camera_motion.CameraMotion):
            @property
            def enforce(self):
                return self.state in ("stable", "compensated")

        camera_motion.REANCHOR_SEC = float("inf")
        pipe.motion = RoadLock(ref, [r.poly for r in camera.roads] + [a.poly for a in camera.areas])
        pipe.motion._set_reference(ref, [])
        pipe._pool = ThreadPoolExecutor(1, thread_name_prefix="camera-motion")
    # Only sideways positions matter for V5, and the zone's length is nominal: trust positions in the whole zone.
    pipe.reliable_y = {id(r): 0.0 for r in camera.roads}
    for r in camera.roads:
        r.reliable_from_m = 0.0
    log("Road zones:\n  " + "\n  ".join(z.describe() for z in camera.roads))
    log(f"Lane line pieces checked: {len(camera.lines)} | number plates: {'on' if plates else 'off'}")
    for code, why in sorted(pipe.engine.inactive.items()):
        log(f"  {code} off: {why}")

    evidence = EvidenceWriter(out)
    writer, raw_video = None, out / "annotated_raw.mp4"
    events = []
    # paint strength of the lines in the reference view (the empty road), for the alignment check
    strength = float(np.nanmedian(paint_at(ref, line_points(camera)))) if ref is not None else 40.0
    align_rows, last_align, every = [], None, max(1, round(source.fps))
    frames, t0, last_print = 0, time.time(), 0.0
    try:
        while True:
            got = source.read()
            if got is None:
                break
            frame, t = got
            if args.end is not None and t > args.end:
                break
            res = pipe.process(frame, t)
            frames += 1
            if frames % every == 1 or every == 1:
                m = pipe.motion
                if m is not None and not m.enforce:
                    last_align = None
                else:
                    last_align = lane_alignment(frame, camera, m, pipe._boxes, strength)
                d = m.describe() if m is not None else {}
                align_rows.append({"video_time_s": round(t, 2), "camera": d.get("state", "fixed"),
                                   "shift_px": d.get("shift_px", 0), "zoom": d.get("zoom", 1),
                                   "lines_on_paint": "" if last_align is None else last_align})
            for ev in res.events:
                ev_frame = frame
                if ev.frame is not None:
                    ev_frame = cv2.imdecode(np.frombuffer(ev.frame, np.uint8), cv2.IMREAD_COLOR)
                stem = evidence.write(ev, ev_frame)
                events.append((stem, ev, last_align))
                if plates:
                    plates.request(ev.obs.tid, stem, t)
                log(f"\n[{ev.code}] {ev.obs.cls} #{ev.obs.tid} crossed {ev.details.get('line')} at {t:.1f}s")
            if not args.no_video:
                if writer is None:
                    h, w = res.display.shape[:2]
                    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), source.fps, (w, h))
                writer.write(res.display)
            if time.time() - last_print > 2:
                last_print = time.time()
                prog = source.progress()
                print(f"\r{frames} frames | {frames / (time.time() - t0):4.1f} FPS | "
                      f"{f'{100 * prog:5.1f}% | ' if prog is not None else ''}{len(events)} violations   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        log("\nStopped - finishing the output files...")
    print()
    if plates:
        log("Reading the violators' number plates...")
        plates.flush()
    evidence.close()
    pipe.close()
    source.close()
    detector.close()
    if writer is not None:
        writer.release()
        finish_video(raw_video, out / "annotated.mp4", log)

    with open(out / "alignment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_time_s", "camera", "shift_px", "zoom", "lines_on_paint"])
        w.writeheader()
        w.writerows(align_rows)
    scores = [r["lines_on_paint"] for r in align_rows if r["lines_on_paint"] != ""]
    if scores:
        bad = [r["video_time_s"] for r in align_rows if r["lines_on_paint"] != "" and r["lines_on_paint"] < 0.6]
        log(f"Road lock check (alignment.csv): lane lines on the paint in {np.mean(scores):.0%} of the points on "
            f"average; {len(bad)} of {len(scores)} checked seconds below 60%"
            + (f" (at {', '.join(f'{b:.0f}s' for b in bad[:15])}{' ...' if len(bad) > 15 else ''})" if bad else ""))
    fields = ["id", "violation", "track_id", "class", "video_time_s", "line", "lane_alignment", "plate", "plate_status",
              "plate_confidence", "agreeing_reads", "reads", "plate_px", "snapshot", "closeup", "plate_image",
              "vehicle_plate_image"]
    with open(out / "plates.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for stem, ev, align in events:
            row = {"id": stem, "violation": f"{ev.code} {ev.title}", "track_id": ev.obs.tid, "class": ev.obs.cls,
                   "video_time_s": round(ev.obs.time_s, 2), "line": ev.details.get("line", ""),
                   "lane_alignment": "" if align is None else align,
                   "snapshot": f"snapshots/{stem}.jpg", "closeup": f"snapshots/{stem}_crop.jpg",
                   **plate_rows.get(stem, {"plate_status": "not read" if plates else "plates off"})}
            w.writerow(row)
    log(f"\nFrames: {frames} | avg {frames / max(1e-6, time.time() - t0):.1f} FPS")
    log(f"===== LANE-LINE VIOLATIONS ({len(events)}) =====")
    for stem, ev, _ in events:
        r = plate_rows.get(stem, {})
        log(f"{stem}: {ev.obs.cls} #{ev.obs.tid} at {ev.obs.time_s:.1f}s, line {ev.details.get('line')}, "
            f"plate {r.get('plate') or '-'} ({r.get('plate_status', 'not read')})")
    log(f"Results: {out / 'plates.csv'} (one row per violation), {out / 'snapshots'}, {out / 'plates'}"
        + ("" if args.no_video else f", {out / 'annotated.mp4'}"))


def finish_video(raw, final, log):
    """H.264 (plays in browsers and on phones) when ffmpeg is installed, else the raw MPEG-4 file."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset", "fast",
                            "-crf", "23", "-pix_fmt", "yuv420p", str(final)])
        if r.returncode == 0:
            raw.unlink(missing_ok=True)
            return
    raw.replace(final)
    log("(annotated.mp4 is MPEG-4 part 2: play it with VLC)")


def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore", message="All-NaN slice")
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.zones:
        zones_path, ref = Path(args.zones), None
        bg = cv2.imread(str(zones_path.parent / "background.jpg"))   # saved with a lane detection: the road lock
        if bg is not None:
            W, H = json.loads(zones_path.read_text())["frame_size"]
            ref = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        zones_path, ref = detect_lanes(args, out)
    if args.lanes_only:
        return
    run(args, zones_path, ref, out)


if __name__ == "__main__":
    main()
