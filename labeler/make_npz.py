#!/usr/bin/env python3
"""
make_npz.py - create guidance npz files for the labeller (YOLOX + RTMPose, run on your own computer).

For every video: read every N-th frame (default 2), detect people with YOLOX, (optionally) find 26 body keypoints
with RTMPose, then save <video_name>.npz in the SAME format as Stage 2 of the FreezingOfGait-SelfAnalysis project
(notebook/2_detector_pose.ipynb):
    det_frame (K,)  det_boxes (K,4)  det_scores (K,)  det_kpts (K,26,2)  det_kpt_scores (K,26)
    frames_processed (T,)  meta (JSON)
The detection functions (decode_yolox, nms, clean_dets) and frame reading (FrameReader) are copied from Stage 2.

Videos that already have a matching npz are skipped, so it is safe to stop and run again.

Examples (from the repository folder):
  python3 labeler/make_npz.py --videos datasets/example_dataset/videos --out datasets/example_dataset/npz
  python3 labeler/make_npz.py --videos ... --out ... --no-pose             # faster, no skeleton
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("Missing packages. Run: pip install opencv-python numpy")

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
N_KPTS = 26
HALPE26_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                 "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle", "head", "neck", "hip",
                 "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe", "left_heel", "right_heel"]


def say(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------- detection (copied from Stage 2)
def nms(boxes, scores, thr):
    if len(boxes) == 0:
        return np.zeros(0, int)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter + 1e-9) <= thr]
    return np.asarray(keep, int)


def decode_yolox(outputs, ratio, input_size, score_thr, nms_thr):
    """ONNX YOLOX output -> (boxes Nx4, scores N) for class 'person'.
    Handles end2end models with NMS inside (dets (1,N,5) [+ labels]) and raw models (1,A,5+C)."""
    out = np.asarray(outputs[0])
    if out.shape[-1] == 5:
        dets = out.reshape(-1, 5).astype(np.float32)
        boxes, scores = dets[:, :4] / ratio, dets[:, 4]
        if len(outputs) > 1:
            labels = np.asarray(outputs[1]).reshape(-1)
            if labels.shape[0] == boxes.shape[0]:
                boxes, scores = boxes[labels == 0], scores[labels == 0]
        m = scores >= score_thr
        return boxes[m].astype(np.float32), scores[m].astype(np.float32)
    pred = out.reshape(-1, out.shape[-1]).astype(np.float32).copy()
    grids, strides = [], []
    for s in (8, 16, 32):
        hs, ws = input_size[0] // s, input_size[1] // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2)); strides.append(np.full((hs * ws, 1), s))
    grids, strides = np.concatenate(grids), np.concatenate(strides)
    pred[:, :2] = (pred[:, :2] + grids) * strides
    pred[:, 2:4] = np.exp(pred[:, 2:4]) * strides
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) / ratio
    scores = pred[:, 4] * pred[:, 5]
    m = scores >= score_thr
    boxes, scores = boxes[m], scores[m]
    keep = nms(boxes, scores, nms_thr)
    return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)


def clean_dets(boxes, scores, w, h, max_dets, min_size=4):
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    ok = ((boxes[:, 2] - boxes[:, 0]) >= min_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_size)
    boxes, scores = boxes[ok], scores[ok]
    order = np.argsort(-scores, kind="stable")[:max_dets]
    return boxes[order], scores[order]


def use_certifi():
    """Python from python.org on macOS often has no SSL certificates yet, so model downloads fail
    (CERTIFICATE_VERIFY_FAILED). Called ONLY after a download failed: use the certificates from the certifi package.
    Returns True if the setting changed (worth retrying)."""
    if os.environ.get("SSL_CERT_FILE"):
        return False
    try:
        import certifi
    except ImportError:
        return False
    os.environ["SSL_CERT_FILE"] = certifi.where()
    return True


def available_devices():
    """Devices actually available to onnxruntime on this computer. cpu is always there."""
    try:
        import onnxruntime as ort
        prov = ort.get_available_providers()
    except Exception:
        return ["cpu"]
    devs = []
    if "CUDAExecutionProvider" in prov:
        devs.append("cuda")
    if "CoreMLExecutionProvider" in prov:
        devs.append("mps")
    return devs + ["cpu"]


def active_providers(model):
    try:
        return model.session.get_providers()
    except Exception:
        return ["(unknown)"]


def build_models(model, device, det_thr, nms_thr, with_pose):
    try:
        from rtmlib import YOLOX, RTMPose, BodyWithFeet
    except ImportError:
        sys.exit("rtmlib is not installed. Run: pip install rtmlib onnxruntime")

    class YOLOXWithScores(YOLOX):
        """rtmlib's YOLOX drops the scores and uses a fixed 0.3 threshold; ByteTrack needs both."""
        def __call__(self, image):
            img, ratio = self.preprocess(image)
            return decode_yolox(self.inference(img), ratio, self.model_input_size, self.score_thr, self.nms_thr)

    cfg = BodyWithFeet.MODE[model]
    det = YOLOXWithScores(cfg["det"], model_input_size=cfg["det_input_size"], nms_thr=nms_thr,
                          score_thr=det_thr, backend="onnxruntime", device=device)
    pose = None
    if with_pose:
        pose = RTMPose(cfg["pose"], model_input_size=cfg["pose_input_size"], to_openpose=False,
                       backend="onnxruntime", device=device)
    return det, pose, cfg


# ----------------------------------------------------------------------------- video (copied from Stage 2)
def probe_video(path):
    """(fps, n_frames): ffprobe first, OpenCV as fallback (same rule as the Stage 1 labelling tool)."""
    fps = nb = None
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=avg_frame_rate,nb_frames", "-of", "json", str(path)],
                           capture_output=True, text=True, timeout=60)
        s = json.loads(r.stdout)["streams"][0]
        num, den = (float(x) for x in s.get("avg_frame_rate", "0/1").split("/"))
        fps = num / den if num > 0 and den > 0 else None
        nb_s = str(s.get("nb_frames", ""))
        nb = int(nb_s) if nb_s.isdigit() and int(nb_s) > 0 else None
    except Exception:
        pass
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = fps or cap.get(cv2.CAP_PROP_FPS)
    nb = nb or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or fps <= 0 or fps > 1000:
        fps = 30.0
    return float(fps), int(nb)


class FrameReader(threading.Thread):
    """Decodes frames in the background. Reads SEQUENTIALLY (grab/retrieve), exactly like the
    Stage 1 labelling tool, so frame_idx values match."""
    def __init__(self, path, stride, maxsize=16):
        super().__init__(daemon=True)
        self.path, self.stride, self.q, self.n_read, self.error = path, stride, queue.Queue(maxsize), 0, None
        self.stop = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(str(self.path))
        i = 0
        try:
            while not self.stop.is_set() and cap.grab():
                if i % self.stride == 0:
                    ok, fr = cap.retrieve()
                    if ok:
                        self.q.put((i, fr))
                i += 1
        except Exception as e:
            self.error = e
        finally:
            cap.release()
            self.n_read = i
            self.q.put((None, None))

    def __iter__(self):
        while True:
            i, fr = self.q.get()
            if i is None:
                if self.error:
                    raise self.error
                return
            yield i, fr


def save_npz_atomic(path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def fmt_sec(s):
    s = int(max(s, 0))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


# ----------------------------------------------------------------------------- one video
def process_video(path, out_dir, det, pose, cfg, args, k, n_vid):
    fps, nb = probe_video(path)
    total = (nb + args.stride - 1) // args.stride if nb else 0
    fr_l, bx_l, sc_l, kp_l, ks_l, done = [], [], [], [], [], []
    W = H = None
    t0 = last = time.time()
    reader = FrameReader(path, args.stride)
    reader.start()
    try:
        for i, frame in reader:
            if H is None:
                H, W = frame.shape[:2]
            boxes, scores = clean_dets(*det(frame), W, H, args.max_dets)
            n = len(boxes)
            kp = np.full((n, N_KPTS, 2), np.nan, np.float32)
            ks = np.zeros((n, N_KPTS), np.float32)
            need = scores >= args.pose_min_score
            if pose is not None and need.any():
                # rtmlib processes the whole image when given no boxes, so only call it when there are boxes
                k_, s_ = pose(frame, [b.tolist() for b in boxes[need]])
                k_, s_ = np.asarray(k_, np.float32), np.asarray(s_, np.float32)
                if k_.shape[1:] != (N_KPTS, 2):
                    raise RuntimeError(f"pose model returned shape {k_.shape}, expected (N, 26, 2) Halpe-26")
                kp[need], ks[need] = k_, s_
            done.append(i)
            if n:
                fr_l.append(np.full(n, i, np.int32)); bx_l.append(boxes); sc_l.append(scores)
                kp_l.append(kp); ks_l.append(ks)
            now = time.time()
            if now - last >= 2:
                last = now
                m = len(done)
                rate = m / max(now - t0, 1e-6)
                if total:
                    say(f"[PROGRESS] video {k}/{n_vid} {path.stem}: {m}/{total} steps "
                        f"({100 * m / total:.0f}%), {rate:.1f} steps/s, ~{fmt_sec((total - m) / max(rate, 1e-6))} left")
                else:
                    say(f"[PROGRESS] video {k}/{n_vid} {path.stem}: {m} steps, {rate:.1f} steps/s")
    finally:
        reader.stop.set()
        while reader.is_alive():                 # release a reader waiting on a full queue
            try:
                reader.q.get_nowait()
            except queue.Empty:
                pass
            reader.join(0.05)
    if not done:
        raise IOError("no frames could be read")
    secs = time.time() - t0
    cat = lambda lst, shape, dt: np.concatenate(lst).astype(dt) if lst else np.zeros(shape, dt)
    meta = dict(video=path.name, video_id=path.stem, source_path=str(path), fps=fps,
                n_frames_read=reader.n_read, n_frames_metadata=nb, width=W, height=H,
                keypoint_format="halpe26", keypoint_names=HALPE26_NAMES,
                det_model=cfg["det"], pose_model=cfg["pose"] if pose is not None else None,
                with_pose=pose is not None, pose_min_score=args.pose_min_score,
                model=args.model, device=args.device, stride=args.stride,
                det_thr=args.det_thr, nms_thr=args.nms_thr, max_dets=args.max_dets,
                seconds=round(secs, 1), n_frames_processed=len(done), n_dets=int(sum(len(x) for x in fr_l)),
                created=datetime.now().isoformat(timespec="seconds"), tool="dense-person-labeler/labeler/make_npz.py")
    save_npz_atomic(out_dir / f"{path.stem}.npz",
                    det_frame=cat(fr_l, (0,), np.int32), det_boxes=cat(bx_l, (0, 4), np.float32),
                    det_scores=cat(sc_l, (0,), np.float32), det_kpts=cat(kp_l, (0, N_KPTS, 2), np.float32),
                    det_kpt_scores=cat(ks_l, (0, N_KPTS), np.float32),
                    frames_processed=np.asarray(done, np.int32), meta=np.array(json.dumps(meta)))
    return len(done), secs


def find_videos(folder, only=None):
    vids = {}
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix in VIDEO_EXT and not p.name.startswith("._") and (not only or only in p.stem):
            vids.setdefault(p.stem, p)
    return dict(sorted(vids.items()))


def main():
    ap = argparse.ArgumentParser(description="Create guidance npz files (YOLOX + RTMPose) for the dense labeller.")
    ap.add_argument("--videos", required=True, help="video folder (searched recursively)")
    ap.add_argument("--out", required=True, help="output folder for npz files")
    ap.add_argument("--model", default="balanced", choices=["lightweight", "balanced", "performance"],
                    help="rtmlib model size (default balanced)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="auto (cuda if available, otherwise cpu), cpu, mps (Mac, via CoreML), cuda (NVIDIA GPU)")
    ap.add_argument("--stride", type=int, default=2, help="process every N-th frame (default 2, same as the labels)")
    ap.add_argument("--det-thr", type=float, default=0.10, help="YOLOX score threshold (default 0.10)")
    ap.add_argument("--nms-thr", type=float, default=0.45)
    ap.add_argument("--max-dets", type=int, default=30, help="max boxes per frame (default 30)")
    ap.add_argument("--no-pose", action="store_true", help="skip RTMPose (faster; skeleton K not available)")
    ap.add_argument("--pose-min-score", type=float, default=0.3,
                    help="RTMPose only for boxes with score >= this (default 0.3 = the boxes shown in the labeller; "
                         "0.1 = all boxes, much slower)")
    ap.add_argument("--redo", action="store_true", help="recreate even if an npz already exists")
    ap.add_argument("--only", help="only videos whose name contains this text")
    args = ap.parse_args()

    videos = find_videos(args.videos, args.only)
    if not videos:
        sys.exit(f"No videos in {args.videos}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for t in out.glob("*.npz.tmp"):            # leftovers of a run stopped half-way
        try:
            t.unlink()
        except OSError:
            pass
    if args.stride < 1:
        sys.exit("--stride must be at least 1")

    def usable(v):
        """An existing npz is reused if its frame spacing divides --stride (npz every 2 frames fits 2, 4, 6, ...)."""
        p = out / f"{v}.npz"
        if args.redo or not p.exists():
            return False
        try:
            with np.load(p, allow_pickle=False) as d:
                fp = d["frames_processed"]
            s = int(np.median(np.diff(fp))) if len(fp) > 1 else 1
        except Exception:
            return False
        return args.stride % s == 0

    todo = [(v, p) for v, p in videos.items() if not usable(v)]
    redo = [v for v, _ in todo if (out / f"{v}.npz").exists()]
    say(f"{len(videos)} videos, {len(videos) - len(todo)} already have a matching npz, {len(todo)} to process "
        f"(every {args.stride} frames).")
    if redo and not args.redo:
        say(f"{len(redo)} existing npz were made with a non-matching frame step and will be recreated: {', '.join(redo)}")
    if not todo:
        say("DONE: all videos already have an npz.")
        return
    avail = available_devices()
    if args.device == "auto":
        args.device = "cuda" if "cuda" in avail else "cpu"
    elif args.device not in avail:
        say(f"[WARNING] device {args.device} is not available on this computer (available: {', '.join(avail)}). Using cpu.")
        args.device = "cpu"
    say(f"Loading YOLOX{' + RTMPose' if not args.no_pose else ''} ({args.model}, {args.device}). "
        "First time: the models are downloaded (needs internet).")

    def is_download_error(e):
        txt = f"{e.__class__.__name__}: {e}"
        return ("CERTIFICATE_VERIFY_FAILED" in txt or "urlopen error" in txt or "URLError" in txt), txt

    def load(device):
        """Load the models. If the download fails because of certificates, retry once with certifi's certificates."""
        try:
            return build_models(args.model, device, args.det_thr, args.nms_thr, not args.no_pose)
        except SystemExit:
            raise
        except Exception as e:
            dl, txt = is_download_error(e)
            if dl and "CERTIFICATE_VERIFY_FAILED" in txt and use_certifi():
                say("Python's default SSL certificates failed; retrying with certifi's certificates.")
                return load(device)
            if dl:
                say("[FAILED] The models could not be downloaded: " + txt)
                say("Check your internet connection (needs access to download.openmmlab.com or huggingface.co).\n"
                    "On macOS with Python from python.org, run once:\n"
                    '  open "/Applications/Python 3.12/Install Certificates.command"\n'
                    "(match your Python version), then try again.")
                sys.exit(1)
            raise

    try:
        det, pose, cfg = load(args.device)
    except SystemExit:
        raise
    except Exception as e:
        if args.device == "cpu":
            raise
        say(f"[WARNING] device {args.device} failed ({e.__class__.__name__}: {e}). Using cpu.")
        args.device = "cpu"
        det, pose, cfg = load("cpu")
    say(f"Devices in use: YOLOX {active_providers(det)}"
        + (f", RTMPose {active_providers(pose)}" if pose is not None else ""))

    t_all, failed = time.time(), []
    try:
        for k, (vid, path) in enumerate(todo, 1):
            say(f"[VIDEO] {k}/{len(todo)} {vid}")
            try:
                n, secs = process_video(path, out, det, pose, cfg, args, k, len(todo))
                say(f"[OK] {vid}: {n} steps in {fmt_sec(secs)} -> {out / (vid + '.npz')}")
            except Exception as e:
                failed.append(vid)
                say(f"[FAILED] {vid}: {e.__class__.__name__}: {e}")
    except KeyboardInterrupt:
        say("Stopped. Finished videos are kept; run again to continue.")
        sys.exit(130)
    say(f"DONE in {fmt_sec(time.time() - t_all)}. Succeeded {len(todo) - len(failed)}, failed {len(failed)}"
        + (f": {', '.join(failed)}" if failed else "."))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
