#!/usr/bin/env python3
"""
make_npz.py - membuat file npz panduan untuk alat label (YOLOX + RTMPose, dijalankan di komputer sendiri).

Untuk setiap video: baca setiap frame ke-2, deteksi orang dengan YOLOX, (opsional) cari 26 titik tubuh dengan
RTMPose, lalu simpan <nama_video>.npz dengan format yang SAMA dengan Stage 2 proyek FreezingOfGait-SelfAnalysis
(notebook/2_detector_pose.ipynb):
    det_frame (K,)  det_boxes (K,4)  det_scores (K,)  det_kpts (K,26,2)  det_kpt_scores (K,26)
    frames_processed (T,)  meta (JSON)
Fungsi deteksi (decode_yolox, nms, clean_dets) dan cara membaca frame (FrameReader) disalin dari Stage 2.

Video yang sudah punya npz dilewati, jadi aman dihentikan lalu dijalankan lagi.

Contoh (dari folder repo):
  python3 labeler/make_npz.py --videos datasets/example_dataset/videos --out datasets/example_dataset/npz
  python3 labeler/make_npz.py --videos ... --out ... --tanpa-pose          # lebih cepat, tanpa kerangka
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
    sys.exit("Paket kurang. Jalankan: pip install opencv-python numpy")

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
N_KPTS = 26
HALPE26_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                 "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle", "head", "neck", "hip",
                 "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe", "left_heel", "right_heel"]


def say(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------- deteksi (disalin dari Stage 2)
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
    """Python dari python.org di macOS sering belum punya sertifikat SSL, sehingga unduhan model gagal
    (CERTIFICATE_VERIFY_FAILED). Dipanggil HANYA setelah unduhan gagal: pakai sertifikat paket certifi.
    Mengembalikan True jika pengaturan berubah (layak dicoba lagi)."""
    if os.environ.get("SSL_CERT_FILE"):
        return False
    try:
        import certifi
    except ImportError:
        return False
    os.environ["SSL_CERT_FILE"] = certifi.where()
    return True


def available_devices():
    """Perangkat yang benar-benar tersedia di onnxruntime komputer ini. cpu selalu ada."""
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
        return ["(tidak diketahui)"]


def build_models(mode, device, det_thr, nms_thr, with_pose):
    try:
        from rtmlib import YOLOX, RTMPose, BodyWithFeet
    except ImportError:
        sys.exit("rtmlib belum terpasang. Jalankan: pip install rtmlib onnxruntime")

    class YOLOXWithScores(YOLOX):
        """rtmlib's YOLOX drops the scores and uses a fixed 0.3 threshold; ByteTrack needs both."""
        def __call__(self, image):
            img, ratio = self.preprocess(image)
            return decode_yolox(self.inference(img), ratio, self.model_input_size, self.score_thr, self.nms_thr)

    cfg = BodyWithFeet.MODE[mode]
    det = YOLOXWithScores(cfg["det"], model_input_size=cfg["det_input_size"], nms_thr=nms_thr,
                          score_thr=det_thr, backend="onnxruntime", device=device)
    pose = None
    if with_pose:
        pose = RTMPose(cfg["pose"], model_input_size=cfg["pose_input_size"], to_openpose=False,
                       backend="onnxruntime", device=device)
    return det, pose, cfg


# ----------------------------------------------------------------------------- video (disalin dari Stage 2)
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


# ----------------------------------------------------------------------------- satu video
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
                # rtmlib: tanpa kotak ia memproses seluruh gambar, jadi hanya dipanggil jika ada kotak
                k_, s_ = pose(frame, [b.tolist() for b in boxes[need]])
                k_, s_ = np.asarray(k_, np.float32), np.asarray(s_, np.float32)
                if k_.shape[1:] != (N_KPTS, 2):
                    raise RuntimeError(f"model pose memberi bentuk {k_.shape}, seharusnya (N, 26, 2) Halpe-26")
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
                    say(f"[PROGRES] video {k}/{n_vid} {path.stem}: {m}/{total} langkah "
                        f"({100 * m / total:.0f}%), {rate:.1f} langkah/s, sisa ~{fmt_sec((total - m) / max(rate, 1e-6))}")
                else:
                    say(f"[PROGRES] video {k}/{n_vid} {path.stem}: {m} langkah, {rate:.1f} langkah/s")
    finally:
        reader.stop.set()
        while reader.is_alive():                 # lepaskan pembaca yang menunggu antrean penuh
            try:
                reader.q.get_nowait()
            except queue.Empty:
                pass
            reader.join(0.05)
    if not done:
        raise IOError("tidak ada frame yang bisa dibaca")
    secs = time.time() - t0
    cat = lambda lst, shape, dt: np.concatenate(lst).astype(dt) if lst else np.zeros(shape, dt)
    meta = dict(video=path.name, video_id=path.stem, source_path=str(path), fps=fps,
                n_frames_read=reader.n_read, n_frames_metadata=nb, width=W, height=H,
                keypoint_format="halpe26", keypoint_names=HALPE26_NAMES,
                det_model=cfg["det"], pose_model=cfg["pose"] if pose is not None else None,
                with_pose=pose is not None, pose_min_score=args.pose_min_score,
                mode=args.mode, device=args.device, stride=args.stride,
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
    ap = argparse.ArgumentParser(description="Buat npz panduan (YOLOX + RTMPose) untuk alat label padat.")
    ap.add_argument("--videos", required=True, help="folder video (dicari sampai subfolder)")
    ap.add_argument("--out", required=True, help="folder tujuan npz")
    ap.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"],
                    help="ukuran model rtmlib (default balanced = sama dengan Stage 2)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="auto (cuda jika ada, selain itu cpu), cpu, mps (Mac, lewat CoreML), cuda (GPU NVIDIA)")
    ap.add_argument("--stride", type=int, default=2, help="proses setiap frame ke-N (default 2, sama dengan label)")
    ap.add_argument("--det-thr", type=float, default=0.10, help="ambang skor YOLOX (default 0.10, sama dengan Stage 2)")
    ap.add_argument("--nms-thr", type=float, default=0.45)
    ap.add_argument("--max-dets", type=int, default=30, help="maks. kotak per frame (default 30)")
    ap.add_argument("--tanpa-pose", action="store_true", help="lewati RTMPose (lebih cepat, kerangka K tidak tersedia)")
    ap.add_argument("--pose-min-score", type=float, default=0.3,
                    help="RTMPose hanya untuk kotak dengan skor >= ini (default 0.3 = kotak yang tampil di alat label; "
                         "0.1 = semua kotak seperti Stage 2, jauh lebih lama)")
    ap.add_argument("--ulang", action="store_true", help="buat ulang walaupun npz sudah ada")
    ap.add_argument("--only", help="hanya video yang namanya mengandung teks ini")
    args = ap.parse_args()

    videos = find_videos(args.videos, args.only)
    if not videos:
        sys.exit(f"Tidak ada video di {args.videos}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for t in out.glob("*.npz.tmp"):            # sisa proses yang dihentikan di tengah jalan
        try:
            t.unlink()
        except OSError:
            pass
    if args.stride < 1:
        sys.exit("--stride minimal 1")

    def usable(v):
        """npz lama dipakai jika jarak framenya membagi --stride (npz setiap 2 frame cocok untuk 2, 4, 6, ...)."""
        p = out / f"{v}.npz"
        if args.ulang or not p.exists():
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
    say(f"{len(videos)} video, {len(videos) - len(todo)} sudah punya npz yang cocok, {len(todo)} akan diproses "
        f"(setiap {args.stride} frame).")
    if redo and not args.ulang:
        say(f"{len(redo)} npz lama dibuat dengan jarak frame yang tidak cocok dan akan dibuat ulang: {', '.join(redo)}")
    if not todo:
        say("SELESAI: semua video sudah punya npz.")
        return
    avail = available_devices()
    if args.device == "auto":
        args.device = "cuda" if "cuda" in avail else "cpu"
    elif args.device not in avail:
        say(f"[PERINGATAN] perangkat {args.device} tidak tersedia di komputer ini (tersedia: {', '.join(avail)}). Memakai cpu.")
        args.device = "cpu"
    say(f"Memuat model YOLOX{' + RTMPose' if not args.tanpa_pose else ''} ({args.mode}, {args.device}). "
        "Pertama kali: model diunduh dulu (perlu internet).")

    def is_download_error(e):
        txt = f"{e.__class__.__name__}: {e}"
        return ("CERTIFICATE_VERIFY_FAILED" in txt or "urlopen error" in txt or "URLError" in txt), txt

    def load(device):
        """Muat model. Jika unduhan gagal karena sertifikat, coba sekali lagi dengan sertifikat certifi."""
        try:
            return build_models(args.mode, device, args.det_thr, args.nms_thr, not args.tanpa_pose)
        except SystemExit:
            raise
        except Exception as e:
            dl, txt = is_download_error(e)
            if dl and "CERTIFICATE_VERIFY_FAILED" in txt and use_certifi():
                say("Sertifikat SSL bawaan Python gagal; mencoba lagi dengan sertifikat certifi.")
                return load(device)
            if dl:
                say("[GAGAL] Model tidak bisa diunduh: " + txt)
                say("Periksa internet (perlu akses ke download.openmmlab.com atau huggingface.co).\n"
                    "Di macOS dengan Python dari python.org, jalankan sekali:\n"
                    '  open "/Applications/Python 3.12/Install Certificates.command"\n'
                    "(sesuaikan versi Python), lalu coba lagi.")
                sys.exit(1)
            raise

    try:
        det, pose, cfg = load(args.device)
    except SystemExit:
        raise
    except Exception as e:
        if args.device == "cpu":
            raise
        say(f"[PERINGATAN] perangkat {args.device} gagal ({e.__class__.__name__}: {e}). Memakai cpu.")
        args.device = "cpu"
        det, pose, cfg = load("cpu")
    say(f"Perangkat yang dipakai: YOLOX {active_providers(det)}"
        + (f", RTMPose {active_providers(pose)}" if pose is not None else ""))

    t_all, gagal = time.time(), []
    try:
        for k, (vid, path) in enumerate(todo, 1):
            say(f"[VIDEO] {k}/{len(todo)} {vid}")
            try:
                n, secs = process_video(path, out, det, pose, cfg, args, k, len(todo))
                say(f"[OK] {vid}: {n} langkah dalam {fmt_sec(secs)} -> {out / (vid + '.npz')}")
            except Exception as e:
                gagal.append(vid)
                say(f"[GAGAL] {vid}: {e.__class__.__name__}: {e}")
    except KeyboardInterrupt:
        say("Dihentikan. Video yang sudah selesai tetap tersimpan; jalankan lagi untuk melanjutkan.")
        sys.exit(130)
    say(f"SELESAI dalam {fmt_sec(time.time() - t_all)}. Berhasil {len(todo) - len(gagal)}, gagal {len(gagal)}"
        + (f": {', '.join(gagal)}" if gagal else "."))
    if gagal:
        sys.exit(1)


if __name__ == "__main__":
    main()
