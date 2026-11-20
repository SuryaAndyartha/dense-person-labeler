#!/usr/bin/env python3
"""
dense_labeler.py - pelabelan pasien PADAT (setiap frame kedua, ~15 fps) untuk video FoG.

Cara kerja singkat:
  1. Klik kotak YOLOX pasien sekali, ATAU tarik untuk menggambar kotak sendiri.
  2. Tekan Spasi: alat mengisi langkah berikutnya sementara kamu menonton.
     - Label terakhir dari KLIK kotak YOLOX  -> langkah berikutnya ikut kotak YOLOX (biru).
     - Label terakhir dari GAMBAR sendiri   -> langkah berikutnya menyalin kotak gambaranmu (ungu).
     Alat BERHENTI sendiri setiap kali ragu.
  3. Kalau kotak salah: Spasi (jeda), mundur dengan panah kiri, klik / gambar kotak yang benar, Spasi lagi.
     Label otomatis sesudahnya dihitung ulang dari koreksimu. Setiap video mulai lagi dengan ikut YOLOX.

Berasal dari alat label padat proyek FreezingOfGait-SelfAnalysis (tools/each_frame_label/1_2_label_padat.py);
logika pelabelan sama, tambahan hanya roda mouse di Linux dan tampilan tajam di Windows.
Biasanya dibuka lewat start.py. Bisa juga langsung, dari folder repo:
  python3 labeler/dense_labeler.py --videos datasets/<nama>/videos --npz datasets/<nama>/npz \
      --out datasets/<nama>/labels [--labels datasets/<nama>/initial_labels.csv]
Tanpa --npz = 100% manual (gambar kotak sendiri). Lihat README.md.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:
    sys.exit("Tkinter tidak tersedia. macOS + Homebrew: brew install python-tk@3.12 (sesuaikan versi Python).")
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError:
    sys.exit("Paket kurang. Jalankan: pip install opencv-python numpy pillow")

# ----------------------------------------------------------------------------- pengaturan
FOLLOW_IOU = 0.5      # kotak berikutnya diikuti otomatis jika IoU dengan kotak sebelumnya >= ini
ANCHOR_IOU = 0.5      # label otomatis harus cocok (IoU >= ini) dengan label awal 8-cuplikan
PROP_MIN_SCORE = 0.1  # kotak deteksi yang boleh diikuti otomatis
SPEEDS = [0.5, 1, 2, 4]
JUMP_STEPS = 15       # Shift + panah = lompat ~1 detik
CACHE_FRAMES = 120
AUTOSAVE_SEC = 20
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
FIELDS = ["video_id", "frame_idx", "t_sec", "status", "x1", "y1", "x2", "y2", "sumber"]
# sumber label:
#   manual      = kamu KLIK kotak YOLOX (atau tombol Y)    -> berikutnya ikut YOLOX
#   gambar      = kamu MENGGAMBAR kotak (atau tombol F)    -> berikutnya menyalin kotak ini
#   otomatis    = diisi alat dari kotak YOLOX
#   ikut_gambar = diisi alat dengan menyalin kotak gambaranmu
#   label_awal  = dari patient_labels.csv (8 cuplikan); tidak mengubah cara mengikuti
WARNA = {"manual": "#22a447", "gambar": "#7b3fbf", "label_awal": "#e6a700", "otomatis": "#2f7fd8",
         "ikut_gambar": "#b58ae0", "tidak_ada": "#8a8a8a"}
WARNA_RGB = {"belum": (232, 232, 228), "manual": (34, 164, 71), "gambar": (123, 63, 191), "label_awal": (230, 167, 0),
             "otomatis": (47, 127, 216), "ikut_gambar": (181, 138, 224), "tidak_ada": (138, 138, 138)}
NAMA_SUMBER = {"manual": "klik YOLOX", "gambar": "gambar sendiri", "label_awal": "label awal",
               "otomatis": "ikut YOLOX", "ikut_gambar": "ikut gambaranmu"}
OTOMATIS = ("otomatis", "ikut_gambar")      # boleh dihitung ulang saat diputar
MODE_GAMBAR = ("gambar", "ikut_gambar")     # label yang membuat langkah berikutnya menyalin kotak
# kerangka Halpe-26 (kiri = cyan, kanan = oranye, tengah = putih)
KP_EDGES = [((11, 13), "L"), ((13, 15), "L"), ((15, 24), "L"), ((15, 20), "L"), ((20, 22), "L"),
            ((12, 14), "R"), ((14, 16), "R"), ((16, 25), "R"), ((16, 21), "R"), ((21, 23), "R"),
            ((11, 12), "C"), ((5, 11), "L"), ((6, 12), "R"), ((5, 6), "C"), ((5, 7), "L"), ((7, 9), "L"),
            ((6, 8), "R"), ((8, 10), "R")]
KP_COL = {"L": "#3fd0e0", "R": "#ff9a3c", "C": "#f2f2f2"}
KP_SHOW = 0.3          # keypoint ditampilkan jika skornya >= ini


def iou_many(b, B):
    if len(B) == 0:
        return np.zeros(0)
    B = np.asarray(B, float)
    iw = np.clip(np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]), 0, None)
    ih = np.clip(np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]), 0, None)
    inter = iw * ih
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-9)


# ----------------------------------------------------------------------------- data satu video
class VideoData:
    def __init__(self, vid, path, npz_path, step, anchors, out_dir):
        self.vid, self.path, self.out_csv = vid, path, out_dir / f"{vid}.csv"
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise IOError(f"Video tidak bisa dibuka: {path}")
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.has_det = False
        self.det_slices, self.det_boxes, self.det_scores = {}, np.zeros((0, 4)), np.zeros(0)
        self.det_kp, self.det_ks = None, None
        self.npz_note = ""
        frames = None
        if npz_path is not None and npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as d:
                fp = d["frames_processed"].astype(int)
                npz_step = int(np.median(np.diff(fp))) if len(fp) > 1 else 1
                if step % npz_step == 0:
                    # npz dipakai: langkah label = frame npz yang merupakan kelipatan step
                    o = np.argsort(d["det_frame"], kind="stable")
                    fr, self.det_boxes, self.det_scores = d["det_frame"][o], d["det_boxes"][o].astype(float), d["det_scores"][o].astype(float)
                    frames = fp[fp % step == 0]
                    if "det_kpts" in d.files and "det_kpt_scores" in d.files:
                        self.det_kp, self.det_ks = d["det_kpts"][o].astype(float), d["det_kpt_scores"][o].astype(float)
                    if "meta" in d.files:
                        try:
                            self.fps = float(json.loads(str(d["meta"])).get("fps", self.fps))
                        except Exception:
                            pass
                    u, s = np.unique(fr, return_index=True)
                    e = np.r_[s[1:], len(fr)]
                    self.det_slices = {int(f): (int(a), int(b)) for f, a, b in zip(u, s, e)}
                    self.has_det = True
                else:
                    self.npz_note = (f"npz video ini dibuat setiap {npz_step} frame, tidak cocok dengan label setiap {step} frame: "
                                     "video ini dilabel manual")
        self.frames = list(frames) if frames is not None else list(range(0, max(n, 1), step))
        self.index_of = {f: i for i, f in enumerate(self.frames)}
        self.labels = {}      # indeks langkah -> dict(status, box, sumber)
        self.extra = {}       # frame -> baris CSV di luar langkah sekarang (disimpan apa adanya)
        self.anchors = {}     # indeks langkah -> kotak label awal
        for f, box in anchors:
            i = self.nearest(f)
            self.anchors[i] = box
        self._load_csv()
        for i, box in self.anchors.items():
            if i not in self.labels:
                self.labels[i] = dict(status="terlihat", box=box, sumber="label_awal")
        self.cache, self.pos = OrderedDict(), 0   # pos = frame yang akan dibaca cap.read() berikutnya
        self.dirty = False

    def nearest(self, f):
        a = np.asarray(self.frames)
        return int(np.argmin(np.abs(a - f)))

    def _load_csv(self):
        if not self.out_csv.exists():
            return
        with open(self.out_csv, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                f = int(r["frame_idx"])
                if f not in self.index_of:
                    # label dari jarak frame lain (mis. dulu setiap 2 frame, sekarang setiap 4): tidak tampil,
                    # tetapi tetap ditulis kembali saat menyimpan supaya tidak hilang
                    self.extra[f] = [r.get(k, "") for k in FIELDS]
                    continue
                box = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")]) if r["status"] == "terlihat" else None
                sb = r.get("sumber") or "manual"
                self.labels[self.index_of[f]] = dict(status=r["status"], box=box, sumber=sb if sb in WARNA else "manual")

    def save(self):
        tmp = self.out_csv.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(FIELDS)
            rows = dict(self.extra)
            for i in self.labels:
                L, f = self.labels[i], self.frames[i]
                b = L["box"] if L["box"] is not None else [""] * 4
                rows[f] = ([self.vid, f, f"{f / self.fps:.3f}", L["status"]] +
                           [f"{v:.1f}" if v != "" else "" for v in b] + [L["sumber"]])
            for f in sorted(rows):
                w.writerow(rows[f])
        os.replace(tmp, self.out_csv)
        self.dirty = False

    def frame(self, f):
        if f in self.cache:
            self.cache.move_to_end(f)
            return self.cache[f]
        if 0 <= f - self.pos <= 8:
            for _ in range(f - self.pos):
                self.cap.grab()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = self.cap.read()
        self.pos = f + 1
        if not ok:
            img = np.zeros((max(self.H, 1), max(self.W, 1), 3), np.uint8)
        self.cache[f] = img
        while len(self.cache) > CACHE_FRAMES:
            self.cache.popitem(last=False)
        return img

    def det_index(self, i, min_score):
        s = self.det_slices.get(self.frames[i])
        if s is None:
            return np.zeros(0, int)
        idx = np.arange(s[0], s[1])
        return idx[self.det_scores[idx] >= min_score]

    def dets(self, i, min_score):
        idx = self.det_index(i, min_score)
        return self.det_boxes[idx], self.det_scores[idx]

    def close(self):
        self.cap.release()


# ----------------------------------------------------------------------------- aplikasi
class App:
    def __init__(self, root, videos, args):
        self.root, self.videos, self.args = root, videos, args
        self.out = Path(args.out); self.out.mkdir(parents=True, exist_ok=True)
        self.anchors = self._read_anchors(args.labels)
        self.data, self.i = None, 0
        self.playing, self.speed_i, self.show_det, self.show_kp = False, 2, True, False
        self.undo = []
        self.click_cycle = (None, [], 0)
        self.drag0, self.rubber = None, None
        self.view = (1.0, 0, 0)
        self.tl_img = None
        self._build()
        self._bind()
        self.open_video(self._first_video())
        self.root.after(AUTOSAVE_SEC * 1000, self._autosave)

    # ---------------- input
    def _read_anchors(self, path):
        out = {}
        if not path:
            return out
        with open(path, newline="", encoding=self.args.labels_encoding) as fh:
            for r in csv.DictReader(fh):
                if r.get("status", "terlihat") != "terlihat" or not r.get("x1"):
                    continue
                vid = Path(str(r["video_id"])).stem
                try:
                    box = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
                except ValueError:
                    continue
                out.setdefault(vid, []).append((int(float(r["frame_idx"])), box))
        return out

    def _npz(self, vid):
        return Path(self.args.npz) / f"{vid}.npz" if self.args.npz else None

    def _progress(self, vid):
        p = self.out / f"{vid}.csv"
        n_lab = max(sum(1 for _ in open(p, encoding="utf-8")) - 1, 0) if p.exists() else 0
        tot = None
        z = self._npz(vid)
        if z is not None and z.exists():
            try:
                with np.load(z, allow_pickle=False) as d:
                    fp = d["frames_processed"].astype(int)
                npz_step = int(np.median(np.diff(fp))) if len(fp) > 1 else 1
                tot = int((fp % self.args.step == 0).sum()) if self.args.step % npz_step == 0 else None
            except Exception:
                tot = None
        return n_lab, tot

    def _first_video(self):
        if self.args.start and self.args.start in self.videos:
            return self.args.start
        for vid in self.videos:
            n, tot = self._progress(vid)
            if tot is None or n < tot:
                return vid
        return next(iter(self.videos))

    def _video_values(self):
        vals = []
        for vid in self.videos:
            n, tot = self._progress(vid)
            vals.append(f"{vid}   ({100 * n / tot:.0f}%)" if tot else f"{vid}   ({n} label)")
        return vals

    # ---------------- tampilan
    def _build(self):
        r = self.root
        r.title("Label padat pasien FoG")
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        r.geometry(f"{int(sw * .9)}x{int(sh * .85)}+{int(sw * .05)}+{int(sh * .04)}")
        r.minsize(980, 620)
        r.columnconfigure(0, weight=1); r.rowconfigure(0, weight=1)
        style = ttk.Style()
        style.configure("Small.TLabel", font=("TkDefaultFont", 10))
        style.configure("Head.TLabel", font=("TkDefaultFont", 12, "bold"))
        style.configure("Big.TLabel", font=("TkDefaultFont", 13, "bold"))

        left = ttk.Frame(r); left.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        left.columnconfigure(0, weight=1); left.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(left, bg="#1d1d1b", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.timeline = tk.Canvas(left, height=26, bg="#f2f2ee", highlightthickness=0, cursor="hand2")
        self.timeline.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.msg = ttk.Label(left, text="", anchor="w"); self.msg.grid(row=2, column=0, sticky="ew", pady=(4, 0))

        # panel kanan dalam kanvas yang bisa digulir: tidak pernah terpotong di layar kecil
        holder = ttk.Frame(r); holder.grid(row=0, column=1, sticky="ns", padx=(4, 8), pady=8)
        holder.rowconfigure(0, weight=1)
        pc = tk.Canvas(holder, width=300, highlightthickness=0, bg=ttk.Style().lookup("TFrame", "background") or None)
        sb = ttk.Scrollbar(holder, orient="vertical", command=pc.yview)
        pc.configure(yscrollcommand=sb.set)
        pc.grid(row=0, column=0, sticky="ns"); sb.grid(row=0, column=1, sticky="ns")
        p = ttk.Frame(pc)
        pc.create_window(0, 0, anchor="nw", window=p, width=300)
        p.bind("<Configure>", lambda e: pc.configure(scrollregion=pc.bbox("all")))

        def _wheel(e):                       # gulir hanya jika kursor di atas panel kanan
            if str(e.widget).startswith(str(holder)):
                pc.yview_scroll(int(-e.delta / 120) if abs(e.delta) >= 120 else -int(e.delta), "units")
        pc.bind_all("<MouseWheel>", _wheel)

        def _wheel_linux(e, d):              # Linux/X11 mengirim roda mouse sebagai tombol 4/5
            if str(e.widget).startswith(str(holder)):
                pc.yview_scroll(d, "units")
        pc.bind_all("<Button-4>", lambda e: _wheel_linux(e, -1))
        pc.bind_all("<Button-5>", lambda e: _wheel_linux(e, 1))
        p.columnconfigure(0, weight=1)
        ttk.Label(p, text="Video", style="Head.TLabel").grid(row=0, column=0, sticky="w")
        vrow = ttk.Frame(p); vrow.grid(row=1, column=0, sticky="ew", pady=(2, 8)); vrow.columnconfigure(1, weight=1)
        ttk.Button(vrow, text="◀", width=3, command=lambda: self.switch_video(-1)).grid(row=0, column=0)
        self.combo = ttk.Combobox(vrow, state="readonly", values=self._video_values(), height=25)
        self.combo.grid(row=0, column=1, sticky="ew", padx=4)
        self.combo.bind("<<ComboboxSelected>>", self._combo_pick)
        ttk.Button(vrow, text="▶", width=3, command=lambda: self.switch_video(1)).grid(row=0, column=2)

        self.info_frame = ttk.Label(p, text="", style="Big.TLabel"); self.info_frame.grid(row=2, column=0, sticky="w")
        self.info_time = ttk.Label(p, text=""); self.info_time.grid(row=3, column=0, sticky="w")
        self.info_state = tk.Label(p, text="", anchor="w", font=("TkDefaultFont", 12, "bold"), fg="white", padx=8, pady=4)
        self.info_state.grid(row=4, column=0, sticky="ew", pady=(6, 2))
        self.info_next = ttk.Label(p, text="", wraplength=290); self.info_next.grid(row=5, column=0, sticky="w", pady=(0, 4))
        self.info_prog = ttk.Label(p, text=""); self.info_prog.grid(row=6, column=0, sticky="w")
        self.pbar = ttk.Progressbar(p, maximum=100); self.pbar.grid(row=7, column=0, sticky="ew", pady=(2, 10))

        sp = ttk.Frame(p); sp.grid(row=8, column=0, sticky="ew")
        ttk.Label(sp, text="Kecepatan putar:").pack(side="left")
        ttk.Button(sp, text="−", width=2, command=lambda: self.set_speed(-1)).pack(side="left", padx=(6, 0))
        self.lbl_speed = ttk.Label(sp, text="", width=5, anchor="center"); self.lbl_speed.pack(side="left")
        ttk.Button(sp, text="+", width=2, command=lambda: self.set_speed(1)).pack(side="left")
        cf = ttk.Frame(p); cf.grid(row=9, column=0, sticky="w", pady=(8, 8))
        self.var_det = tk.BooleanVar(value=True); self.var_kp = tk.BooleanVar(value=False)
        ttk.Checkbutton(cf, text="Kotak deteksi (H)", variable=self.var_det,
                        command=self._toggle_det_from_box).pack(side="left")
        ttk.Checkbutton(cf, text="Kerangka (K)", variable=self.var_kp,
                        command=self._toggle_kp_from_box).pack(side="left", padx=(10, 0))

        ttk.Label(p, text="Warna", style="Head.TLabel").grid(row=10, column=0, sticky="w")
        lg = ttk.Frame(p); lg.grid(row=11, column=0, sticky="w", pady=(2, 10))
        legend = (("klik YOLOX", "manual"), ("ikut YOLOX", "otomatis"), ("gambar sendiri", "gambar"),
                  ("ikut gambaranmu", "ikut_gambar"), ("label awal", "label_awal"), ("tidak terlihat", "tidak_ada"))
        for k, (nm, key) in enumerate(legend):
            tk.Label(lg, bg=WARNA[key], width=2).grid(row=k // 2, column=2 * (k % 2), sticky="w", pady=1)
            ttk.Label(lg, text=nm).grid(row=k // 2, column=2 * (k % 2) + 1, sticky="w", padx=(4, 10))

        ttk.Label(p, text="Tombol", style="Head.TLabel").grid(row=12, column=0, sticky="w")
        keys = [("Klik kotak", "pilih kotak YOLOX (klik lagi = kotak lain)"), ("Tarik", "gambar kotak sendiri"),
                ("Spasi", "putar / jeda (isi otomatis)"), ("→  /  D", "langkah berikut"), ("←  /  A", "langkah sebelumnya"),
                ("Shift + ← →", "lompat ~1 detik"), ("F", "kunci kotak ini: berikutnya menyalin kotak ini"),
                ("Y", "kembali ikut YOLOX dari langkah ini"), ("N", "pasien tidak terlihat"),
                ("Backspace", "hapus label langkah ini"), ("G", "ke langkah belum dilabel"), ("[  ]", "pelan / cepat"),
                ("H / K", "kotak deteksi / kerangka"), ("Ctrl/Cmd + Z", "batal"), ("S", "simpan"),
                ("PgUp / PgDn", "video sebelum / berikut")]
        kf = ttk.Frame(p); kf.grid(row=13, column=0, sticky="ew", pady=(2, 10))
        for k, (a, b) in enumerate(keys):
            ttk.Label(kf, text=a, style="Small.TLabel", font=("TkDefaultFont", 10, "bold")).grid(row=k, column=0, sticky="w")
            ttk.Label(kf, text=b, style="Small.TLabel", wraplength=180).grid(row=k, column=1, sticky="w", padx=(8, 0))
        bf = ttk.Frame(p); bf.grid(row=14, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(bf, text="Simpan (S)", command=self.save).pack(side="left", expand=True, fill="x")
        ttk.Button(bf, text="Belum dilabel (G)", command=self.goto_unlabeled).pack(side="left", expand=True, fill="x", padx=(4, 0))

        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.timeline.bind("<Configure>", lambda e: self.draw_timeline(full=True))
        self.timeline.bind("<ButtonPress-1>", self._tl_click)
        self.timeline.bind("<B1-Motion>", self._tl_click)
        r.protocol("WM_DELETE_WINDOW", self.quit)

    def _bind(self):
        def b(seq, fn):
            try:
                self.root.bind(seq, fn)
            except tk.TclError:      # mis. <Command-...> di luar macOS
                pass

        def no_focus(w):             # tombol tidak mencuri Spasi / panah dari kanvas
            for ch in w.winfo_children():
                if isinstance(ch, (ttk.Button, ttk.Checkbutton)):
                    ch.configure(takefocus=0)
                no_focus(ch)
        no_focus(self.root)
        for k in ("<Right>", "<d>", "<D>"):
            b(k, lambda e: self.step(1, follow=True))
        for k in ("<Left>", "<a>", "<A>"):
            b(k, lambda e: self.step(-1))
        b("<Shift-Right>", lambda e: self.step(JUMP_STEPS)); b("<Shift-Left>", lambda e: self.step(-JUMP_STEPS))
        b("<space>", lambda e: self.toggle_play())
        for k in ("<n>", "<N>"):
            b(k, lambda e: self.set_absent())
        for k in ("<BackSpace>", "<Delete>"):
            b(k, lambda e: self.clear_label())
        for k in ("<g>", "<G>"):
            b(k, lambda e: self.goto_unlabeled())
        for k in ("<h>", "<H>"):
            b(k, lambda e: self.toggle_det())
        for k in ("<k>", "<K>"):
            b(k, lambda e: self.toggle_kp())
        for k in ("<f>", "<F>"):
            b(k, lambda e: self.lock_box())
        for k in ("<y>", "<Y>"):
            b(k, lambda e: self.back_to_yolox())
        for k in ("<s>", "<S>", "<Control-s>", "<Command-s>"):
            b(k, lambda e: self.save())
        for k in ("<Control-z>", "<Command-z>"):
            b(k, lambda e: self.do_undo())
        b("<bracketleft>", lambda e: self.set_speed(-1)); b("<bracketright>", lambda e: self.set_speed(1))
        b("<Prior>", lambda e: self.switch_video(-1)); b("<Next>", lambda e: self.switch_video(1))

    # ---------------- video
    def open_video(self, vid):
        self.stop()
        if self.data is not None:
            if self.data.dirty:
                self.data.save()
            self.data.close()
        try:
            self.data = VideoData(vid, self.videos[vid], self._npz(vid), self.args.step, self.anchors.get(vid, []), self.out)
        except Exception as e:
            messagebox.showerror("Gagal membuka video", str(e)); return
        self.undo.clear()
        keys = list(self.videos)
        self.combo.current(keys.index(vid))
        self.i = self._first_unlabeled(0)
        self.draw_timeline(full=True)
        if self.data.npz_note:
            note = "  |  " + self.data.npz_note
        elif not self.data.has_det:
            note = "  |  file npz tidak ada: tidak ada kotak deteksi, ikuti otomatis hanya menyalin kotak"
        else:
            note = ""
        self.say(f"Membuka {vid}: {len(self.data.frames)} langkah (setiap {self.args.step} frame), "
                 f"{self.data.fps:.0f} fps video.{note}", warn=bool(self.data.npz_note))
        self.render()
        self.canvas.focus_set()

    def switch_video(self, d):
        keys = list(self.videos)
        k = keys.index(self.data.vid) + d if self.data else 0
        if 0 <= k < len(keys):
            self.open_video(keys[k])
            self.combo.configure(values=self._video_values()); self.combo.current(k)

    def _combo_pick(self, _):
        vid = list(self.videos)[self.combo.current()]
        if self.data is None or vid != self.data.vid:
            self.open_video(vid)
        self.canvas.focus_set()

    # ---------------- label
    def _set(self, i, lab, record=True):
        if record:
            self.undo.append((i, self.data.labels.get(i)))
            del self.undo[:-500]
        if lab is None:
            self.data.labels.pop(i, None)
        else:
            self.data.labels[i] = lab
        self.data.dirty = True

    def follow_source(self, i):
        """Cara mengisi langkah i+1 berdasarkan label langkah i: ("yolox" | "gambar" | None, kotak acuan).
        Label awal tidak mengubah cara mengikuti: alat melihat label sebelum label awal itu."""
        d = self.data
        L = d.labels.get(i)
        if L is None or L["status"] != "terlihat":
            return None, None
        k = i
        while k >= 0:
            Lk = d.labels.get(k)
            if Lk is None or Lk["status"] != "terlihat":
                break
            if Lk["sumber"] != "label_awal":
                if Lk["sumber"] in MODE_GAMBAR:
                    return "gambar", Lk["box"]
                break
            k -= 1
        return ("yolox" if d.has_det else "gambar"), L["box"]

    def follow(self, i):
        """Isi langkah i dari langkah i-1. Label klik / gambar / label awal tidak pernah ditimpa.
        Mengembalikan (berhasil, pesan)."""
        d = self.data
        cur = d.labels.get(i)
        mode, ref = self.follow_source(i - 1)

        def calon():
            if mode == "gambar":
                return ref
            B, _ = d.dets(i, PROP_MIN_SCORE)
            io = iou_many(ref, B)
            return B[int(io.argmax())] if len(io) and io.max() >= FOLLOW_IOU else None

        if cur is not None and cur["sumber"] not in OTOMATIS:
            if cur["sumber"] == "label_awal" and mode is not None:
                c = calon()
                if c is not None and iou_many(c, [cur["box"]])[0] < ANCHOR_IOU:
                    return False, "Kotak yang diikuti TIDAK cocok dengan label awal (kuning). Periksa mundur dengan ←."
            return True, ""
        if mode is None:
            if cur is not None:
                self._set(i, None, record=False)
            return False, "Langkah sebelumnya belum ada kotak pasien. Klik atau gambar kotak pasien untuk mulai."
        c = calon()
        if c is None:
            if cur is not None:
                self._set(i, None, record=False)
            return False, ("Kotak YOLOX pasien tidak jelas di sini. Klik kotak pasien, tarik kotak sendiri "
                           "(berikutnya akan menyalin kotakmu), atau tekan N.")
        if i in d.anchors and iou_many(c, [d.anchors[i]])[0] < ANCHOR_IOU:
            if cur is not None:
                self._set(i, None, record=False)
            return False, "Kotak yang diikuti TIDAK cocok dengan label awal (kuning). Periksa mundur dengan ←."
        sumber = "ikut_gambar" if mode == "gambar" else "otomatis"
        self._set(i, dict(status="terlihat", box=np.asarray(c, float).copy(), sumber=sumber), record=False)
        return True, ""

    def lock_box(self):
        """F: pakai kotak langkah ini sebagai acuan; langkah berikutnya menyalin kotak ini."""
        self.stop()
        L = self.data.labels.get(self.i)
        if L is None or L["status"] != "terlihat":
            self.say("Belum ada kotak pasien di langkah ini untuk dikunci.", warn=True); return
        self._set(self.i, dict(status="terlihat", box=L["box"].copy(), sumber="gambar"))
        self.say("Kotak dikunci: langkah berikutnya menyalin kotak ini. Klik kotak YOLOX atau tekan Y untuk kembali ikut YOLOX.")
        self.render(); self.draw_timeline(full=True)

    def back_to_yolox(self):
        """Y: ganti kotak langkah ini dengan kotak YOLOX yang paling mirip; langkah berikutnya ikut YOLOX lagi."""
        self.stop()
        d = self.data
        L = d.labels.get(self.i)
        if not d.has_det:
            self.say("Tidak ada file npz: tidak ada kotak YOLOX.", warn=True); return
        if L is None or L["status"] != "terlihat":
            self.say("Belum ada kotak pasien di langkah ini. Klik kotak YOLOX pasien.", warn=True); return
        B, _ = d.dets(self.i, PROP_MIN_SCORE)
        io = iou_many(L["box"], B)
        if not len(io) or io.max() < 0.3:
            self.say("Tidak ada kotak YOLOX yang mirip di langkah ini. Klik kotak lain atau pindah langkah.", warn=True); return
        self._set(self.i, dict(status="terlihat", box=B[int(io.argmax())].copy(), sumber="manual"))
        self.say("Kembali ikut YOLOX mulai langkah ini.")
        self.render(); self.draw_timeline(full=True)

    def step(self, d, follow=False):
        n = len(self.data.frames)
        j = min(max(self.i + d, 0), n - 1)
        if j == self.i:
            if d > 0:
                self.say("Sudah langkah terakhir video ini. PgDn = video berikutnya.")
            self.stop(); return False
        ok, msg = (self.follow(j) if follow and d == 1 else (True, ""))
        self.i = j
        self.render()
        self.draw_timeline(full=follow)
        if not ok:
            self.stop(); self.say(msg, warn=True)
        elif not self.playing:
            self.say("")
        return ok

    def set_absent(self):
        self.stop()
        self._set(self.i, dict(status="tidak_ada", box=None, sumber="manual"))
        self.say("Ditandai: pasien tidak terlihat di langkah ini.")
        self.render(); self.draw_timeline(full=True)

    def clear_label(self):
        self.stop()
        if self.i in self.data.labels:
            self._set(self.i, None)
            self.render(); self.draw_timeline(full=True)

    def do_undo(self):
        self.stop()
        if not self.undo:
            self.say("Tidak ada yang bisa dibatalkan."); return
        i, old = self.undo.pop()
        self._set(i, old, record=False)
        self.i = i
        self.say("Dibatalkan."); self.render(); self.draw_timeline(full=True)

    def _first_unlabeled(self, start):
        n = len(self.data.frames)
        for k in list(range(start, n)) + list(range(0, start)):
            if k not in self.data.labels:
                return max(k - 1, 0) if (k - 1) in self.data.labels else k
        return 0

    def goto_unlabeled(self):
        self.stop()
        self.i = self._first_unlabeled(self.i)
        if len(self.data.labels) == len(self.data.frames):
            self.say("Semua langkah video ini sudah berlabel.")
        self.render(); self.draw_timeline(full=False)

    # ---------------- putar
    def toggle_play(self):
        if self.playing:
            self.stop(); self.say("Jeda.")
        else:
            cur = self.data.labels.get(self.i)
            if cur is None or cur["status"] != "terlihat":
                self.say("Klik kotak pasien dulu di langkah ini, lalu tekan Spasi.", warn=True); return
            self.playing = True
            self.say(f"Memutar {SPEEDS[self.speed_i]}x dan mengikuti pasien. Spasi = jeda.")
            self._tick()

    def stop(self):
        self.playing = False

    def _tick(self):
        if not self.playing:
            return
        t0 = time.perf_counter()
        if not self.step(1, follow=True):
            self.draw_timeline(full=True); return
        delay = 1000.0 / (self.data.fps / max(self.args.step, 1) * SPEEDS[self.speed_i])
        el = (time.perf_counter() - t0) * 1000
        self.root.after(max(1, int(delay - el)), self._tick)

    def set_speed(self, d):
        self.speed_i = min(max(self.speed_i + d, 0), len(SPEEDS) - 1)
        self.lbl_speed.configure(text=f"{SPEEDS[self.speed_i]}x")

    def toggle_det(self):
        self.var_det.set(not self.var_det.get()); self._toggle_det_from_box()

    def _toggle_det_from_box(self):
        self.show_det = self.var_det.get(); self.render(); self.canvas.focus_set()

    def toggle_kp(self):
        self.var_kp.set(not self.var_kp.get()); self._toggle_kp_from_box()

    def _toggle_kp_from_box(self):
        self.show_kp = self.var_kp.get()
        if self.show_kp and self.data is not None and self.data.det_kp is None:
            self.say("File npz tidak berisi keypoint.", warn=True)
        self.render(); self.canvas.focus_set()

    # ---------------- mouse
    def _to_img(self, x, y):
        s, ox, oy = self.view
        return (x - ox) / s, (y - oy) / s

    def _press(self, e):
        self.stop()
        self.canvas.focus_set()
        self.drag0 = (e.x, e.y)

    def _drag(self, e):
        if self.drag0 is None:
            return
        if self.rubber is None:
            self.rubber = self.canvas.create_rectangle(*self.drag0, e.x, e.y, outline=WARNA["manual"], width=2, dash=(5, 3))
        self.canvas.coords(self.rubber, *self.drag0, e.x, e.y)

    def _release(self, e):
        if self.drag0 is None:
            return
        x0, y0 = self.drag0
        self.drag0 = None
        if self.rubber is not None:
            self.canvas.delete(self.rubber); self.rubber = None
        d = self.data
        if abs(e.x - x0) > 6 and abs(e.y - y0) > 6:
            (ax, ay), (bx, by) = self._to_img(x0, y0), self._to_img(e.x, e.y)
            box = np.array([max(min(ax, bx), 0), max(min(ay, by), 0), min(max(ax, bx), d.W), min(max(ay, by), d.H)])
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                return
            self._set(self.i, dict(status="terlihat", box=box, sumber="gambar"))
            self.say("Kotak digambar. Spasi = langkah berikutnya menyalin kotak ini (ungu). Klik kotak YOLOX atau Y untuk kembali.")
        else:
            px, py = self._to_img(e.x, e.y)
            B, S = d.dets(self.i, self.args.min_score)
            inside = [k for k in range(len(B)) if B[k][0] <= px <= B[k][2] and B[k][1] <= py <= B[k][3]]
            if not inside:
                self.say("Tidak ada kotak di titik itu. Tarik untuk menggambar kotak sendiri.", warn=True); return
            inside.sort(key=lambda k: (B[k][2] - B[k][0]) * (B[k][3] - B[k][1]))       # kotak terkecil dulu
            pos, lst, n = self.click_cycle
            if pos is not None and pos[0] == self.i and abs(pos[1] - e.x) < 6 and abs(pos[2] - e.y) < 6 and lst == inside:
                n = (n + 1) % len(inside)
            else:
                n = 0
            self.click_cycle = ((self.i, e.x, e.y), inside, n)
            k = inside[n]
            self._set(self.i, dict(status="terlihat", box=B[k].copy(), sumber="manual"))
            extra = f" ({n + 1}/{len(inside)} kotak di titik ini, klik lagi untuk ganti)" if len(inside) > 1 else ""
            self.say(f"Kotak YOLOX dipilih{extra}. Spasi = lanjut ikut YOLOX.")
        self.render(); self.draw_timeline(full=True)

    def _tl_click(self, e):
        self.stop()
        w = max(self.timeline.winfo_width(), 1)
        self.i = int(min(max(e.x / w, 0), 0.9999) * len(self.data.frames))
        self.render(); self.draw_timeline(full=False)
        self.canvas.focus_set()

    # ---------------- gambar
    def render(self):
        d = self.data
        if d is None:
            return
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        f = d.frames[self.i]
        img = d.frame(f)
        H, W = img.shape[:2]
        s = min(cw / W, ch / H)
        dw, dh = max(int(W * s), 1), max(int(H * s), 1)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.view = (s, ox, oy)
        small = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        c = self.canvas
        c.delete("all")
        c.create_image(ox, oy, anchor="nw", image=self.photo)

        def R(b):
            return ox + b[0] * s, oy + b[1] * s, ox + b[2] * s, oy + b[3] * s

        if self.show_det:
            B, S = d.dets(self.i, self.args.min_score)
            for b, sc in zip(B, S):
                x1, y1, x2, y2 = R(b)
                c.create_rectangle(x1, y1, x2, y2, outline="#d9d9d4", width=1)
                c.create_text(x1 + 3, y2 - 3, text=f"{sc:.2f}", anchor="sw", fill="#d9d9d4", font=("TkDefaultFont", 9))
        if self.show_kp and d.det_kp is not None:
            for j in d.det_index(self.i, self.args.min_score):
                kp, ks = d.det_kp[j], d.det_ks[j]
                for (a, b_), side in KP_EDGES:
                    if a < len(ks) and b_ < len(ks) and ks[a] >= KP_SHOW and ks[b_] >= KP_SHOW:
                        c.create_line(ox + kp[a, 0] * s, oy + kp[a, 1] * s, ox + kp[b_, 0] * s, oy + kp[b_, 1] * s,
                                      fill=KP_COL[side], width=2)
                for q in range(len(ks)):
                    if ks[q] >= KP_SHOW:
                        x, y = ox + kp[q, 0] * s, oy + kp[q, 1] * s
                        c.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5, fill="#ffffff", outline="")
        if self.i in d.anchors:
            x1, y1, x2, y2 = R(d.anchors[self.i])
            c.create_rectangle(x1, y1, x2, y2, outline=WARNA["label_awal"], width=2, dash=(6, 4))
        L = d.labels.get(self.i)
        if L is not None and L["status"] == "terlihat":
            col = WARNA[L["sumber"]]
            x1, y1, x2, y2 = R(L["box"])
            c.create_rectangle(x1, y1, x2, y2, outline=col, width=3)
            tag = f" PASIEN · {NAMA_SUMBER[L['sumber']]} "
            ty = y1 - 2 if y1 - 22 > oy else y1 + 22
            t = c.create_text(x1 + 2, ty, text=tag, anchor="sw", fill="white", font=("TkDefaultFont", 11, "bold"))
            bb = c.bbox(t)
            c.tag_lower(c.create_rectangle(bb, fill=col, outline=col), t)
        elif L is not None:
            c.create_text(ox + 12, oy + 12, text="Pasien tidak terlihat", anchor="nw", fill="white",
                          font=("TkDefaultFont", 14, "bold"))

        n = len(d.frames)
        self.info_frame.configure(text=f"Langkah {self.i + 1} / {n}")
        self.info_time.configure(text=f"Frame {f}   ·   {f / d.fps:.2f} s")
        if L is None:
            txt, bg = "BELUM DILABEL", "#b9b9b3"
        elif L["status"] == "tidak_ada":
            txt, bg = "PASIEN TIDAK TERLIHAT", WARNA["tidak_ada"]
        else:
            txt, bg = f"PASIEN · {NAMA_SUMBER[L['sumber']].upper()}", WARNA[L["sumber"]]
        self.info_state.configure(text=txt, bg=bg)
        mode, _ = self.follow_source(self.i)
        self.info_next.configure(text={"yolox": "Langkah berikutnya: ikut kotak YOLOX (biru)",
                                       "gambar": "Langkah berikutnya: salin kotak gambaranmu (ungu)",
                                       None: "Langkah berikutnya: menunggu kotak pasien"}[mode])
        k = len(d.labels)
        self.info_prog.configure(text=f"Berlabel: {k} / {n} langkah ({100 * k / n:.1f}%)")
        self.pbar.configure(value=100 * k / n)
        self.lbl_speed.configure(text=f"{SPEEDS[self.speed_i]}x")

    def draw_timeline(self, full=True):
        d = self.data
        if d is None:
            return
        tl = self.timeline
        w, h = max(tl.winfo_width(), 10), 26
        n = len(d.frames)
        if full or self.tl_img is None or self.tl_img.width() != w:
            code = np.zeros(n, np.uint8)
            order = ["belum", "manual", "gambar", "label_awal", "otomatis", "ikut_gambar", "tidak_ada"]
            for i, L in d.labels.items():
                code[i] = order.index("tidak_ada" if L["status"] == "tidak_ada" else L["sumber"])
            pal = np.array([WARNA_RGB[k] for k in order], np.uint8)
            cols = pal[code[np.minimum((np.arange(w) * n) // w, n - 1)]]
            img = np.repeat(cols[None], h - 8, axis=0)
            self.tl_img = ImageTk.PhotoImage(Image.fromarray(img))
        tl.delete("all")
        tl.create_image(0, 4, anchor="nw", image=self.tl_img)
        x = (self.i + 0.5) / n * w
        tl.create_rectangle(x - 1.5, 0, x + 1.5, h, fill="#d23b3b", outline="")

    def say(self, text, warn=False):
        self.msg.configure(text=text, foreground="#b3261e" if warn else "#3b3b38")

    # ---------------- simpan
    def save(self):
        if self.data is not None:
            self.data.save()
            self.say(f"Tersimpan: {self.data.out_csv}")
            k = list(self.videos).index(self.data.vid)
            self.combo.configure(values=self._video_values()); self.combo.current(k)

    def _autosave(self):
        if self.data is not None and self.data.dirty and not self.playing:
            self.data.save()
        self.root.after(AUTOSAVE_SEC * 1000, self._autosave)

    def quit(self):
        self.stop()
        if self.data is not None:
            if self.data.dirty:
                self.data.save()
            self.data.close()
        self.root.destroy()


# ----------------------------------------------------------------------------- utilitas baris perintah
def find_videos(folder, only):
    vids = {}
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix in VIDEO_EXT and not p.name.startswith("._") and (not only or only in p.stem):
            vids.setdefault(p.stem, p)
    return dict(sorted(vids.items()))


def merge(out_dir):
    out_dir = Path(out_dir)
    files = sorted(p for p in out_dir.glob("*.csv") if p.name != "all_labels.csv")
    tgt = out_dir / "all_labels.csv"
    n = 0
    with open(tgt, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo); w.writerow(FIELDS)
        for p in files:
            with open(p, newline="", encoding="utf-8") as fi:
                rd = csv.reader(fi); next(rd, None)
                for row in rd:
                    w.writerow(row); n += 1
    print(f"Digabung: {len(files)} video, {n} baris -> {tgt}")


def main():
    ap = argparse.ArgumentParser(description="Label pasien padat (setiap frame kedua) untuk video FoG.")
    ap.add_argument("--videos", help="folder berisi video (dicari sampai subfolder)")
    ap.add_argument("--npz", help="folder npz (dari labeler/make_npz.py atau pipeline lain dengan format sama). Tanpa ini = 100% manual")
    ap.add_argument("--out", default="labels", help="folder hasil (satu CSV per video). Default: ./labels")
    ap.add_argument("--labels", help="opsional: patient_labels.csv (8 cuplikan) sebagai label awal dan pemeriksa")
    ap.add_argument("--labels-encoding", default="cp1252", help="encoding patient_labels.csv (default cp1252)")
    ap.add_argument("--step", type=int, default=2,
                    help="label setiap N frame (default 2 = ~15 label/detik untuk video 30 fps). npz hanya dipakai jika "
                         "dibuat dengan jarak frame yang membagi N (npz setiap 2 frame cocok untuk N = 2, 4, 6, ...)")
    ap.add_argument("--min-score", type=float, default=0.3, help="kotak deteksi yang ditampilkan dan bisa diklik (default 0.3)")
    ap.add_argument("--only", help="hanya video yang namanya mengandung teks ini")
    ap.add_argument("--start", help="video yang dibuka pertama")
    ap.add_argument("--gabung", action="store_true", help="gabungkan semua CSV di --out menjadi all_labels.csv lalu keluar")
    args = ap.parse_args()
    if args.gabung:
        merge(args.out); return
    if args.step < 1:
        ap.error("--step minimal 1 (1 = setiap frame, sama dengan fps video)")
    if not args.videos:
        ap.error("--videos wajib diisi")
    videos = find_videos(args.videos, args.only)
    if not videos:
        sys.exit(f"Tidak ada video di {args.videos}")
    if args.npz and not Path(args.npz).is_dir():
        sys.exit(f"Folder npz tidak ada: {args.npz}")
    if os.name == "nt":                      # Windows: tampilan tajam di layar resolusi tinggi
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    App(root, videos, args)
    root.mainloop()


if __name__ == "__main__":
    main()
