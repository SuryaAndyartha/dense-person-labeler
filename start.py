#!/usr/bin/env python3
"""
start.py - pintu masuk dense-person-labeler: label padat (setiap frame ke-2) untuk dataset video apa pun.

  1. Pilih dataset (folder di dalam datasets/, atau folder lain di komputer) dan label setiap berapa frame (bawaan 2).
  2. Program memeriksa video mana yang sudah punya npz yang cocok.
  3. Pilih cara:
       a. Pakai npz yang ada           -> klik kotak YOLOX + ikuti otomatis
       b. 100% manual                  -> gambar kotak sendiri, tanpa npz
       c. Buat npz dengan YOLOX dulu   -> jalankan YOLOX (+ RTMPose) untuk video yang belum punya npz, lalu melabel
  4. Alat label (labeler/dense_labeler.py) terbuka. Setelah ditutup, kamu kembali ke jendela ini.

Struktur satu dataset:
  <dataset>/videos/          video (wajib)
  <dataset>/npz/             npz panduan (dibuat cara c, atau disalin sendiri)
  <dataset>/initial_labels.csv  opsional: label 8-cuplikan sebagai pemeriksa
  <dataset>/labels/              hasil label (dibuat otomatis)

Jalankan dari folder repo:
  python3 start.py
Tanpa jendela pemilih:
  python3 start.py --dataset datasets/example_dataset --cara npz|manual|yolox [--tanpa-pose]
"""
import argparse
import csv
import importlib.util
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

if sys.version_info < (3, 8):
    sys.exit("Butuh Python 3.8 atau lebih baru (teruji: 3.12).")

ROOT = Path(__file__).resolve().parent
TOOLS_DIR = ROOT / "labeler"
DATASET_DIR = ROOT / "datasets"
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
MODELS = ["lightweight", "balanced", "performance"]
# modul python -> nama paket pip
NEED_LABEL = {"cv2": "opencv-python", "numpy": "numpy", "PIL": "pillow"}
NEED_YOLOX = {"rtmlib": "rtmlib", "onnxruntime": "onnxruntime", "certifi": "certifi"}
DEVICE_INFO = {"cuda": "cuda = kartu grafis NVIDIA (paling cepat)",
               "mps": "mps = akselerator Apple lewat CoreML (coba; jika gagal otomatis kembali ke cpu)",
               "cpu": "cpu = prosesor biasa (selalu bisa, paling lambat)"}


def missing_packages(need):
    """Paket pip yang belum terpasang (dicek tanpa mengimpornya)."""
    return [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]


def pip_hint(pkgs):
    return f'"{sys.executable}" -m pip install ' + " ".join(pkgs)


def available_devices():
    """Perangkat yang benar-benar tersedia di onnxruntime komputer ini (urut dari yang tercepat). cpu selalu ada."""
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


# ----------------------------------------------------------------------------- dataset
_STRIDE_CACHE, _FPS_CACHE = {}, {}


def npz_step(path):
    """Setiap berapa frame npz ini dibuat (dari frames_processed). None jika tidak terbaca."""
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return None
    if key not in _STRIDE_CACHE:
        try:
            import numpy as np
            with np.load(path, allow_pickle=False) as d:
                fp = d["frames_processed"]
            _STRIDE_CACHE[key] = int(np.median(np.diff(fp))) if len(fp) > 1 else 1
        except Exception:
            _STRIDE_CACHE[key] = None
    return _STRIDE_CACHE[key]


def video_fps(path):
    key = str(path)
    if key not in _FPS_CACHE:
        fps = None
        try:
            import cv2
            cap = cv2.VideoCapture(key)
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else None
            cap.release()
        except Exception:
            pass
        _FPS_CACHE[key] = fps if fps and 0 < fps < 1000 else None
    return _FPS_CACHE[key]


class Dataset:
    def __init__(self, folder, step=2):
        self.step = step                     # label setiap N frame
        self.root = Path(folder).resolve()
        self.videos_dir = self.root / "videos" if (self.root / "videos").is_dir() else self.root
        self.npz_dir = self.root / "npz"
        self.out_dir = self.root / "labels"
        self.label_awal = self.root / "initial_labels.csv"
        self.refresh()

    @property
    def name(self):
        return self.root.name

    def refresh(self):
        skip = {self.npz_dir, self.out_dir}
        vids = {}
        if self.videos_dir.is_dir():
            for p in sorted(self.videos_dir.rglob("*")):
                if any(s in p.parents for s in skip):
                    continue
                if p.suffix in VIDEO_EXT and not p.name.startswith("._"):
                    vids.setdefault(p.stem, p)
        self.videos = dict(sorted(vids.items()))
        # npz cocok jika jarak framenya membagi step (npz setiap 2 frame cocok untuk label setiap 2, 4, 6, ... frame)
        self.with_npz, self.bad_npz, self.no_npz = [], [], []
        for v in self.videos:
            p = self.npz_dir / f"{v}.npz"
            if not p.exists():
                self.no_npz.append(v)
                continue
            s = npz_step(p)
            (self.with_npz if s and self.step % s == 0 else self.bad_npz).append(v)
        self.without_npz = self.no_npz + self.bad_npz          # yang perlu dibuat (ulang) oleh YOLOX
        self.labeled = [v for v in self.videos if (self.out_dir / f"{v}.csv").exists()]

    def summary(self):
        if not self.videos:
            return f"Belum ada video. Taruh video di: {self.videos_dir}"
        bad = f" ({len(self.bad_npz)} npz tidak cocok dengan jarak frame ini)" if self.bad_npz else ""
        return (f"{len(self.videos)} video  ·  {len(self.with_npz)} punya npz yang cocok{bad}  ·  "
                f"{len(self.no_npz)} belum punya npz  ·  {len(self.labeled)} sudah ada label")

    def rate_text(self):
        """Contoh: 'video 30 fps -> label setiap 2 frame = ~15.0 label per detik'."""
        fps = sorted({round(f, 2) for f in (video_fps(p) for p in self.videos.values()) if f})
        if not fps:
            return ""
        lo, hi = fps[0], fps[-1]
        fps_txt = f"{lo:g} fps" if lo == hi else f"{lo:g}–{hi:g} fps"
        rate = f"~{lo / self.step:.1f}" if lo == hi else f"~{lo / self.step:.1f}–{hi / self.step:.1f}"
        return (f"Video {fps_txt} → label setiap {self.step} frame = {rate} label per detik. "
                f"Paling rapat: setiap 1 frame (= fps video).")


def list_datasets(step=2):
    if not DATASET_DIR.is_dir():
        return []
    return [Dataset(p, step) for p in sorted(DATASET_DIR.iterdir()) if p.is_dir() and not p.name.startswith(".")]


def guess_encoding(path):
    try:
        with open(path, encoding="utf-8") as fh:
            fh.read()
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def check_label_awal(path):
    """Kembalikan pesan kesalahan, atau None jika kolom label awal lengkap."""
    try:
        with open(path, newline="", encoding=guess_encoding(path)) as fh:
            cols = set(csv.DictReader(fh).fieldnames or [])
    except OSError as e:
        return str(e)
    need = {"video_id", "frame_idx", "x1", "y1", "x2", "y2"}
    return None if need <= cols else f"kolom kurang: {', '.join(sorted(need - cols))}"


def labeler_cmd(ds, cara, label_awal=None):
    cmd = [sys.executable, str(TOOLS_DIR / "dense_labeler.py"), "--videos", str(ds.videos_dir), "--out", str(ds.out_dir),
           "--step", str(ds.step)]
    if cara != "manual" and ds.npz_dir.is_dir():
        cmd += ["--npz", str(ds.npz_dir)]
    if label_awal:
        cmd += ["--labels", str(label_awal), "--labels-encoding", guess_encoding(label_awal)]
    return cmd


def yolox_cmd(ds, model="balanced", device="auto", tanpa_pose=False):
    cmd = [sys.executable, "-u", str(TOOLS_DIR / "make_npz.py"), "--videos", str(ds.videos_dir),
           "--out", str(ds.npz_dir), "--mode", model, "--device", device, "--stride", str(ds.step)]
    return cmd + (["--tanpa-pose"] if tanpa_pose else [])


def open_folder(path):
    path = str(path)
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)  # noqa: (Windows)
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


# ----------------------------------------------------------------------------- tanpa jendela
def run_cli(args):
    if args.step < 1:
        sys.exit("--step minimal 1 (1 = setiap frame)")
    ds = Dataset(args.dataset, args.step)
    print(ds.summary())
    if ds.rate_text():
        print(ds.rate_text())
    if not ds.videos:
        sys.exit(1)
    miss = missing_packages(NEED_LABEL) + (missing_packages(NEED_YOLOX) if args.cara == "yolox" else [])
    if miss:
        sys.exit("Paket belum terpasang: " + ", ".join(miss) + "\nJalankan: " + pip_hint(miss))
    if args.cara == "npz" and not ds.with_npz:
        sys.exit("Belum ada npz untuk dataset ini. Pakai --cara yolox atau --cara manual.")
    if args.cara == "yolox" and ds.without_npz:
        r = subprocess.run(yolox_cmd(ds, args.model, args.device, args.tanpa_pose))
        if r.returncode != 0:
            sys.exit("Pembuatan npz tidak selesai. Jalankan lagi untuk melanjutkan.")
    la = Path(args.label_awal) if args.label_awal else (ds.label_awal if ds.label_awal.exists() else None)
    if la is not None:
        err = check_label_awal(la)
        if err:
            sys.exit(f"Label awal tidak bisa dipakai ({la}): {err}")
    sys.exit(subprocess.run(labeler_cmd(ds, args.cara, la)).returncode)


# ----------------------------------------------------------------------------- jendela pemilih
def run_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog, ttk
    except ImportError:
        sys.exit("Tkinter tidak tersedia. macOS + Homebrew: brew install python-tk@3.12 (sesuaikan versi Python).")

    class Launcher:
        def __init__(self, root):
            self.root = root
            self.datasets = []
            self.ds = None
            self.proc = None
            self.lines = queue.Queue()
            self.after_yolox = False
            self._build()
            self.reload()

        # ---------------- tampilan
        def _build(self):
            r = self.root
            r.title("Label padat · pilih dataset")
            r.geometry("820x780")
            r.minsize(720, 640)
            r.columnconfigure(0, weight=1)
            r.rowconfigure(3, weight=1)
            st = ttk.Style()
            st.configure("Head.TLabel", font=("TkDefaultFont", 13, "bold"))

            # 1. dataset
            f1 = ttk.Frame(r, padding=(12, 10, 12, 4)); f1.grid(row=0, column=0, sticky="nsew")
            f1.columnconfigure(0, weight=1)
            ttk.Label(f1, text="1. Pilih dataset", style="Head.TLabel").grid(row=0, column=0, sticky="w")
            self.lst = tk.Listbox(f1, height=6, activestyle="none", exportselection=False)
            self.lst.grid(row=1, column=0, sticky="ew", pady=(4, 4))
            self.lst.bind("<<ListboxSelect>>", lambda e: self._pick_from_list())
            b1 = ttk.Frame(f1); b1.grid(row=2, column=0, sticky="w")
            ttk.Button(b1, text="Dataset baru…", command=self.new_dataset).pack(side="left")
            ttk.Button(b1, text="Folder lain…", command=self.pick_other).pack(side="left", padx=6)
            ttk.Button(b1, text="Buka folder", command=lambda: self.ds and open_folder(self.ds.root)).pack(side="left")
            ttk.Button(b1, text="Muat ulang", command=self.reload).pack(side="left", padx=6)
            self.lbl_ds = ttk.Label(f1, text="", wraplength=780, justify="left")
            self.lbl_ds.grid(row=3, column=0, sticky="w", pady=(8, 0))
            srow = ttk.Frame(f1); srow.grid(row=4, column=0, sticky="w", pady=(8, 0))
            ttk.Label(srow, text="Label setiap").pack(side="left")
            self.var_step = tk.StringVar(value="2")
            self.sp_step = ttk.Spinbox(srow, from_=1, to=30, width=4, textvariable=self.var_step, command=self.on_step)
            self.sp_step.pack(side="left", padx=4)
            self.sp_step.bind("<Return>", lambda e: self.on_step())
            self.sp_step.bind("<FocusOut>", lambda e: self.on_step())
            ttk.Label(srow, text="frame   (bawaan 2)").pack(side="left")
            self.lbl_rate = ttk.Label(f1, text="", wraplength=780, justify="left", foreground="#555")
            self.lbl_rate.grid(row=5, column=0, sticky="w", pady=(2, 0))

            # 2. cara
            f2 = ttk.Frame(r, padding=(12, 8, 12, 4)); f2.grid(row=1, column=0, sticky="nsew")
            f2.columnconfigure(0, weight=1)
            ttk.Label(f2, text="2. Pilih cara melabel", style="Head.TLabel").grid(row=0, column=0, sticky="w")
            self.cara = tk.StringVar(value="npz")
            self.rb_npz = ttk.Radiobutton(f2, variable=self.cara, value="npz", command=self._update)
            self.rb_npz.grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.rb_man = ttk.Radiobutton(f2, variable=self.cara, value="manual", command=self._update,
                                          text="100% manual: gambar kotak sendiri, tanpa kotak YOLOX")
            self.rb_man.grid(row=2, column=0, sticky="w")
            self.rb_yolo = ttk.Radiobutton(f2, variable=self.cara, value="yolox", command=self._update)
            self.rb_yolo.grid(row=3, column=0, sticky="w")
            opt = ttk.Frame(f2, padding=(24, 2, 0, 0)); opt.grid(row=4, column=0, sticky="w")
            ttk.Label(opt, text="Model:").pack(side="left")
            self.var_model = tk.StringVar(value="balanced")
            self.cb_model = ttk.Combobox(opt, textvariable=self.var_model, values=MODELS, state="readonly", width=12)
            self.cb_model.pack(side="left", padx=(4, 12))
            ttk.Label(opt, text="Perangkat:").pack(side="left")
            self.devices = available_devices()          # hanya yang tersedia di komputer ini
            self.var_dev = tk.StringVar(value="cuda" if "cuda" in self.devices else "cpu")
            self.cb_dev = ttk.Combobox(opt, textvariable=self.var_dev, values=self.devices, state="readonly", width=6)
            self.cb_dev.pack(side="left", padx=(4, 12))
            self.cb_dev.bind("<<ComboboxSelected>>", lambda e: self._update())
            self.var_pose = tk.BooleanVar(value=True)
            self.ck_pose = ttk.Checkbutton(opt, text="Sertakan kerangka (RTMPose)", variable=self.var_pose)
            self.ck_pose.pack(side="left")
            self.lbl_cara = ttk.Label(f2, text="", wraplength=780, justify="left", foreground="#555")
            self.lbl_cara.grid(row=5, column=0, sticky="w", pady=(6, 0))

            # 3. label awal (opsional)
            f3 = ttk.Frame(r, padding=(12, 8, 12, 4)); f3.grid(row=2, column=0, sticky="nsew")
            f3.columnconfigure(1, weight=1)
            ttk.Label(f3, text="3. Label awal (opsional)", style="Head.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
            self.var_use_la = tk.BooleanVar(value=False)
            ttk.Checkbutton(f3, text="Pakai", variable=self.var_use_la).grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.var_la = tk.StringVar(value="")
            ttk.Entry(f3, textvariable=self.var_la).grid(row=1, column=1, sticky="ew", padx=6, pady=(4, 0))
            ttk.Button(f3, text="Pilih…", command=self.pick_label_awal).grid(row=1, column=2, pady=(4, 0))
            ttk.Label(f3, text="CSV berkolom video_id, frame_idx, x1, y1, x2, y2 (mis. patient_labels.csv). "
                               "Ditampilkan kuning dan dipakai untuk memeriksa label otomatis.",
                      foreground="#555", wraplength=780).grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))

            # 4. log + tombol
            f4 = ttk.Frame(r, padding=(12, 8, 12, 12)); f4.grid(row=3, column=0, sticky="nsew")
            f4.columnconfigure(0, weight=1); f4.rowconfigure(1, weight=1)
            ttk.Label(f4, text="Proses", style="Head.TLabel").grid(row=0, column=0, sticky="w")
            box = ttk.Frame(f4); box.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
            box.columnconfigure(0, weight=1); box.rowconfigure(0, weight=1)
            self.log = tk.Text(box, height=8, wrap="word", state="disabled", font=("TkFixedFont", 10))
            self.log.grid(row=0, column=0, sticky="nsew")
            sb = ttk.Scrollbar(box, command=self.log.yview); sb.grid(row=0, column=1, sticky="ns")
            self.log.configure(yscrollcommand=sb.set)
            self.lbl_prog = ttk.Label(f4, text=""); self.lbl_prog.grid(row=2, column=0, sticky="w")
            bb = ttk.Frame(f4); bb.grid(row=3, column=0, sticky="ew", pady=(6, 0))
            self.btn_go = ttk.Button(bb, text="Mulai", command=self.start); self.btn_go.pack(side="left")
            self.btn_stop = ttk.Button(bb, text="Hentikan YOLOX", command=self.stop_yolox, state="disabled")
            self.btn_stop.pack(side="left", padx=6)
            ttk.Button(bb, text="Keluar", command=self.quit).pack(side="right")
            r.protocol("WM_DELETE_WINDOW", self.quit)

        def write(self, text):
            self.log.configure(state="normal")
            self.log.insert("end", text.rstrip("\n") + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")

        # ---------------- dataset
        def get_step(self):
            try:
                return min(max(int(float(self.var_step.get())), 1), 30)
            except (ValueError, tk.TclError):
                return 2

        def on_step(self):
            self.var_step.set(str(self.get_step()))
            if self.ds is not None:
                self.set_dataset(self.ds)

        def reload(self):
            keep = self.ds.root if self.ds else None
            self.datasets = list_datasets(self.get_step())
            if self.ds is not None and all(d.root != self.ds.root for d in self.datasets):
                self.ds.refresh(); self.datasets.append(self.ds)       # folder lain yang dipilih sendiri
            self.lst.delete(0, "end")
            for d in self.datasets:
                lab = "  [folder lain]" if not str(d.root).startswith(str(DATASET_DIR.resolve())) else ""
                self.lst.insert("end", f"{d.name}   ({len(d.videos)} video, {len(d.with_npz)} npz){lab}")
            idx = next((k for k, d in enumerate(self.datasets) if d.root == keep), 0 if self.datasets else None)
            if idx is not None:
                self.lst.selection_clear(0, "end"); self.lst.selection_set(idx); self.lst.see(idx)
                self.set_dataset(self.datasets[idx])
            else:
                self.ds = None
                self.lbl_ds.configure(text=f"Belum ada dataset. Klik 'Dataset baru…' atau taruh folder di {DATASET_DIR}")
                self._update()

        def _pick_from_list(self):
            sel = self.lst.curselection()
            if sel:
                self.set_dataset(self.datasets[sel[0]])

        def set_dataset(self, ds):
            self.ds = ds
            ds.step = self.get_step()
            ds.refresh()
            self.lbl_ds.configure(text=f"{ds.root}\n{ds.summary()}")
            self.lbl_rate.configure(text=ds.rate_text())
            if ds.label_awal.exists():
                self.var_la.set(str(ds.label_awal)); self.var_use_la.set(True)
            elif self.var_la.get().startswith(str(DATASET_DIR.resolve())):
                self.var_la.set(""); self.var_use_la.set(False)
            if ds.with_npz:
                self.cara.set("npz")
            elif self.cara.get() == "npz":
                self.cara.set("yolox")
            self._update()

        def new_dataset(self):
            name = simpledialog.askstring("Dataset baru", "Nama dataset (tanpa spasi lebih baik):", parent=self.root)
            if not name:
                return
            name = name.strip().replace("/", "_").replace("\\", "_")
            p = DATASET_DIR / name
            (p / "videos").mkdir(parents=True, exist_ok=True)
            self.ds = Dataset(p, self.get_step())
            self.reload()
            open_folder(p / "videos")
            messagebox.showinfo("Dataset baru", f"Folder dibuat:\n{p / 'videos'}\n\nTaruh video di sana, lalu klik 'Muat ulang'.")

        def pick_other(self):
            d = filedialog.askdirectory(title="Pilih folder dataset (berisi folder videos/, atau langsung berisi video)")
            if d:
                self.ds = Dataset(d, self.get_step())
                self.reload()

        def pick_label_awal(self):
            f = filedialog.askopenfilename(title="Pilih CSV label awal", filetypes=[("CSV", "*.csv"), ("Semua", "*")])
            if f:
                self.var_la.set(f); self.var_use_la.set(True)

        # ---------------- cara
        def _update(self):
            ds = self.ds
            nv = len(ds.videos) if ds else 0
            nn = len(ds.with_npz) if ds else 0
            nw = len(ds.without_npz) if ds else 0
            nb = len(ds.bad_npz) if ds else 0
            self.rb_npz.configure(text=f"Pakai npz yang ada: klik kotak YOLOX + ikuti otomatis ({nn}/{nv} video punya npz yang cocok)",
                                  state="normal" if nn else "disabled")
            self.rb_yolo.configure(text=(f"Buat npz dengan YOLOX dulu untuk {nw} video"
                                         + (f" ({nb} di antaranya dibuat ulang karena jarak framenya tidak cocok)" if nb else "")
                                         + ", lalu melabel"
                                         if nw else "Buat npz dengan YOLOX dulu (semua video sudah punya npz yang cocok)"),
                                   state="normal" if nw else "disabled")
            if self.cara.get() == "npz" and not nn:
                self.cara.set("yolox" if nw else "manual")
            if self.cara.get() == "yolox" and not nw:
                self.cara.set("npz" if nn else "manual")
            c = self.cara.get()
            yolo_state = "readonly" if c == "yolox" else "disabled"
            self.cb_model.configure(state=yolo_state); self.cb_dev.configure(state=yolo_state)
            self.ck_pose.configure(state="normal" if c == "yolox" else "disabled")
            if not nv:
                txt = "Tambahkan video ke dataset dulu."
            elif c == "npz":
                txt = ("Semua video punya npz yang cocok." if not nw else
                       f"{nw} video tanpa npz yang cocok akan dilabel manual (gambar kotak sendiri). "
                       "Pilih 'Buat npz' kalau ingin semuanya punya panduan YOLOX.")
                if nb:
                    txt += (f" npz cocok jika dibuat dengan jarak frame yang membagi {ds.step} "
                            "(npz setiap 2 frame cocok untuk label setiap 2, 4, 6, ... frame).")
            elif c == "manual":
                txt = "npz tidak dipakai sama sekali, walaupun ada. Gambar kotak pasien, lalu Spasi menyalin kotak itu ke langkah berikutnya."
            else:
                miss = missing_packages(NEED_YOLOX)
                dev = "Perangkat tersedia di komputer ini: " + "; ".join(DEVICE_INFO[d] for d in self.devices) + "."
                if miss:
                    txt = f"Paket untuk YOLOX belum terpasang ({', '.join(miss)}). Jalankan di terminal: {pip_hint(miss)}"
                else:
                    txt = ("YOLOX berjalan di komputer ini: di cpu bisa beberapa menit per video (~1800 langkah per video "
                           "2 menit; perkiraan sisa waktu tampil di bawah). Model diunduh saat pertama kali (perlu internet). "
                           "Bisa dihentikan kapan saja; video yang sudah selesai tetap tersimpan. Lebih cepat: model "
                           "'lightweight' atau matikan kerangka. 'balanced' = sama dengan Stage 2.\n" + dev)
            self.lbl_cara.configure(text=txt)
            busy = self.proc is not None
            self.btn_go.configure(state="normal" if (nv and not busy) else "disabled")

        # ---------------- jalankan
        def _label_awal(self):
            if not self.var_use_la.get() or not self.var_la.get().strip():
                return None
            p = Path(self.var_la.get().strip())
            if not p.exists():
                messagebox.showerror("Label awal", f"File tidak ada:\n{p}"); return False
            err = check_label_awal(p)
            if err:
                messagebox.showerror("Label awal", f"File tidak bisa dipakai:\n{p}\n\n{err}"); return False
            return p

        def start(self):
            ds = self.ds
            if ds is None or not ds.videos:
                return
            ds.refresh()
            c = self.cara.get()
            miss = missing_packages(NEED_LABEL) + (missing_packages(NEED_YOLOX) if c == "yolox" and ds.without_npz else [])
            if miss:
                messagebox.showerror("Paket belum terpasang",
                                     "Paket belum terpasang: " + ", ".join(miss) + "\n\nJalankan di terminal:\n" + pip_hint(miss))
                return
            la = self._label_awal()
            if la is False:
                return
            if c == "yolox" and ds.without_npz:
                if ds.bad_npz and not messagebox.askyesno(
                        "Buat ulang npz",
                        f"{len(ds.bad_npz)} npz dibuat dengan jarak frame yang tidak cocok dengan label setiap {ds.step} frame "
                        f"dan akan DITIMPA (dibuat ulang setiap {ds.step} frame):\n{', '.join(ds.bad_npz[:10])}"
                        + (" …" if len(ds.bad_npz) > 10 else "") + "\n\nLanjut?"):
                    return
                self.run_yolox(ds)
            else:
                self.run_labeler(ds, "manual" if c == "manual" else "npz", la)

        def run_yolox(self, ds):
            cmd = yolox_cmd(ds, self.var_model.get(), self.var_dev.get(), not self.var_pose.get())
            self.write("$ " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            except Exception as e:
                self.proc = None
                messagebox.showerror("YOLOX", f"Gagal menjalankan make_npz.py:\n{e}"); return
            self.btn_stop.configure(state="normal")
            self._update()
            threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
            self.root.after(200, self._poll_yolox)

        def _reader(self, proc):
            for line in proc.stdout:
                self.lines.put(line)
            proc.stdout.close()

        def _poll_yolox(self):
            while True:
                try:
                    line = self.lines.get_nowait()
                except queue.Empty:
                    break
                if line.startswith("[PROGRES]"):
                    self.lbl_prog.configure(text=line[10:].strip())
                else:
                    self.write(line)
            if self.proc is None:
                return
            rc = self.proc.poll()
            if rc is None:
                self.root.after(200, self._poll_yolox); return
            self.root.after(300, lambda: self._yolox_done(rc))

        def _yolox_done(self, rc):
            while not self.lines.empty():                       # sisa keluaran
                self.write(self.lines.get_nowait())
            self.proc = None
            self.btn_stop.configure(state="disabled")
            self.lbl_prog.configure(text="")
            ds = self.ds
            ds.refresh()
            self.set_dataset(ds)
            if rc == 0:
                self.write("Pembuatan npz selesai.")
                la = self._label_awal()
                if la is not False and messagebox.askyesno("YOLOX selesai", "Semua npz sudah dibuat. Buka alat label sekarang?"):
                    self.run_labeler(ds, "npz", la)
            else:
                self.write(f"Pembuatan npz berhenti (kode {rc}). Video yang sudah selesai tetap tersimpan; "
                           "klik Mulai lagi untuk melanjutkan.")

        def stop_yolox(self):
            if self.proc is not None and self.proc.poll() is None:
                self.write("Menghentikan YOLOX… (video yang sedang diproses tidak disimpan)")
                self.proc.terminate()

        def run_labeler(self, ds, cara, la):
            cmd = labeler_cmd(ds, cara, la)
            self.write("$ " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
            try:
                p = subprocess.Popen(cmd)
            except Exception as e:
                messagebox.showerror("Alat label", f"Gagal membuka alat label:\n{e}"); return
            self.root.withdraw()
            self._wait_labeler(p)

        def _wait_labeler(self, p):
            rc = p.poll()
            if rc is None:
                self.root.after(500, lambda: self._wait_labeler(p)); return
            self.root.deiconify()
            self.write(f"Alat label ditutup (kode {rc}). Hasil di: {self.ds.out_dir}")
            self.set_dataset(self.ds)

        def quit(self):
            if self.proc is not None and self.proc.poll() is None:
                if not messagebox.askyesno("Keluar", "YOLOX masih berjalan. Hentikan dan keluar?"):
                    return
                self.proc.terminate()
            self.root.destroy()

    if os.name == "nt":                      # Windows: tampilan tajam di layar resolusi tinggi
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Pilih dataset dan cara melabel, lalu buka alat label padat.")
    ap.add_argument("--dataset", help="folder dataset; jika diisi, jendela pemilih dilewati")
    ap.add_argument("--cara", choices=["npz", "manual", "yolox"], default="npz",
                    help="npz = pakai npz yang ada; manual = tanpa npz; yolox = buat npz yang belum ada dulu")
    ap.add_argument("--model", choices=MODELS, default="balanced", help="untuk --cara yolox (default balanced)")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="untuk --cara yolox (default auto: cuda jika ada, selain itu cpu)")
    ap.add_argument("--tanpa-pose", action="store_true", help="untuk --cara yolox: tanpa RTMPose (lebih cepat)")
    ap.add_argument("--label-awal", help="CSV label awal (default <dataset>/initial_labels.csv jika ada)")
    ap.add_argument("--step", type=int, default=2, help="label setiap N frame (default 2; 1 = setiap frame)")
    args = ap.parse_args()
    if args.dataset:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
