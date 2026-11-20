#!/usr/bin/env python3
"""
start.py - entry point of dense-person-labeler: dense labelling (every N frames) for any video dataset.

  1. Choose a dataset (a folder inside datasets/, or any other folder) and the frame step (default 2).
  2. The program checks which videos already have a matching npz.
  3. Choose a mode:
       a. Use existing npz            -> click a YOLOX box + automatic following
       b. 100% manual                 -> draw boxes yourself, no npz
       c. Create npz with YOLOX first -> run YOLOX (+ RTMPose) for videos without a matching npz, then label
  4. The labeller (labeler/dense_labeler.py) opens. When it closes, you return to this window.

Dataset layout:
  <dataset>/videos/              videos (required)
  <dataset>/npz/                 guidance npz (created by mode c, or copied by you)
  <dataset>/initial_labels.csv   optional: initial labels used as a check
  <dataset>/labels/              results (created automatically)

Run from the repository folder:
  python3 start.py
Without the launcher window:
  python3 start.py --dataset datasets/example_dataset --mode npz|manual|yolox [--no-pose]
"""
import argparse
import csv
import importlib.util
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or newer is required (tested: 3.12).")

ROOT = Path(__file__).resolve().parent
TOOLS_DIR = ROOT / "labeler"
DATASET_DIR = ROOT / "datasets"
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
MODELS = ["lightweight", "balanced", "performance"]
DEFAULT_LABEL = "target"     # object label name when none is given
# python module -> pip package name
NEED_LABELER = {"cv2": "opencv-python", "numpy": "numpy", "PIL": "pillow"}
NEED_YOLOX = {"rtmlib": "rtmlib", "onnxruntime": "onnxruntime", "certifi": "certifi"}
DEVICE_INFO = {"cuda": "cuda = NVIDIA graphics card (fastest)",
               "mps": "mps = Apple accelerator via CoreML (try it; falls back to cpu automatically if it fails)",
               "cpu": "cpu = regular processor (always works, slowest)"}


def missing_packages(need):
    """pip packages that are not installed (checked without importing them)."""
    return [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]


def pip_hint(pkgs):
    return f'"{sys.executable}" -m pip install ' + " ".join(pkgs)


def available_devices():
    """Devices actually available to onnxruntime on this computer (fastest first). cpu is always there."""
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
    """Every how many frames this npz was made (from frames_processed). None if it cannot be read."""
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
        self.step = step                     # label every N frames
        self.root = Path(folder).resolve()
        self.videos_dir = self.root / "videos" if (self.root / "videos").is_dir() else self.root
        self.npz_dir = self.root / "npz"
        self.out_dir = self.root / "labels"
        self.initial_labels = self.root / "initial_labels.csv"
        self.trash_dir = self.root / ".trash"     # deleted npz / label files are moved here
        self.refresh()

    @property
    def name(self):
        return self.root.name

    def refresh(self):
        skip = {self.npz_dir, self.out_dir, self.trash_dir}
        vids = {}
        if self.videos_dir.is_dir():
            for p in sorted(self.videos_dir.rglob("*")):
                if any(s in p.parents for s in skip):
                    continue
                if p.suffix in VIDEO_EXT and not p.name.startswith("._"):
                    vids.setdefault(p.stem, p)
        self.videos = dict(sorted(vids.items()))
        # an npz matches if its frame spacing divides step (npz every 2 frames matches labelling every 2, 4, 6, ... frames)
        self.with_npz, self.bad_npz, self.no_npz = [], [], []
        for v in self.videos:
            p = self.npz_dir / f"{v}.npz"
            if not p.exists():
                self.no_npz.append(v)
                continue
            s = npz_step(p)
            (self.with_npz if s and self.step % s == 0 else self.bad_npz).append(v)
        self.without_npz = self.no_npz + self.bad_npz          # those YOLOX must create (or recreate)
        self.labeled = [v for v in self.videos if (self.out_dir / f"{v}.csv").exists()]

    def summary(self):
        if not self.videos:
            return f"No videos yet. Put videos in: {self.videos_dir}"
        bad = f" ({len(self.bad_npz)} npz do not match this frame step)" if self.bad_npz else ""
        return (f"{len(self.videos)} videos  ·  {len(self.with_npz)} with a matching npz{bad}  ·  "
                f"{len(self.no_npz)} without npz  ·  {len(self.labeled)} already have labels")

    def rate_text(self):
        """E.g. 'Videos 30 fps -> labelling every 2 frames = ~15.0 labels per second'."""
        fps = sorted({round(f, 2) for f in (video_fps(p) for p in self.videos.values()) if f})
        if not fps:
            return ""
        lo, hi = fps[0], fps[-1]
        fps_txt = f"{lo:g} fps" if lo == hi else f"{lo:g}–{hi:g} fps"
        rate = f"~{lo / self.step:.1f}" if lo == hi else f"~{lo / self.step:.1f}–{hi / self.step:.1f}"
        return (f"Videos {fps_txt} → labelling every {self.step} frames = {rate} labels per second. "
                f"Densest: every 1 frame (= video fps).")


def labels_in_csv(path):
    """Object labels present in a labels CSV (rows without a label count as the default label)."""
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return sorted({(r.get("label") or "").strip() or DEFAULT_LABEL for r in csv.DictReader(fh)})
    except Exception:
        return ["(unreadable)"]


def move_to_trash(path, trash_dir):
    """Move a file into the dataset's .trash folder with a timestamp, so it can be restored."""
    trash_dir.mkdir(parents=True, exist_ok=True)
    dst = trash_dir / f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}"
    shutil.move(str(path), str(dst))
    return dst


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


def check_initial_labels(path):
    """Return an error message, or None if the initial-label columns are complete."""
    try:
        with open(path, newline="", encoding=guess_encoding(path)) as fh:
            cols = set(csv.DictReader(fh).fieldnames or [])
    except OSError as e:
        return str(e)
    need = {"video_id", "frame_idx", "x1", "y1", "x2", "y2"}
    return None if need <= cols else f"missing columns: {', '.join(sorted(need - cols))}"


def labeler_cmd(ds, mode, initial_labels=None, label=DEFAULT_LABEL):
    cmd = [sys.executable, str(TOOLS_DIR / "dense_labeler.py"), "--videos", str(ds.videos_dir), "--out", str(ds.out_dir),
           "--step", str(ds.step), "--label-name", label]
    if mode != "manual" and ds.npz_dir.is_dir():
        cmd += ["--npz", str(ds.npz_dir)]
    if initial_labels:
        cmd += ["--initial-labels", str(initial_labels), "--initial-labels-encoding", guess_encoding(initial_labels)]
    return cmd


def yolox_cmd(ds, model="balanced", device="auto", no_pose=False, redo=False):
    cmd = [sys.executable, "-u", str(TOOLS_DIR / "make_npz.py"), "--videos", str(ds.videos_dir),
           "--out", str(ds.npz_dir), "--model", model, "--device", device, "--stride", str(ds.step)]
    return cmd + (["--no-pose"] if no_pose else []) + (["--redo"] if redo else [])


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


# ----------------------------------------------------------------------------- without the window
def run_cli(args):
    if args.step < 1:
        sys.exit("--step must be at least 1 (1 = every frame)")
    ds = Dataset(args.dataset, args.step)
    print(ds.summary())
    if ds.rate_text():
        print(ds.rate_text())
    if not ds.videos:
        sys.exit(1)
    miss = missing_packages(NEED_LABELER) + (missing_packages(NEED_YOLOX) if args.mode == "yolox" else [])
    if miss:
        sys.exit("Packages not installed: " + ", ".join(miss) + "\nRun: " + pip_hint(miss))
    if args.mode == "npz" and not ds.with_npz:
        sys.exit("This dataset has no npz yet. Use --mode yolox or --mode manual.")
    if args.mode == "yolox" and (ds.without_npz or args.redo):
        r = subprocess.run(yolox_cmd(ds, args.model, args.device, args.no_pose, args.redo))
        if r.returncode != 0:
            sys.exit("Creating npz did not finish. Run again to continue.")
    init = Path(args.initial_labels) if args.initial_labels else (ds.initial_labels if ds.initial_labels.exists() else None)
    if init is not None:
        err = check_initial_labels(init)
        if err:
            sys.exit(f"Initial labels cannot be used ({init}): {err}")
    sys.exit(subprocess.run(labeler_cmd(ds, args.mode, init, args.label_name.strip() or DEFAULT_LABEL)).returncode)


# ----------------------------------------------------------------------------- launcher window
def run_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog, ttk
    except ImportError:
        sys.exit("Tkinter is not available. macOS + Homebrew: brew install python-tk@3.12 (match your Python version).")

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

        # ---------------- layout
        def _build(self):
            r = self.root
            r.title("Dense person labeler · choose dataset")
            r.geometry("820x860")
            r.minsize(720, 640)
            r.columnconfigure(0, weight=1)
            r.rowconfigure(3, weight=1)
            st = ttk.Style()
            st.configure("Head.TLabel", font=("TkDefaultFont", 13, "bold"))

            # 1. dataset
            f1 = ttk.Frame(r, padding=(12, 10, 12, 4)); f1.grid(row=0, column=0, sticky="nsew")
            f1.columnconfigure(0, weight=1)
            ttk.Label(f1, text="1. Choose a dataset", style="Head.TLabel").grid(row=0, column=0, sticky="w")
            self.lst = tk.Listbox(f1, height=6, activestyle="none", exportselection=False)
            self.lst.grid(row=1, column=0, sticky="ew", pady=(4, 4))
            self.lst.bind("<<ListboxSelect>>", lambda e: self._pick_from_list())
            b1 = ttk.Frame(f1); b1.grid(row=2, column=0, sticky="w")
            ttk.Button(b1, text="New dataset…", command=self.new_dataset).pack(side="left")
            ttk.Button(b1, text="Other folder…", command=self.pick_other).pack(side="left", padx=6)
            ttk.Button(b1, text="Open folder", command=lambda: self.ds and open_folder(self.ds.root)).pack(side="left")
            ttk.Button(b1, text="Reload", command=self.reload).pack(side="left", padx=6)
            ttk.Button(b1, text="Manage files…", command=self.manage_files).pack(side="left")
            self.lbl_ds = ttk.Label(f1, text="", wraplength=780, justify="left")
            self.lbl_ds.grid(row=3, column=0, sticky="w", pady=(8, 0))
            srow = ttk.Frame(f1); srow.grid(row=4, column=0, sticky="w", pady=(8, 0))
            ttk.Label(srow, text="Label every").pack(side="left")
            self.var_step = tk.StringVar(value="2")
            self.sp_step = ttk.Spinbox(srow, from_=1, to=30, width=4, textvariable=self.var_step, command=self.on_step)
            self.sp_step.pack(side="left", padx=4)
            self.sp_step.bind("<Return>", lambda e: self.on_step())
            self.sp_step.bind("<FocusOut>", lambda e: self.on_step())
            ttk.Label(srow, text="frames   (default 2)").pack(side="left")
            self.lbl_rate = ttk.Label(f1, text="", wraplength=780, justify="left", foreground="#555")
            self.lbl_rate.grid(row=5, column=0, sticky="w", pady=(2, 0))
            lrow = ttk.Frame(f1); lrow.grid(row=6, column=0, sticky="w", pady=(8, 0))
            ttk.Label(lrow, text="Object label").pack(side="left")
            self.var_label = tk.StringVar(value=DEFAULT_LABEL)
            ttk.Entry(lrow, textvariable=self.var_label, width=20).pack(side="left", padx=4)
            ttk.Label(f1, text="Name of the object you label, e.g. patient, examiner, dog. Saved in the 'label' column. "
                               "Run again with another name to label another object in the same videos.",
                      wraplength=780, justify="left", foreground="#555").grid(row=7, column=0, sticky="w", pady=(2, 0))

            # 2. mode
            f2 = ttk.Frame(r, padding=(12, 8, 12, 4)); f2.grid(row=1, column=0, sticky="nsew")
            f2.columnconfigure(0, weight=1)
            ttk.Label(f2, text="2. Choose a labelling mode", style="Head.TLabel").grid(row=0, column=0, sticky="w")
            self.mode = tk.StringVar(value="npz")
            self.rb_npz = ttk.Radiobutton(f2, variable=self.mode, value="npz", command=self._update)
            self.rb_npz.grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.rb_man = ttk.Radiobutton(f2, variable=self.mode, value="manual", command=self._update,
                                          text="100% manual: draw boxes yourself, no YOLOX boxes")
            self.rb_man.grid(row=2, column=0, sticky="w")
            self.rb_yolo = ttk.Radiobutton(f2, variable=self.mode, value="yolox", command=self._update)
            self.rb_yolo.grid(row=3, column=0, sticky="w")
            opt = ttk.Frame(f2, padding=(24, 2, 0, 0)); opt.grid(row=4, column=0, sticky="w")
            ttk.Label(opt, text="Model:").pack(side="left")
            self.var_model = tk.StringVar(value="balanced")
            self.cb_model = ttk.Combobox(opt, textvariable=self.var_model, values=MODELS, state="readonly", width=12)
            self.cb_model.pack(side="left", padx=(4, 12))
            ttk.Label(opt, text="Device:").pack(side="left")
            self.devices = available_devices()          # only what this computer has
            self.var_dev = tk.StringVar(value="cuda" if "cuda" in self.devices else "cpu")
            self.cb_dev = ttk.Combobox(opt, textvariable=self.var_dev, values=self.devices, state="readonly", width=6)
            self.cb_dev.pack(side="left", padx=(4, 12))
            self.cb_dev.bind("<<ComboboxSelected>>", lambda e: self._update())
            self.var_pose = tk.BooleanVar(value=True)
            self.ck_pose = ttk.Checkbutton(opt, text="Include skeleton (RTMPose)", variable=self.var_pose)
            self.ck_pose.pack(side="left")
            self.var_redo = tk.BooleanVar(value=False)
            self.ck_redo = ttk.Checkbutton(f2, text="Recreate existing npz too (overwrite)", variable=self.var_redo,
                                           command=self._update)
            self.ck_redo.grid(row=6, column=0, sticky="w", padx=(24, 0), pady=(2, 0))
            self.lbl_mode = ttk.Label(f2, text="", wraplength=780, justify="left", foreground="#555")
            self.lbl_mode.grid(row=7, column=0, sticky="w", pady=(6, 0))

            # 3. initial labels (optional)
            f3 = ttk.Frame(r, padding=(12, 8, 12, 4)); f3.grid(row=2, column=0, sticky="nsew")
            f3.columnconfigure(1, weight=1)
            ttk.Label(f3, text="3. Initial labels (optional)", style="Head.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
            self.var_use_init = tk.BooleanVar(value=False)
            ttk.Checkbutton(f3, text="Use", variable=self.var_use_init).grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.var_init = tk.StringVar(value="")
            ttk.Entry(f3, textvariable=self.var_init).grid(row=1, column=1, sticky="ew", padx=6, pady=(4, 0))
            ttk.Button(f3, text="Browse…", command=self.pick_initial_labels).grid(row=1, column=2, pady=(4, 0))
            ttk.Label(f3, text="CSV with columns video_id, frame_idx, x1, y1, x2, y2. "
                               "Shown in yellow and used to check automatic labels.",
                      foreground="#555", wraplength=780).grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))

            # 4. log + buttons
            f4 = ttk.Frame(r, padding=(12, 8, 12, 12)); f4.grid(row=3, column=0, sticky="nsew")
            f4.columnconfigure(0, weight=1); f4.rowconfigure(1, weight=1)
            ttk.Label(f4, text="Progress", style="Head.TLabel").grid(row=0, column=0, sticky="w")
            box = ttk.Frame(f4); box.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
            box.columnconfigure(0, weight=1); box.rowconfigure(0, weight=1)
            self.log = tk.Text(box, height=8, wrap="word", state="disabled", font=("TkFixedFont", 10))
            self.log.grid(row=0, column=0, sticky="nsew")
            sb = ttk.Scrollbar(box, command=self.log.yview); sb.grid(row=0, column=1, sticky="ns")
            self.log.configure(yscrollcommand=sb.set)
            self.lbl_prog = ttk.Label(f4, text=""); self.lbl_prog.grid(row=2, column=0, sticky="w")
            bb = ttk.Frame(f4); bb.grid(row=3, column=0, sticky="ew", pady=(6, 0))
            self.btn_go = ttk.Button(bb, text="Start", command=self.start); self.btn_go.pack(side="left")
            self.btn_stop = ttk.Button(bb, text="Stop YOLOX", command=self.stop_yolox, state="disabled")
            self.btn_stop.pack(side="left", padx=6)
            ttk.Button(bb, text="Quit", command=self.quit).pack(side="right")
            r.protocol("WM_DELETE_WINDOW", self.quit)

        def write(self, text):
            self.log.configure(state="normal")
            self.log.insert("end", text.rstrip("\n") + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")

        # ---------------- dataset
        def get_label(self):
            return self.var_label.get().strip() or DEFAULT_LABEL

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
                self.ds.refresh(); self.datasets.append(self.ds)       # an outside folder chosen by the user
            self.lst.delete(0, "end")
            for d in self.datasets:
                tag = "  [other folder]" if not str(d.root).startswith(str(DATASET_DIR.resolve())) else ""
                self.lst.insert("end", f"{d.name}   ({len(d.videos)} videos, {len(d.with_npz)} npz){tag}")
            idx = next((k for k, d in enumerate(self.datasets) if d.root == keep), 0 if self.datasets else None)
            if idx is not None:
                self.lst.selection_clear(0, "end"); self.lst.selection_set(idx); self.lst.see(idx)
                self.set_dataset(self.datasets[idx])
            else:
                self.ds = None
                self.lbl_ds.configure(text=f"No dataset yet. Click 'New dataset…' or put a folder in {DATASET_DIR}")
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
            if ds.initial_labels.exists():
                self.var_init.set(str(ds.initial_labels)); self.var_use_init.set(True)
            elif self.var_init.get().startswith(str(DATASET_DIR.resolve())):
                self.var_init.set(""); self.var_use_init.set(False)
            if ds.with_npz:
                self.mode.set("npz")
            elif self.mode.get() == "npz":
                self.mode.set("yolox")
            self._update()

        def new_dataset(self):
            name = simpledialog.askstring("New dataset", "Dataset name (preferably without spaces):", parent=self.root)
            if not name:
                return
            name = name.strip().replace("/", "_").replace("\\", "_")
            p = DATASET_DIR / name
            (p / "videos").mkdir(parents=True, exist_ok=True)
            self.ds = Dataset(p, self.get_step())
            self.reload()
            open_folder(p / "videos")
            messagebox.showinfo("New dataset", f"Folder created:\n{p / 'videos'}\n\nPut videos there, then click 'Reload'.")

        def pick_other(self):
            d = filedialog.askdirectory(title="Choose a dataset folder (containing a videos/ folder, or the videos directly)")
            if d:
                self.ds = Dataset(d, self.get_step())
                self.reload()

        def pick_initial_labels(self):
            f = filedialog.askopenfilename(title="Choose an initial-labels CSV", filetypes=[("CSV", "*.csv"), ("All", "*")])
            if f:
                self.var_init.set(f); self.var_use_init.set(True)

        # ---------------- mode
        def _update(self):
            ds = self.ds
            nv = len(ds.videos) if ds else 0
            nn = len(ds.with_npz) if ds else 0
            nw = len(ds.without_npz) if ds else 0
            nb = len(ds.bad_npz) if ds else 0
            redo = self.var_redo.get() and nv > 0
            self.rb_npz.configure(text=f"Use existing npz: click a YOLOX box + automatic following ({nn}/{nv} videos have a matching npz)",
                                  state="normal" if nn else "disabled")
            if redo:
                yolo_txt = f"Create npz with YOLOX first for all {nv} videos (existing npz are overwritten), then label"
            elif nw:
                yolo_txt = (f"Create npz with YOLOX first for {nw} videos"
                            + (f" ({nb} of them recreated because their frame step does not match)" if nb else "")
                            + ", then label")
            else:
                yolo_txt = "Create npz with YOLOX first (all videos already have a matching npz; tick 'Recreate' to redo them)"
            self.rb_yolo.configure(text=yolo_txt, state="normal" if (nw or redo) else "disabled")
            self.ck_redo.configure(state="normal" if nv else "disabled")
            if self.mode.get() == "npz" and not nn:
                self.mode.set("yolox" if nw else "manual")
            if self.mode.get() == "yolox" and not nw and not redo:
                self.mode.set("npz" if nn else "manual")
            m = self.mode.get()
            yolo_state = "readonly" if m == "yolox" else "disabled"
            self.cb_model.configure(state=yolo_state); self.cb_dev.configure(state=yolo_state)
            self.ck_pose.configure(state="normal" if m == "yolox" else "disabled")
            if not nv:
                txt = "Add videos to the dataset first."
            elif m == "npz":
                txt = ("All videos have a matching npz." if not nw else
                       f"{nw} videos without a matching npz will be labelled manually (draw boxes yourself). "
                       "Choose 'Create npz' if you want YOLOX guidance for all of them.")
                if nb:
                    txt += (f" An npz matches if its frame step divides {ds.step} "
                            "(an npz made every 2 frames matches labelling every 2, 4, 6, ... frames).")
            elif m == "manual":
                txt = "npz files are not used at all, even if present. Draw the target's box, then Space copies it to the next steps."
            else:
                miss = missing_packages(NEED_YOLOX)
                dev = "Devices available on this computer: " + "; ".join(DEVICE_INFO[d] for d in self.devices) + "."
                if miss:
                    txt = f"Packages for YOLOX are not installed ({', '.join(miss)}). Run in a terminal: {pip_hint(miss)}"
                else:
                    txt = ("YOLOX runs on this computer: on cpu it can take several minutes per video (~1800 steps for a "
                           "2-minute video; the remaining time is shown below). Models are downloaded on first use (needs internet). "
                           "Can be stopped at any time; finished videos are kept. Faster: the 'lightweight' model "
                           "or no skeleton.\n" + dev)
            self.lbl_mode.configure(text=txt)
            busy = self.proc is not None
            self.btn_go.configure(state="normal" if (nv and not busy) else "disabled")

        # ---------------- run
        def _initial_labels(self):
            if not self.var_use_init.get() or not self.var_init.get().strip():
                return None
            p = Path(self.var_init.get().strip())
            if not p.exists():
                messagebox.showerror("Initial labels", f"File not found:\n{p}"); return False
            err = check_initial_labels(p)
            if err:
                messagebox.showerror("Initial labels", f"File cannot be used:\n{p}\n\n{err}"); return False
            return p

        def start(self):
            ds = self.ds
            if ds is None or not ds.videos:
                return
            ds.refresh()
            m = self.mode.get()
            redo = self.var_redo.get()
            miss = missing_packages(NEED_LABELER) + (missing_packages(NEED_YOLOX) if m == "yolox" and (ds.without_npz or redo) else [])
            if miss:
                messagebox.showerror("Packages not installed",
                                     "Packages not installed: " + ", ".join(miss) + "\n\nRun in a terminal:\n" + pip_hint(miss))
                return
            init = self._initial_labels()
            if init is False:
                return
            if m == "yolox" and (ds.without_npz or redo):
                over = (ds.with_npz + ds.bad_npz) if redo else ds.bad_npz
                if over and not messagebox.askyesno(
                        "Recreate npz",
                        f"{len(over)} existing npz will be OVERWRITTEN (recreated every {ds.step} frames):\n"
                        f"{', '.join(over[:10])}" + (" …" if len(over) > 10 else "") + "\n\nContinue?"):
                    return
                self.run_yolox(ds, redo)
            else:
                self.run_labeler(ds, "manual" if m == "manual" else "npz", init)

        def run_yolox(self, ds, redo=False):
            cmd = yolox_cmd(ds, self.var_model.get(), self.var_dev.get(), not self.var_pose.get(), redo)
            self.write("$ " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            except Exception as e:
                self.proc = None
                messagebox.showerror("YOLOX", f"Could not run make_npz.py:\n{e}"); return
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
                if line.startswith("[PROGRESS]"):
                    self.lbl_prog.configure(text=line[len("[PROGRESS]"):].strip())
                else:
                    self.write(line)
            if self.proc is None:
                return
            rc = self.proc.poll()
            if rc is None:
                self.root.after(200, self._poll_yolox); return
            self.root.after(300, lambda: self._yolox_done(rc))

        def _yolox_done(self, rc):
            while not self.lines.empty():                       # remaining output
                self.write(self.lines.get_nowait())
            self.proc = None
            self.btn_stop.configure(state="disabled")
            self.lbl_prog.configure(text="")
            if rc == 0:
                self.var_redo.set(False)                         # do not overwrite again by accident
            ds = self.ds
            ds.refresh()
            self.set_dataset(ds)
            if rc == 0:
                self.write("Creating npz finished.")
                init = self._initial_labels()
                if init is not False and messagebox.askyesno("YOLOX finished", "All npz have been created. Open the labeller now?"):
                    self.run_labeler(ds, "npz", init)
            else:
                self.write(f"Creating npz stopped (code {rc}). Finished videos are kept; "
                           "click Start again to continue.")

        def stop_yolox(self):
            if self.proc is not None and self.proc.poll() is None:
                self.write("Stopping YOLOX… (the video being processed is not saved)")
                self.proc.terminate()

        def run_labeler(self, ds, mode, init):
            cmd = labeler_cmd(ds, mode, init, self.get_label())
            self.write("$ " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
            try:
                p = subprocess.Popen(cmd)
            except Exception as e:
                messagebox.showerror("Labeller", f"Could not open the labeller:\n{e}"); return
            self.root.withdraw()
            self._wait_labeler(p)

        def _wait_labeler(self, p):
            rc = p.poll()
            if rc is None:
                self.root.after(500, lambda: self._wait_labeler(p)); return
            self.root.deiconify()
            self.write(f"Labeller closed (code {rc}). Results in: {self.ds.out_dir}")
            self.set_dataset(self.ds)

        # ---------------- manage files (delete npz / labels, moved to .trash)
        def manage_files(self):
            ds = self.ds
            if ds is None or not ds.videos:
                messagebox.showinfo("Manage files", "Choose a dataset with videos first."); return
            win = tk.Toplevel(self.root)
            win.title(f"Manage files · {ds.name}")
            win.geometry("680x440")
            win.transient(self.root)
            win.grab_set()
            win.columnconfigure(0, weight=1); win.rowconfigure(1, weight=1)
            ttk.Label(win, wraplength=640, justify="left",
                      text="Select one or more videos (Shift/Ctrl/Cmd for several). Deleted files are moved to the "
                           f"dataset's trash folder, so they can be restored:\n{ds.trash_dir}").grid(
                row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 6))
            tv = ttk.Treeview(win, columns=("npz", "labels"), selectmode="extended")
            tv.heading("#0", text="Video"); tv.heading("npz", text="npz"); tv.heading("labels", text="Labelled objects")
            tv.column("#0", width=200); tv.column("npz", width=130); tv.column("labels", width=300)
            tv.grid(row=1, column=0, sticky="nsew", padx=(10, 0))
            sb = ttk.Scrollbar(win, command=tv.yview); sb.grid(row=1, column=1, sticky="ns", padx=(0, 10))
            tv.configure(yscrollcommand=sb.set)

            def fill():
                tv.delete(*tv.get_children())
                ds.refresh()
                for v in ds.videos:
                    z = ds.npz_dir / f"{v}.npz"
                    st = npz_step(z) if z.exists() else None
                    ztxt = (f"every {st} frames" if st else "yes") if z.exists() else "—"
                    labs = labels_in_csv(ds.out_dir / f"{v}.csv")
                    tv.insert("", "end", iid=v, text=v, values=(ztxt, ", ".join(labs) if labs else "—"))

            def delete(kind):
                vids = list(tv.selection())
                if not vids:
                    messagebox.showinfo("Manage files", "Select at least one video first.", parent=win); return
                if kind == "npz":
                    paths, what = [ds.npz_dir / f"{v}.npz" for v in vids], "npz file(s)"
                else:
                    paths, what = [ds.out_dir / f"{v}.csv" for v in vids], "label file(s) (all objects)"
                paths = [x for x in paths if x.exists()]
                if not paths:
                    messagebox.showinfo("Manage files", f"The selected videos have no {what.split(' (')[0]}.", parent=win); return
                if not messagebox.askyesno("Delete", f"Move {len(paths)} {what} to the trash?\n\n"
                                           + ", ".join(x.stem for x in paths[:10]) + (" …" if len(paths) > 10 else ""),
                                           parent=win):
                    return
                for x in paths:
                    move_to_trash(x, ds.trash_dir)
                self.write(f"Moved {len(paths)} {what} to {ds.trash_dir}")
                fill()
                self.set_dataset(ds)

            bb = ttk.Frame(win); bb.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
            ttk.Button(bb, text="Delete npz", command=lambda: delete("npz")).pack(side="left")
            ttk.Button(bb, text="Delete labels", command=lambda: delete("labels")).pack(side="left", padx=6)
            ttk.Button(bb, text="Open trash folder",
                       command=lambda: ds.trash_dir.is_dir() and open_folder(ds.trash_dir)).pack(side="left")
            ttk.Button(bb, text="Close", command=win.destroy).pack(side="right")
            fill()

        def quit(self):
            if self.proc is not None and self.proc.poll() is None:
                if not messagebox.askyesno("Quit", "YOLOX is still running. Stop it and quit?"):
                    return
                self.proc.terminate()
            self.root.destroy()

    if os.name == "nt":                      # Windows: sharp rendering on high-DPI screens
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Choose a dataset and labelling mode, then open the dense labeller.")
    ap.add_argument("--dataset", help="dataset folder; if given, the launcher window is skipped")
    ap.add_argument("--mode", choices=["npz", "manual", "yolox"], default="npz",
                    help="npz = use existing npz; manual = no npz; yolox = create missing npz first")
    ap.add_argument("--model", choices=MODELS, default="balanced", help="for --mode yolox (default balanced)")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="for --mode yolox (default auto: cuda if available, otherwise cpu)")
    ap.add_argument("--no-pose", action="store_true", help="for --mode yolox: skip RTMPose (faster)")
    ap.add_argument("--redo", action="store_true", help="for --mode yolox: recreate existing npz too (overwrite)")
    ap.add_argument("--initial-labels", help="initial-labels CSV (default <dataset>/initial_labels.csv if present)")
    ap.add_argument("--step", type=int, default=2, help="label every N frames (default 2; 1 = every frame)")
    ap.add_argument("--label-name", default=DEFAULT_LABEL,
                    help=f"name of the object being labelled, e.g. patient, examiner, dog (default {DEFAULT_LABEL})")
    args = ap.parse_args()
    if args.dataset:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
