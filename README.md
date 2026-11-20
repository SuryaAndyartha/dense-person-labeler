# dense-person-labeler

A tool for labelling **one target person** (e.g. a patient) in **every N-th frame** of a video (default: every 2nd frame), for any video dataset. The result is one bounding box per labelled frame, saved as CSV.

Three ways to work, chosen according to what your dataset already has:

| Your dataset | Mode | What happens |
|---|---|---|
| Already has npz files | **Use existing npz** | YOLOX person boxes are shown; click the target once and the tool follows it automatically |
| No npz, fully manual | **100% manual** | Draw the box yourself; Space copies it to the next frames |
| No npz, want guidance | **Create npz with YOLOX first** | YOLOX (+ RTMPose) runs on your computer, creates the npz files, then the labeller opens as in the first mode |

Runs on **macOS, Windows and Linux**. Tested on macOS with Python 3.12.

Derived from the dense labelling tool of [FreezingOfGait-SelfAnalysis](https://github.com/SuryaAndyartha/FreezingOfGait-SelfAnalysis); the npz format is the same as Stage 2 of that project.

> The program's interface is in Indonesian. Button names are quoted below with an English translation.

---

## 1. Setup (once)

Requires **Python 3.9–3.12** (3.12 recommended). Run all commands from the repository folder. A separate Python environment (`.venv`) is recommended.

**macOS / Linux**
```bash
git clone https://github.com/SuryaAndyartha/dense-person-labeler.git
cd dense-person-labeler
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (Command Prompt or PowerShell)**
```bat
git clone https://github.com/SuryaAndyartha/dense-person-labeler.git
cd dense-person-labeler
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` uses version ranges so it installs on many machines (rtmlib is pinned to 0.0.16). The exact combination that was tested (macOS, Python 3.12) is in `requirements-tested.txt`; use `pip install -r requirements-tested.txt` if the latest versions cause problems.

Install **only one** OpenCV package (`opencv-python`). If `opencv-contrib-python` is also installed in the same Python, both fight over the `cv2` module; this is one reason to use a `.venv`.

Extra steps per system:

| System | Install separately |
|---|---|
| macOS, Python from python.org | Nothing. If model downloads still fail with a certificate error: `open "/Applications/Python 3.12/Install Certificates.command"` (adjust the version) |
| macOS, Python from Homebrew | `brew install python-tk@3.12` (adjust the version) |
| Ubuntu / Debian | `sudo apt install python3-tk python3-venv` |
| Windows | Nothing (Tkinter ships with the official Python). When installing Python, tick "Add python.exe to PATH" |
| NVIDIA GPU, want `cuda` | Replace `onnxruntime` with `onnxruntime-gpu`: `pip uninstall -y onnxruntime && pip install onnxruntime-gpu` (needs a matching NVIDIA driver + CUDA) |

## 2. Running

```bash
# macOS / Linux                          # Windows
source .venv/bin/activate                # .venv\Scripts\activate
python start.py                          # python start.py
```

The launcher window opens:

1. **Choose a dataset.** The list shows every folder inside `datasets/`.
   - *Dataset baru…* (new dataset) creates `datasets/<name>/videos/` and opens it. Put your videos there, then click *Muat ulang* (reload).
   - *Folder lain…* (other folder) picks a folder outside the repository that contains `videos/` or contains the videos directly.
2. **Choose the frame step** (*Label setiap N frame*, see section 3). Default: 2.
3. **Choose the mode.** Modes that are not possible are disabled automatically.
4. **Initial labels (optional)** (*Label awal*). Filled in automatically if `datasets/<name>/initial_labels.csv` exists.
5. Click **Mulai** (start). The labeller opens; when you close it, you return to the launcher.

If a package is missing, the program shows the exact `pip install` command to run.

---

## 3. Frame step: label every N frames

*Label setiap N frame* sets how densely you label:

- **N = 2 (default):** every 2nd frame. A 30 fps video gives ~15 labels per second.
- **N = 1:** every frame. This is the densest possible, equal to the video's own fps. You cannot label more densely than the video's fps (a 30 fps video has only 30 frames per second).
- **N = 3, 5, 10, …:** sparser and faster to label.

The launcher shows the resulting rate for your videos, e.g. `Video 30 fps → label setiap 2 frame = ~15.0 label per detik` (labels per second).

**npz and the frame step.** An npz can guide labelling when its frame spacing divides N. An npz made every 2 frames works for N = 2, 4, 6, …, but **not** for N = 1 or 3. In that case:
- *Use existing npz* labels those videos manually and says why.
- *Create npz with YOLOX* offers to recreate them with the new spacing (after asking you, because the old npz files are overwritten).

If you change N on a video that already has labels, labels at frames outside the new step are kept in the CSV (just not shown).

---

## 4. How to label

1. **Click** the target's detection box once, or **drag** to draw your own box.
2. Press **Space**: the video plays and the tool fills in the next steps while you watch.
   - Last label came from **clicking a detection box** → the next steps follow the detection boxes (blue).
   - Last label came from **drawing** → the next steps copy your drawn box (purple).
   - The tool **stops by itself** whenever it is unsure (no clear next box, or the box does not match an initial label).
3. If a box is wrong: Space (pause), go back with ←, click or draw the correct box, Space again. Automatic labels after it are recomputed.

| Key | Action |
|---|---|
| Click a box | Select a detection box (click again at the same spot = next overlapping box) |
| Drag | Draw your own box |
| Space | Play / pause (auto-fill) |
| → / D, ← / A | Next / previous step |
| Shift + ← → | Jump ~1 second |
| F | Lock this box: the next steps copy this box |
| Y | Follow the detection boxes again from this step |
| N | Target not visible |
| Backspace | Delete the label of this step |
| G | Go to the first unlabelled step |
| [ ] | Slower / faster |
| H / K | Show detection boxes / skeleton |
| Ctrl/Cmd + Z | Undo |
| S | Save |
| PgUp / PgDn | Previous / next video |

Label colours (on the box and on the timeline): green = clicked detection box, blue = follows detection boxes, dark purple = drawn box, light purple = copies your drawn box, yellow = initial label, grey = not visible.

**Auto-save** every 20 seconds, when switching video, and when the labeller closes.

---

## 5. Mode "Create npz with YOLOX"

### Device

The *Perangkat* (device) list shows only what is **available on this computer**:

| Device | Meaning | Shown when |
|---|---|---|
| `cuda` | NVIDIA graphics card. Fastest | `onnxruntime-gpu` is installed and an NVIDIA GPU is found |
| `mps` | Apple accelerator via CoreML | A Mac whose onnxruntime supports CoreML. Not always faster; falls back to `cpu` automatically if it fails |
| `cpu` | Regular processor. Slowest | Always |

The *Proses* (progress) box shows the device actually used (e.g. `CPUExecutionProvider`).

### Time and speed-ups

- A 2-minute 30 fps video at N = 2 is ~1,800 steps. The remaining time is estimated while it runs. Try one video first.
- The skeleton (RTMPose) is computed only for boxes with score ≥ 0.3, i.e. the boxes the labeller actually shows.
- Faster still: untick *Sertakan kerangka* (include skeleton), choose the `lightweight` model, or use a larger N.
- Defaults: `balanced` model, score ≥ 0.10, at most 30 boxes per frame, every N frames.

### Other notes

- Only videos **without** a matching npz are processed.
- On **first use** the models are downloaded automatically (needs internet access to `download.openmmlab.com` or `huggingface.co`) and cached in `~/.cache/rtmlib`.
- **Can be stopped** (*Hentikan YOLOX*). Finished videos are kept; click *Mulai* again to continue.
- If *Use existing npz* is chosen while some videos have no matching npz, those videos are labelled manually.

---

## 6. Folder structure

```
dense-person-labeler/
├── start.py                  ← main program
├── labeler/
│   ├── dense_labeler.py      ← the labelling tool
│   └── make_npz.py           ← YOLOX + RTMPose → npz
├── datasets/
│   └── README.md
├── requirements.txt          ← version ranges (default)
├── requirements-tested.txt   ← exact tested versions
├── .gitignore                ← contents of datasets/ are never pushed
└── README.md
```

One dataset = one folder in `datasets/`:

```
datasets/<dataset_name>/
├── videos/              REQUIRED, you fill it: videos (.mp4/.avi/.mov/.mkv)
├── npz/                 created by "Create npz", or copied by you (same name as the video)
├── initial_labels.csv   OPTIONAL: initial labels (video_id, frame_idx, x1, y1, x2, y2)
└── labels/              created automatically: results, one CSV per video
```

Output: `labels/<video>.csv` with columns `video_id,frame_idx,t_sec,status,x1,y1,x2,y2,sumber`.
- `status`: `terlihat` (visible) or `tidak_ada` (not visible).
- `sumber` (source): `manual` (clicked detection box), `gambar` (drawn box), `otomatis` (followed detection box), `ikut_gambar` (copy of your drawn box), `label_awal` (initial label).

## 7. Without the launcher window

```bash
python start.py --dataset datasets/example_dataset --cara npz      [--step 2]
python start.py --dataset datasets/example_dataset --cara manual   [--step 2]
python start.py --dataset datasets/example_dataset --cara yolox    [--step 2] [--tanpa-pose] [--device auto|cpu|mps|cuda]

python labeler/make_npz.py --videos datasets/example_dataset/videos --out datasets/example_dataset/npz --stride 2
python labeler/dense_labeler.py --videos datasets/example_dataset/videos --npz datasets/example_dataset/npz --out datasets/example_dataset/labels --step 2
python labeler/dense_labeler.py --out datasets/example_dataset/labels --gabung      # merge all CSVs -> all_labels.csv
```

(`--cara` = mode, `--tanpa-pose` = without skeleton, `--gabung` = merge.)

---

## 8. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `Tkinter tidak tersedia` | Python without Tk | See the table in section 1 (Homebrew / Linux) |
| `Paket belum terpasang: …` (package not installed) | `pip install` not run, or `.venv` not activated | Activate `.venv`, then run the command shown |
| `CERTIFICATE_VERIFY_FAILED` | Python has no SSL certificates | The program already retries with `certifi`. If it still fails on macOS, run `Install Certificates.command` |
| `Model tidak bisa diunduh` (model cannot be downloaded) | No internet / sites blocked | Try another network. Manual and existing-npz modes still work |
| `pip install onnxruntime` fails | Python too new, or an older computer no longer supported by the latest release | Use Python 3.12; if it still fails, install an older onnxruntime |
| `perangkat cuda tidak tersedia` (cuda not available) | Only plain `onnxruntime` is installed | Install `onnxruntime-gpu` (section 1) |
| Video does not open / black frames | Codec not supported by OpenCV | Convert to MP4 H.264, e.g. `ffmpeg -i in.mov -c:v libx264 out.mp4` |

---

## 9. npz format

One file per video, with the same name as the video:

| Array | Shape | Content |
|---|---|---|
| `det_frame` | (K,) | frame number of each box |
| `det_boxes` | (K, 4) | box x1, y1, x2, y2 (pixels) |
| `det_scores` | (K,) | YOLOX score |
| `det_kpts` | (K, 26, 2) | 26 Halpe-26 body keypoints (NaN if not computed) |
| `det_kpt_scores` | (K, 26) | keypoint scores (0 if not computed) |
| `frames_processed` | (T,) | every processed frame (0, N, 2N, …) |
| `meta` | JSON | fps, video size, models, settings |

Frames are read sequentially (`grab/retrieve`), so frame numbers match the labeller. npz files from another pipeline can be used if they follow this format; the minimum is `det_frame`, `det_boxes`, `det_scores` and `frames_processed`.

---

## 10. Testing (for developers)

1. **Existing npz:** a dataset with videos + their npz (+ optional `initial_labels.csv`). Expected: detection boxes and yellow initial labels appear; Space follows the boxes.
2. **100% manual:** a dataset with videos only. Expected: no detection boxes; a drawn box is copied (purple) on Space.
3. **Create npz:** a dataset with one video. Expected: the npz is created, then the labeller opens with detection boxes.
4. **Stop and resume:** with 2 videos, stop during the second, then *Mulai* again. Expected: the first video is skipped.
5. **Frame step:** with an npz made at N = 2, choose N = 4 (npz still used) and N = 1 or 3 (labelled manually, or recreated by *Create npz*).
6. **Fresh user:** new `.venv` + temporarily move `~/.cache/rtmlib`, then follow sections 1–2 from scratch.
