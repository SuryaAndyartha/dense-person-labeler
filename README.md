# dense-person-labeler

A tool for labelling **one target person** (e.g. a patient) in **every N-th frame** of a video (default: every 2nd frame), for any video dataset. The result is one bounding box per labelled frame, saved as CSV.

Three modes, chosen according to what your dataset already has:

| Your dataset | Mode | What happens |
|---|---|---|
| Already has npz files | **Use existing npz** | YOLOX person boxes are shown; click the target once and the tool follows it automatically |
| No npz, fully manual | **100% manual** | Draw the box yourself; Space copies it to the next steps |
| No npz, want guidance | **Create npz with YOLOX first** | YOLOX (+ RTMPose) runs on your computer, creates the npz files, then the labeller opens as in the first mode |

Runs on **macOS, Windows and Linux**. Tested on macOS with Python 3.12.

Derived from the dense labelling tool of [FreezingOfGait-SelfAnalysis](https://github.com/SuryaAndyartha/FreezingOfGait-SelfAnalysis); the npz format is the same as Stage 2 of that project.

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
   - *New dataset…* creates `datasets/<name>/videos/` and opens it. Put your videos there, then click *Reload*.
   - *Other folder…* picks a folder outside the repository that contains `videos/` or contains the videos directly.
   - *Manage files…* lists the dataset's videos with their npz and labelled objects, and deletes npz or label files of selected videos (section 7).
2. **Choose the frame step** (*Label every N frames*, see section 3). Default: 2.
   **Name the object** (*Object label*, see section 4). Default: `target`.
3. **Choose a labelling mode.** Modes that are not possible are disabled automatically.
4. **Initial labels (optional).** Filled in automatically if `datasets/<name>/initial_labels.csv` exists.
5. Click **Start**. The labeller opens; when you close it, you return to the launcher.

If a package is missing, the program shows the exact `pip install` command to run.

---

## 3. Frame step: label every N frames

*Label every N frames* sets how densely you label:

- **N = 2 (default):** every 2nd frame. A 30 fps video gives ~15 labels per second.
- **N = 1:** every frame. This is the densest possible, equal to the video's own fps. You cannot label more densely than the video's fps (a 30 fps video has only 30 frames per second).
- **N = 3, 5, 10, …:** sparser and faster to label.

The launcher shows the resulting rate for your videos, e.g. `Videos 30 fps → labelling every 2 frames = ~15.0 labels per second`.

**npz and the frame step.** An npz can guide labelling when its frame spacing divides N. An npz made every 2 frames works for N = 2, 4, 6, …, but **not** for N = 1 or 3. In that case:
- *Use existing npz* labels those videos manually and says why.
- *Create npz with YOLOX first* offers to recreate them with the new spacing (after asking you, because the old npz files are overwritten).

If you change N on a video that already has labels, labels at frames outside the new step are kept in the CSV (just not shown).

---

## 4. Object label: what you are labelling

*Object label* is the name of the object you label in this session, e.g. `patient`, `examiner`, `police`, `dog`. It appears on the boxes and messages in the labeller and is saved in the `label` column of the CSV.

To label **several objects in the same videos**, run the labeller once per object with a different name (first `patient`, then `examiner`, …). All objects of a video are stored in the same `labels/<video>.csv`; each session edits only the rows of its own label and keeps the others untouched. Labels of the other objects are shown as thin dashed boxes, each label in its own colour with its name, so you can see what is already labelled (read-only; toggle with **O**).

If `initial_labels.csv` has a `label` column, only the rows with the current object label are used.

---

## 5. How to label

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
| G | Go to an unlabelled step |
| [ ] | Slower / faster |
| H / K / O | Show detection boxes / skeleton / other objects' labels |
| Ctrl/Cmd + Z | Undo |
| S | Save |
| PgUp / PgDn | Previous / next video |

Label colours (on the box and on the timeline): green = clicked detection box, blue = following detection boxes, dark purple = drawn box, light purple = copying the drawn box, yellow = initial label, grey = not visible.

**Auto-save** every 20 seconds, when switching video, and when the labeller closes.

---

## 6. Mode "Create npz with YOLOX first"

### Device

The *Device* list shows only what is **available on this computer**:

| Device | Meaning | Shown when |
|---|---|---|
| `cuda` | NVIDIA graphics card. Fastest | `onnxruntime-gpu` is installed and an NVIDIA GPU is found |
| `mps` | Apple accelerator via CoreML | A Mac whose onnxruntime supports CoreML. Not always faster; falls back to `cpu` automatically if it fails |
| `cpu` | Regular processor. Slowest | Always |

The *Progress* box shows the device actually used (e.g. `CPUExecutionProvider`).

### Time and speed-ups

- A 2-minute 30 fps video at N = 2 is ~1,800 steps. The remaining time is estimated while it runs. Try one video first.
- The skeleton (RTMPose) is computed only for boxes with score ≥ 0.3, i.e. the boxes the labeller actually shows.
- Faster still: untick *Include skeleton*, choose the `lightweight` model, or use a larger N.
- Defaults: `balanced` model, score ≥ 0.10, at most 30 boxes per frame, every N frames.

### Other notes

- Only videos **without** a matching npz are processed. To redo videos that already have a matching npz (e.g. with another model or with the skeleton), tick **Recreate existing npz too (overwrite)**; all videos are then processed again after a confirmation. To redo only some videos, delete their npz in *Manage files…* first.
- On **first use** the models are downloaded automatically (needs internet access to `download.openmmlab.com` or `huggingface.co`) and cached in `~/.cache/rtmlib`.
- **Can be stopped** with *Stop YOLOX*. Finished videos are kept; click *Start* again to continue.
- If *Use existing npz* is chosen while some videos have no matching npz, those videos are labelled manually.

---

## 7. Deleting labels and npz

Nothing is deleted permanently: files are moved (or, for a cleared label, the CSV is first copied) to the dataset's trash folder `datasets/<name>/.trash/`, with a timestamp in the name. To restore a file, move it back and remove the timestamp from its name.

- **In the labeller:** *Clear all '<label>' labels in this video…* removes every label of the current object in the open video. Labels of other objects in the same CSV are kept; a backup of the CSV is copied to the trash first.
- **In the launcher:** *Manage files…* → select videos → *Delete npz* or *Delete labels*. *Delete labels* moves the whole CSV of those videos (all objects).

---

## 8. Folder structure

```
dense-person-labeler/
├── start.py                  ← main program (launcher)
├── labeler/
│   ├── dense_labeler.py      ← the labelling tool
│   └── make_npz.py           ← YOLOX + RTMPose → npz
├── datasets/
│   └── README.md
├── requirements.txt          ← version ranges (default)
├── requirements-tested.txt   ← exact tested versions
├── LICENSE
├── .gitignore                ← contents of datasets/ are never pushed
└── README.md
```

One dataset = one folder in `datasets/`:

```
datasets/<dataset_name>/
├── videos/              REQUIRED, you fill it: videos (.mp4/.avi/.mov/.mkv)
├── npz/                 created by "Create npz", or copied by you (same name as the video)
├── initial_labels.csv   OPTIONAL: initial labels (video_id, frame_idx, x1, y1, x2, y2)
├── labels/              created automatically: results, one CSV per video
└── .trash/              created when something is deleted: deleted / backed-up files
```

Output: `labels/<video>.csv` with columns `video_id,frame_idx,t_sec,status,x1,y1,x2,y2,source,label`, grouped by label, then frame (so `frame_idx` restarts for each label). For analysis, filter by `label` or sort by `label, frame_idx`.
- `status`: `visible` or `absent`.
- `source`: `click` (clicked detection box), `drawn` (drawn box), `auto` (following detection boxes), `follow_drawn` (copy of your drawn box), `initial` (initial label).
- `label`: the object label (section 4).

## 9. Without the launcher window

```bash
python start.py --dataset datasets/example_dataset --mode npz      [--step 2] [--label-name patient]
python start.py --dataset datasets/example_dataset --mode manual   [--step 2]
python start.py --dataset datasets/example_dataset --mode yolox    [--step 2] [--no-pose] [--redo] [--device auto|cpu|mps|cuda]

python labeler/make_npz.py --videos datasets/example_dataset/videos --out datasets/example_dataset/npz --stride 2
python labeler/dense_labeler.py --videos datasets/example_dataset/videos --npz datasets/example_dataset/npz --out datasets/example_dataset/labels --step 2 --label-name patient
python labeler/dense_labeler.py --out datasets/example_dataset/labels --merge      # merge all CSVs -> all_labels.csv
```

---

## 10. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `Tkinter is not available` | Python without Tk | See the table in section 1 (Homebrew / Linux) |
| `Packages not installed: …` | `pip install` not run, or `.venv` not activated | Activate `.venv`, then run the command shown |
| `CERTIFICATE_VERIFY_FAILED` | Python has no SSL certificates | The program already retries with `certifi`. If it still fails on macOS, run `Install Certificates.command` |
| `The models could not be downloaded` | No internet / sites blocked | Try another network. Manual and existing-npz modes still work |
| `pip install onnxruntime` fails | Python too new, or an older computer no longer supported by the latest release | Use Python 3.12; if it still fails, install an older onnxruntime |
| `device cuda is not available` | Only plain `onnxruntime` is installed | Install `onnxruntime-gpu` (section 1) |
| Video does not open / black frames | Codec not supported by OpenCV | Convert to MP4 H.264, e.g. `ffmpeg -i in.mov -c:v libx264 out.mp4` |

---

## 11. npz format

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

## 12. Testing (for developers)

1. **Existing npz:** a dataset with videos + their npz (+ optional `initial_labels.csv`). Expected: detection boxes and yellow initial labels appear; Space follows the boxes.
2. **100% manual:** a dataset with videos only. Expected: no detection boxes; a drawn box is copied (purple) on Space.
3. **Create npz:** a dataset with one video. Expected: the npz is created, then the labeller opens with detection boxes.
4. **Stop and resume:** with 2 videos, stop during the second, then *Start* again. Expected: the first video is skipped.
5. **Object labels:** label a video as `patient`, then again as `examiner`. Expected: one CSV with both labels; in the second session the `patient` boxes appear as thin dashed boxes.
6. **Frame step:** with an npz made at N = 2, choose N = 4 (npz still used) and N = 1 or 3 (labelled manually, or recreated by *Create npz*).
7. **Recreate npz:** with all videos having a matching npz, tick *Recreate existing npz too*. Expected: the YOLOX mode becomes available, a confirmation lists the npz to overwrite, then all are recreated.
8. **Delete:** *Manage files…* → delete the npz of one video (it then counts as "without npz") and the labels of another; *Clear all … labels* in the labeller. Expected: files appear in `.trash/`, other objects' labels are kept.
9. **Fresh user:** new `.venv` + temporarily move `~/.cache/rtmlib`, then follow sections 1–2 from scratch.
