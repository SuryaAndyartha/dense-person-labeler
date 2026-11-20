# datasets/

One folder = one dataset. The contents of this folder are never pushed (see `../.gitignore`).

```
<dataset_name>/
├── videos/              required: video files
├── npz/                 optional: guidance npz files (same name as the video), or created by "Create npz with YOLOX"
├── initial_labels.csv   optional: initial labels (video_id, frame_idx, x1, y1, x2, y2)
└── labels/              created automatically: results, one CSV per video
```

Create a new dataset with the *Dataset baru…* (new dataset) button in `start.py`, or create the folders yourself.
