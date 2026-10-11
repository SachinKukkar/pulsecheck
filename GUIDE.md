# Building PulseCheck, step by step

This guide walks through the whole project in the order you'd build it: what each
step does, the command that runs it, what you should see, and where it lives in
the code. Steps 1-6 follow the article "Deepfake detection from your pulse"
(find the face, average patches, filter, read the beats, check against a real
pulse, compare patches, train, split by person). Steps 7-10 take it to real data,
a GPU model, a website and a deployment.

> **Ground rule.** Every result this project reports comes from real datasets.
> Synthetic clips with a painted-on pulse appear in exactly one place, the test
> suite, where they check the code against a known answer. They never reach
> `reports/`, `models/`, the website or the Word report.

| Step | What | Time on a laptop |
|---|---|---|
| 0 | Setup and dataset requests | 30 min (+ days of waiting for approvals) |
| 1-4 | The pipeline on one video | 1 h |
| 5 | Heart rate vs a real pulse (UBFC-rPPG) | download + ~15 min |
| 6 | Compression study | ~45 min |
| 7 | Extract FaceForensics++ and Celeb-DF | 2-3 h, unattended, resumable |
| 8 | Train and test the classical detector | ~20-40 min (the SVM is the slow one) |
| 9 | Train the CNN on the GPU | ~5-15 min |
| 10 | Website, Word report, deployment | 1 h |

---

## 0. Setup

### 0.1 Request the datasets (do this first)

Approval takes days. Everything is in [DATASETS.md](DATASETS.md): links, sizes,
the exact download commands and the folder layout. In short:

- **UBFC-rPPG**: direct Google Drive download, ~60 GB, into `C:\datasets\UBFC-rPPG`
- **FaceForensics++**: form, then a script, ~12 GB, into `C:\datasets\FaceForensics`
- **Celeb-DF v2**: form, then a Drive link, ~10 GB, into `C:\datasets\Celeb-DF-v2`

Check each download before processing it:

```bash
python -m pulsecheck check-data ubfc    --root C:\datasets\UBFC-rPPG
python -m pulsecheck check-data ffpp    --root C:\datasets\FaceForensics
python -m pulsecheck check-data celebdf --root C:\datasets\Celeb-DF-v2
```

`extract` refuses to run on an incomplete download unless you pass `--allow-partial`.

### 0.2 Python environment

```bash
cd pulsecheck
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
python -m pulsecheck download-model
python -m pytest
```

**Checkpoint:** every test passes. The PyTorch line installs the CUDA 12.6 build, which runs
on any NVIDIA driver whose `nvidia-smi` reports CUDA 12.6 or newer (the RTX 3050 laptop used
here reports 12.7). Check the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Windows Smart App Control.** If an import fails with *"DLL load failed ... An Application
> Control policy has blocked this file"*, Windows is checking a newly installed compiled
> file's reputation. Re-running usually works a minute later. If it keeps failing, run that
> step inside Docker instead. Don't turn Smart App Control off just for this: it can't be
> turned back on without reinstalling Windows.

---

## 1. Read the video and find the face

**Code:** `pulsecheck/video.py`, `pulsecheck/face.py`

OpenCV decodes frames. Every beats-a-minute number depends on the frame clock, and phone
or browser recordings are often variable-frame-rate, so for every frame we keep both the
decoder's timestamp and index / FPS, and use whichever is trustworthy (`choose_timestamps`).
MediaPipe Face Landmarker then places 478 points on the face, in VIDEO mode so it tracks
the face between frames instead of re-detecting it each time.

```bash
python -m pulsecheck landmarks some_photo_of_you.jpg --out landmarks.png
```

**Checkpoint:** `landmarks.png` shows every landmark number sitting on your face, and
`landmarks_patches.png` shows the six skin patches.

## 2. Average skin patches, frame by frame

**Code:** `pulsecheck/roi.py`, `pulsecheck/pipeline.py`, `pulsecheck/traces.py`

- **Six named patches**, polygons through fixed landmark numbers: forehead left/right,
  cheek left/right, nose bridge, chin. They were chosen by plotting the landmark numbers,
  exactly as in step 1, and shrunk slightly so their edges avoid brows and hairline.
- **A 32-patch grid** of small squares for the CNN's "PPG map".
- **A skin-colour model per video**: a Gaussian over chrominance (Cr, Cb) learned from the
  cheeks in the first ten frames. Pixels too far from it (hair, brows, glasses frames) are
  dropped. It is learned from each video's own pixels rather than fixed thresholds, so it
  doesn't favour one skin tone.

Each frame becomes the mean R, G, B of every patch. The series is resampled onto an exact
30 Hz clock and cached as a `Traces` file (`.npz`), because landmarking is the only slow
step; everything after it re-reads the cache.

## 3. Clean the signal

**Code:** `pulsecheck/rppg.py`, `pulsecheck/signals.py`

- **Green** (Verkruysse 2008): the green channel, inverted.
- **CHROM** (de Haan 2013) and **POS** (Wang 2017): combine all three colours so that
  brightness changes (lighting, movement) cancel. POS is the default.
- **Detrend** removes slow drift; a zero-phase Butterworth **band-pass** keeps 0.7-3 Hz
  (42-180 bpm).

The tests show why CHROM/POS exist: with a lamp flickering at 1.8 Hz twice as strongly as
the pulse, POS and CHROM still find 72 bpm, while Green reports the flicker's 108 bpm.

## 4. Read the beats

The Fourier transform splits the pulse into frequencies; the strongest one inside the band,
times 60, is the heart rate. A 10 s clip at 30 fps is 300 numbers, and its frequency steps
are 1/10 s = 0.1 Hz = 6 bpm apart. 72 bpm is 1.2 Hz, step 12. The code zero-pads the
spectrum and fits a parabola to the peak to locate it between steps. SNR (de Haan 2013)
measures how much of the band's power sits on the pulse.

```bash
python -m pulsecheck analyze my_webcam_clip.mp4 --json result.json
```

Record 20-30 s of yourself in good, even light, sitting still, and count your pulse at the
wrist for the same 30 s. **Checkpoint:** within ~5 bpm. The website's How it works page has
a live lab for this maths.

## 5. Check it against a real pulse (UBFC-rPPG)

**Code:** `pulsecheck/datasets.py` (`ubfc`, `load_ubfc_gt`), `pulsecheck/hr_eval.py`

UBFC-rPPG records a finger pulse oximeter alongside each video. Following the dataset
authors, ground-truth heart rate is read from the oximeter's waveform with the same spectral
method as the video, so both sides get the same frequency resolution.

```bash
python -m pulsecheck check-data ubfc --root C:\datasets\UBFC-rPPG
python -m pulsecheck extract ubfc --root C:\datasets\UBFC-rPPG --max-seconds 0
python -m pulsecheck eval-hr
```

`--max-seconds 0` processes whole videos (~1 min each). Output: `reports/hr_validation.json`.
It reports MAE, RMSE, Pearson r, Bland-Altman bias and limits, and the share within 5 bpm,
for each method and for 5, 10, 20 and 30 s windows, plus per-person error and error by
estimated skin tone (ITA). Published classical methods on UBFC-rPPG are typically a few bpm
MAE for POS/CHROM and clearly worse for Green; your numbers are whatever the data says.

## 6. Compression (a pitfall from the article)

**Code:** `pulsecheck/studies.py`

```bash
python -m pulsecheck compression-study --root C:\datasets\UBFC-rPPG --limit 20
```

Re-encodes UBFC's uncompressed videos with H.264 at CRF 18, 23, 28, 33 and 38 and measures
heart-rate error at each level. Output: `reports/compression_study.json`.

## 7. Extract the deepfake datasets

**Code:** `pulsecheck/datasets.py` (`ffpp`, `celebdf`), `pulsecheck/extract.py`

```bash
python -m pulsecheck extract ffpp --root C:\datasets\FaceForensics --compression c23
python -m pulsecheck extract celebdf --root C:\datasets\Celeb-DF-v2
```

- FaceForensics++: 6,000 videos (1,000 originals + 5 x 1,000 fakes), first 20 s each,
  4 parallel workers. **Resumable**: stop it any time with Ctrl+C, run the same command
  again, and it skips what's done. Failures are logged and skipped, not fatal.
- Want to see the whole flow first? Add `--limit 50` (50 videos of each type, ~10 min).
- Celeb-DF: only the 518 official test videos. It's used purely to test on a dataset the
  detector never saw.

**Splitting by person (step 7 of the article).** FaceForensics++ ships official
train/val/test lists of source-target pairs (`data/splits/ffpp`). Both videos of a pair,
and every fake made from them, share one group, so no person lands on both sides.

## 8. Train and test the classical detector

**Code:** `pulsecheck/features.py`, `pulsecheck/detector.py`, `pulsecheck/detect_eval.py`

Every 8 s window (4 s hop) becomes 91 numbers describing how well the six patches agree:
heart-band coherence (weighted by the patches' own power), correlation and lag for all 15
pairs; per-patch SNR, rate disagreement and agreement with the others; and the 4-12 Hz
"noise" band FakeCatcher found useful.

```bash
python -m pulsecheck train --cross Celeb-DF=cache/celebdf --real UBFC=cache/ubfc
```

What happens, all printed as it runs:

1. Logistic regression, SVM, gradient boosting and random forest are compared with 5-fold
   **GroupKFold** inside the training split; the best video-level AUC wins.
2. The same model is re-scored with a *random* split of windows, to show how much leakage
   inflates the score (the article's pitfall).
3. The decision threshold is chosen on **val** and frozen.
4. Everything is measured on **test** (video level and window level, ROC/PR, confusion
   matrix, each manipulation method separately), then on Celeb-DF (never seen) and on
   UBFC's genuine videos (where any "fake" is a false alarm).
5. Permutation importance shows which features the model relies on.

Outputs: `models/detector.joblib` + `models/detector.json`, `reports/detector.json`.

The SVM with calibrated probabilities is by far the slowest candidate. For a quick run, leave it
out with `--models logreg hist_gb random_forest`.

**What to expect.** Well below FakeCatcher's reported 96% (that was one older test set; the
same paper reports ~91% on a harder one, and good modern fakes can copy the source's pulse).
Above chance on people the model never saw is a real result for a project like this. The
cross-dataset number is the honest one to quote.

## 9. Train the CNN on the GPU

**Code:** `pulsecheck/deep.py`

For each window, the 32 grid patches' POS pulses form a 2 x 32 x 240 image (heart band and
wide band), the "PPG map". A small CNN (~80k parameters) learns real vs fake from it on the
RTX 3050: AdamW, cosine learning rate, class-weighted loss, mixed precision, random time
shifts, early stopping on validation AUC. Same splits, same threshold rule, same reports.

```bash
python -m pulsecheck train-cnn --cross Celeb-DF=cache/celebdf --real UBFC=cache/ubfc
```

Outputs: `models/cnn.pt` + `models/cnn.json`, `reports/cnn.json`. The website deploys
whichever detector has the higher **validation** AUC. The test set is never used to choose.

## 10. Website, report, deployment

### 10.1 Run the site locally

```bash
python -m pulsecheck serve
```

Open <http://127.0.0.1:7860>. **Analyze** takes an upload or a 15 s webcam recording and
shows every step: the patches on your face, the raw → filtered → POS pulse, the spectrum
with its FFT steps, heart rate over time, each patch's pulse, the coherence matrix, and the
detector's window votes. **Results** shows every chart from `reports/*.json`. Each chart has
a Table button showing the same numbers.

### 10.2 The Word report

```bash
python -m pulsecheck report-doc --author "Your Name"
```

Writes `docs/PulseCheck_Technical_Report.docx`, built from the code and `reports/*.json`.
Re-run it after every training run. Open it in Word and accept "update fields" so the table
of contents fills in.

### 10.3 Test the container

Start Docker Desktop, then:

```bash
docker build -t pulsecheck .
docker run --rm -p 7860:7860 pulsecheck
```

### 10.4 Deploy to Hugging Face Spaces

1. Create a free account at <https://huggingface.co>, then **New Space**: name `pulsecheck`,
   SDK **Docker**, template **Blank**, hardware **CPU basic** (free).
2. Create a **write** token at <https://huggingface.co/settings/tokens>, then log in. You
   paste the token yourself; never commit it.

   ```bash
   pip install huggingface_hub
   hf auth login
   ```

3. Upload the project, but not the environment, caches, datasets, tests or docs:

   ```bash
   hf upload <your-username>/pulsecheck . . --repo-type space --exclude ".venv/*" "cache/*" ".scratch/*" "data/*" "tests/*" "docs/*" ".github/*" "*.pyc"
   ```

4. The Space builds the Docker image (5-10 minutes the first time; watch the **Logs** tab),
   then serves the site at `https://<your-username>-pulsecheck.hf.space`.
5. After retraining, run the same `hf upload` again: it sends only what changed
   (`models/`, `reports/`).

Notes: a free Space sleeps after 48 h without visitors and wakes on the next visit (~1 min).
The webcam needs HTTPS, which Spaces provide; if the camera is blocked inside the
huggingface.co page frame, open the direct `*.hf.space` URL.

### 10.5 GitHub

The repository is already initialised. After a training run, commit what changed and push:

```bash
python -m pulsecheck update-readme
python -m pulsecheck report-doc --author "Your Name"
git add models reports README.md docs
git commit -m "Results: <what you ran>"
git push
```

`.gitignore` keeps out the venv, caches, datasets and machine-specific settings; the trained
model files are small, so they're committed directly (no Git LFS needed). `.github/workflows/tests.yml`
runs the test suite on every push.

---

## Where people go wrong (and what this project does about it)

| Pitfall | What PulseCheck does |
|---|---|
| Compressed or moving video | measures error vs compression; warns about motion and weak SNR |
| Clips too short | reports error at 5-30 s windows; the site shows the FFT step size |
| Testing on people you trained on | official pair-disjoint splits + GroupKFold; reports the leaky score next to the honest one |
| Skin tone | per-video skin model; error by ITA; says when the data is too narrow to judge |
| Quoting "96%" | reports test, cross-dataset and false-alarm numbers, each with what it means |
| Good fakes have a heart (2025) | presents the verdict as a screening signal, never proof |

## Troubleshooting

| Problem | Fix |
|---|---|
| `No usable face` | face the camera; even, bright light; face at least ~1/4 of the frame |
| `DLL load failed ... Application Control` | Smart App Control is checking a new file; re-run, or use Docker |
| `check-data` says PARTIAL | a Drive folder arrives as several zips; download and extract all of them |
| Drive: "too many users have downloaded" | wait a few hours, or `python -m pulsecheck download-ubfc` (resumes, skips done files) |
| Extraction is slow or memory-hungry | `--workers 2`; it's resumable, so stopping loses nothing |
| `torch.cuda.is_available()` is False | install the cu126 build above; update the NVIDIA driver if needed |
| HF build fails at a package | check the Logs tab; the versions in `requirements.txt` are the tested ones |
