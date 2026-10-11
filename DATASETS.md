# Getting the datasets

Three real datasets, all free for research. None of them may be redistributed,
so they stay outside this repository, in `C:\datasets\`. Total: about 85 GB.

| Dataset | Used for | Access | Size |
|---|---|---|---|
| UBFC-rPPG (DATASET_2) | heart-rate accuracy vs a finger pulse oximeter; false alarms on genuine videos | direct download, no form | ~55-65 GB |
| FaceForensics++ (c23) | training and testing the deepfake detector | Google Form, then a download script by email | ~12 GB |
| Celeb-DF v2 | cross-dataset test (never trained on) | Google Form, then a Drive link by email | ~10-12 GB |

Start the two forms first: approval takes days, and UBFC can download meanwhile.

---

## 1. UBFC-rPPG: download now

1. Sign in to your Google account in your browser. Signed-in downloads avoid most
   "quota exceeded" errors that anonymous downloads hit.
2. Open the authors' folder:
   <https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk>
   (linked from the official page <https://sites.google.com/view/ybenezeth/ubfcrppg>).
3. Read `Agreement.xlsx` and `readme.txt` first. By downloading you accept them. One rule to
   remember: **facial images of DATASET_2 subjects 21 and 27 must not appear in any
   publication, presentation or report.** This project never shows dataset frames.
4. Right-click **DATASET_2** -> **Download**. Drive zips it into many parts
   (`DATASET_2-...-001.zip`, `-002.zip`, ...). Download every part. `DATASET_1` is optional; skip it.
5. Extract every part into `C:\datasets\UBFC-rPPG\` so that you end up with:

   ```
   C:\datasets\UBFC-rPPG\DATASET_2\subject1\vid.avi
   C:\datasets\UBFC-rPPG\DATASET_2\subject1\ground_truth.txt
   C:\datasets\UBFC-rPPG\DATASET_2\subject3\...      (42 subject folders, numbers have gaps)
   ```

If Drive says *"Too many users have viewed or downloaded this file recently"*, wait a few
hours and retry those folders. Alternatively, `python -m pulsecheck download-ubfc --root C:\datasets\UBFC-rPPG`
downloads file by file, resumes, skips what you already have, and lists any file still blocked.

Check: `C:\datasets\UBFC-rPPG\DATASET_2` has 42 folders, each with `vid.avi` (~1.3 GB) and `ground_truth.txt` (~70 KB).

---

## 2. FaceForensics++: request access now

1. Read the repository page: <https://github.com/ondyari/FaceForensics>.
2. Fill in the request form linked there:
   <https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform>.
   Use an institutional (university) email address if you have one; those are approved more reliably.
   Purpose: a non-commercial student research project on remote photoplethysmography for deepfake detection.
3. You will get a link to `download-FaceForensics.py` by email (usually within a week; if not,
   check your spam folder and that the email didn't bounce). Save it **outside** this repository,
   e.g. `C:\datasets\download-FaceForensics.py`.
4. Download the c23 videos: the originals and the five manipulation methods, **not** `all`
   (`all` also pulls the 22 GB DeepFakeDetection set, which this project doesn't use).
   The script shows its terms of use first; you accept them yourself.

   ```bash
   cd C:\datasets
   python download-FaceForensics.py C:\datasets\FaceForensics -d original -c c23 -t videos
   python download-FaceForensics.py C:\datasets\FaceForensics -d Deepfakes -c c23 -t videos
   python download-FaceForensics.py C:\datasets\FaceForensics -d Face2Face -c c23 -t videos
   python download-FaceForensics.py C:\datasets\FaceForensics -d FaceSwap -c c23 -t videos
   python download-FaceForensics.py C:\datasets\FaceForensics -d NeuralTextures -c c23 -t videos
   python download-FaceForensics.py C:\datasets\FaceForensics -d FaceShifter -c c23 -t videos
   ```

   Run `python download-FaceForensics.py -h` first to confirm the dataset names your copy
   accepts. If it is slow, add `--server EU2` or `--server CA`.
5. Optional, for the compression experiment: the same six commands with `-c c40` (~2 GB more).

Check: you should see

```
C:\datasets\FaceForensics\original_sequences\youtube\c23\videos\000.mp4 ... 999.mp4
C:\datasets\FaceForensics\manipulated_sequences\Deepfakes\c23\videos\000_003.mp4 ...
   (and Face2Face, FaceSwap, NeuralTextures, FaceShifter; 1,000 videos each)
```

The official train/val/test split lists are already in `data\splits\ffpp\`.

---

## 3. Celeb-DF v2: request access now

1. Read <https://github.com/yuezunli/celeb-deepfakeforensics>.
2. Fill in its Google Form: <https://forms.gle/2jYBby6y1FBU3u6q9> (same details as above).
3. You will get a Google Drive link to `Celeb-DF-v2.zip` (~10-12 GB). Extract it to
   `C:\datasets\Celeb-DF-v2\`:

   ```
   C:\datasets\Celeb-DF-v2\Celeb-real\id0_0000.mp4 ...        590 videos
   C:\datasets\Celeb-DF-v2\YouTube-real\00000.mp4 ...         300 videos
   C:\datasets\Celeb-DF-v2\Celeb-synthesis\id0_id1_0000.mp4 ... 5,639 videos
   C:\datasets\Celeb-DF-v2\List_of_testing_videos.txt          518 lines
   ```

Only the 518 test videos are processed: Celeb-DF is used purely to test on data the
detector never saw.

---

## Citations (put these in your README / report when you publish results)

* S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois. Unsupervised skin tissue
  segmentation for remote photoplethysmography. *Pattern Recognition Letters* 124, 2019.
* A. Rössler et al. FaceForensics++: Learning to detect manipulated facial images. *ICCV* 2019.
* Y. Li, X. Yang, P. Sun, H. Qi, S. Lyu. Celeb-DF: A large-scale challenging dataset for
  DeepFake forensics. *CVPR* 2020.
