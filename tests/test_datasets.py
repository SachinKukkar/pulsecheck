"""Loaders on tiny fake folder trees laid out exactly like the official downloads."""

import json

import numpy as np

from pulsecheck import datasets


def touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_ffpp_labels_groups_and_official_splits(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.json").write_text(json.dumps([["000", "003"]]))
    (splits / "val.json").write_text(json.dumps([["001", "004"]]))
    (splits / "test.json").write_text(json.dumps([["002", "005"]]))
    root = tmp_path / "ffpp"
    for v in ("000", "003", "002"):
        touch(root / "original_sequences/youtube/c23/videos" / f"{v}.mp4")
    for m in ("Deepfakes", "FaceSwap"):
        touch(root / "manipulated_sequences" / m / "c23/videos/000_003.mp4")
        touch(root / "manipulated_sequences" / m / "c23/videos/002_005.mp4")
    items = datasets.ffpp(root, "c23", methods=("Deepfakes", "FaceSwap"), splits_dir=splits)
    by = {it.key: it for it in items}
    assert by["ffpp_c23_original_000"].label == 0 and by["ffpp_c23_original_000"].split == "train"
    assert by["ffpp_c23_Deepfakes_000_003"].label == 1 and by["ffpp_c23_Deepfakes_000_003"].split == "train"
    assert by["ffpp_c23_FaceSwap_002_005"].split == "test"
    # source and target of a pair share one group, so they can never straddle a split
    assert by["ffpp_c23_original_003"].group == by["ffpp_c23_Deepfakes_000_003"].group == "ffpp_000_003"


def test_celebdf_uses_only_the_official_test_list(tmp_path):
    root = tmp_path / "celeb"
    for rel in ("Celeb-real/id0_0000.mp4", "Celeb-real/id1_0001.mp4", "YouTube-real/00000.mp4",
                "Celeb-synthesis/id0_id1_0000.mp4", "Celeb-synthesis/id2_id3_0003.mp4"):
        touch(root / rel)
    (root / "List_of_testing_videos.txt").write_text("1 Celeb-real/id0_0000.mp4\n1 YouTube-real/00000.mp4\n0 Celeb-synthesis/id0_id1_0000.mp4\n")
    items = datasets.celebdf(root)
    assert sorted(it.key for it in items) == ["celebdf_Celeb-real_id0_0000", "celebdf_Celeb-synthesis_id0_id1_0000", "celebdf_YouTube-real_00000"]
    assert {it.key: it.label for it in items}["celebdf_Celeb-synthesis_id0_id1_0000"] == 1
    assert len(datasets.celebdf(root, test_only=False)) == 5


def test_ubfc_finds_videos_and_both_ground_truth_formats(tmp_path):
    root = tmp_path / "ubfc"
    touch(root / "DATASET_2/subject1/vid.avi")
    t = np.arange(90) / 30.0
    ppg = np.sin(2 * np.pi * 1.2 * t)
    np.savetxt(root / "DATASET_2/subject1/ground_truth.txt", np.vstack([ppg, np.full_like(t, 72), t]))
    touch(root / "DATASET_1/5-gt/vid.avi")
    np.savetxt(root / "DATASET_1/5-gt/gtdump.xmp", np.column_stack([t * 1000, np.full_like(t, 72), np.full_like(t, 98), ppg]), delimiter=",")
    items = datasets.ubfc(root)
    assert len(items) == 2 and all(it.label == 0 and it.gt for it in items)
    for it in items:
        gt_t, gt_ppg = datasets.load_ubfc_gt(it.gt)
        assert gt_t[0] == 0 and abs(gt_t[-1] - t[-1]) < 1e-6
        assert np.allclose(gt_ppg, ppg)


def test_hash_split_is_deterministic_and_balanced():
    s = [datasets.hash_split(f"g{i}") for i in range(2000)]
    assert s == [datasets.hash_split(f"g{i}") for i in range(2000)]
    frac = {k: s.count(k) / len(s) for k in ("train", "val", "test")}
    assert 0.65 < frac["train"] < 0.75 and 0.11 < frac["test"] < 0.19
