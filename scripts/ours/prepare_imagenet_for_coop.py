#!/usr/bin/env python3
import os
import tarfile
import shutil
from pathlib import Path

import scipy.io as sio


DOWNLOAD_DIR = Path("/data/meta_prompt_1/_downloads/imagenet")
DATASET_DIR = Path("/data/meta_prompt_1/datasets/imagenet")

TRAIN_TAR = DOWNLOAD_DIR / "ILSVRC2012_img_train.tar"
VAL_TAR = DOWNLOAD_DIR / "ILSVRC2012_img_val.tar"
DEVKIT_TAR = DOWNLOAD_DIR / "ILSVRC2012_devkit_t12.tar.gz"

IMAGES_DIR = DATASET_DIR / "images"
TRAIN_DIR = IMAGES_DIR / "train"
VAL_DIR = IMAGES_DIR / "val"
DEVKIT_DIR = DATASET_DIR / "devkit"
VAL_FLAT_DIR = DATASET_DIR / "_val_flat_tmp"


def count_images(root: Path):
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png"})


def safe_extract_tar(tar: tarfile.TarFile, path: Path):
    path = path.resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(path)):
            raise RuntimeError(f"Unsafe tar path: {member.name}")
    tar.extractall(path)


def extract_devkit():
    marker = DEVKIT_DIR / ".extracted"
    if marker.exists():
        print("[SKIP] devkit already extracted")
        return

    DEVKIT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[EXTRACT] devkit: {DEVKIT_TAR} -> {DEVKIT_DIR}")
    with tarfile.open(DEVKIT_TAR, "r:gz") as tar:
        safe_extract_tar(tar, DEVKIT_DIR)
    marker.write_text("ok\n")


def find_file(root: Path, name: str):
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"Cannot find {name} under {root}")
    return hits[0]


def parse_synsets():
    meta_path = find_file(DEVKIT_DIR, "meta.mat")
    print(f"[READ] meta.mat: {meta_path}")

    meta = sio.loadmat(meta_path, squeeze_me=True)
    synsets = meta["synsets"]

    id_to_wnid = {}
    wnid_to_words = {}

    for s in synsets:
        # MATLAB struct fields: ILSVRC2012_ID, WNID, words, num_children, ...
        ilsvrc_id = int(s["ILSVRC2012_ID"])
        wnid = str(s["WNID"])
        words = str(s["words"])
        num_children = int(s["num_children"])

        # ImageNet-1K classes are leaf synsets.
        if num_children == 0:
            id_to_wnid[ilsvrc_id] = wnid
            wnid_to_words[wnid] = words

    if len(wnid_to_words) != 1000:
        print(f"[WARN] expected 1000 leaf synsets, got {len(wnid_to_words)}")

    return id_to_wnid, wnid_to_words


def write_classnames(wnid_to_words):
    out = DATASET_DIR / "classnames.txt"
    if out.exists():
        print(f"[SKIP] classnames.txt exists: {out}")
        return

    print(f"[WRITE] {out}")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for wnid in sorted(wnid_to_words):
            classname = wnid_to_words[wnid].replace("_", " ")
            f.write(f"{wnid} {classname}\n")


def extract_train():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    existing_dirs = [p for p in TRAIN_DIR.iterdir() if p.is_dir()] if TRAIN_DIR.exists() else []
    existing_count = count_images(TRAIN_DIR)

    if len(existing_dirs) >= 1000 and existing_count > 1000000:
        print(f"[SKIP] train seems extracted: dirs={len(existing_dirs)}, images={existing_count}")
        return

    print(f"[EXTRACT] train outer tar: {TRAIN_TAR}")
    print("[INFO] This streams nested class tar files directly, without keeping 1000 class .tar files.")

    with tarfile.open(TRAIN_TAR, "r") as outer:
        members = [m for m in outer.getmembers() if m.isfile() and m.name.endswith(".tar")]
        print(f"[INFO] class tar files: {len(members)}")

        for i, member in enumerate(members, 1):
            wnid = Path(member.name).stem
            class_dir = TRAIN_DIR / wnid

            if class_dir.exists() and count_images(class_dir) > 0:
                if i % 50 == 0:
                    print(f"[SKIP] {i}/{len(members)} {wnid}")
                continue

            class_dir.mkdir(parents=True, exist_ok=True)
            print(f"[EXTRACT] {i}/{len(members)} {wnid}")

            fileobj = outer.extractfile(member)
            if fileobj is None:
                raise RuntimeError(f"Failed to extract nested tar: {member.name}")

            with tarfile.open(fileobj=fileobj, mode="r:*") as inner:
                safe_extract_tar(inner, class_dir)

    print(f"[DONE] train images: {count_images(TRAIN_DIR)}")


def extract_val(id_to_wnid):
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    if len([p for p in VAL_DIR.iterdir() if p.is_dir()]) >= 1000 and count_images(VAL_DIR) >= 50000:
        print(f"[SKIP] val seems extracted: images={count_images(VAL_DIR)}")
        return

    gt_path = find_file(DEVKIT_DIR, "ILSVRC2012_validation_ground_truth.txt")
    labels = [int(x.strip()) for x in gt_path.read_text().splitlines() if x.strip()]
    if len(labels) != 50000:
        print(f"[WARN] expected 50000 val labels, got {len(labels)}")

    if not VAL_FLAT_DIR.exists() or count_images(VAL_FLAT_DIR) < 50000:
        if VAL_FLAT_DIR.exists():
            shutil.rmtree(VAL_FLAT_DIR)
        VAL_FLAT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[EXTRACT] val tar: {VAL_TAR} -> {VAL_FLAT_DIR}")
        with tarfile.open(VAL_TAR, "r") as tar:
            safe_extract_tar(tar, VAL_FLAT_DIR)
    else:
        print(f"[SKIP] val flat already extracted: {VAL_FLAT_DIR}")

    print("[MOVE] val images into wnid folders")
    for idx, label_id in enumerate(labels, 1):
        wnid = id_to_wnid[label_id]
        dst_dir = VAL_DIR / wnid
        dst_dir.mkdir(parents=True, exist_ok=True)

        imname = f"ILSVRC2012_val_{idx:08d}.JPEG"
        src = VAL_FLAT_DIR / imname
        dst = dst_dir / imname

        if dst.exists():
            continue
        if not src.exists():
            raise FileNotFoundError(f"Missing val image: {src}")

        shutil.move(str(src), str(dst))

        if idx % 5000 == 0:
            print(f"[MOVE] {idx}/50000")

    print(f"[DONE] val images: {count_images(VAL_DIR)}")

    # Clean temporary flat val dir if empty or no longer needed.
    if VAL_FLAT_DIR.exists():
        remaining = count_images(VAL_FLAT_DIR)
        if remaining == 0:
            shutil.rmtree(VAL_FLAT_DIR)
            print("[CLEAN] removed empty val flat tmp")


def main():
    print(f"[DOWNLOAD_DIR] {DOWNLOAD_DIR}")
    print(f"[DATASET_DIR]  {DATASET_DIR}")

    assert TRAIN_TAR.exists(), f"Missing {TRAIN_TAR}"
    assert VAL_TAR.exists(), f"Missing {VAL_TAR}"
    assert DEVKIT_TAR.exists(), f"Missing {DEVKIT_TAR}"

    extract_devkit()
    id_to_wnid, wnid_to_words = parse_synsets()
    write_classnames(wnid_to_words)
    extract_train()
    extract_val(id_to_wnid)

    print("\n[SUMMARY]")
    print(f"classnames: {DATASET_DIR / 'classnames.txt'}")
    print(f"train dirs : {len([p for p in TRAIN_DIR.iterdir() if p.is_dir()])}")
    print(f"train imgs : {count_images(TRAIN_DIR)}")
    print(f"val dirs   : {len([p for p in VAL_DIR.iterdir() if p.is_dir()])}")
    print(f"val imgs   : {count_images(VAL_DIR)}")


if __name__ == "__main__":
    main()
