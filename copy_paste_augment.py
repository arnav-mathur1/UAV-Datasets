"""
Offline copy-paste augmentation for multi-UAV detection.

Reads the original COCO dataset, extracts UAV crops, and pastes them onto
images to create synthetic multi-UAV training/valid/test data.
Outputs ONLY synthetic images + their own COCO JSON to separate folders.
Does NOT modify the original dataset.

Paste counts are size-adaptive:
  - Large images (>= 800px height): 3-4 UAVs pasted (total 2-5 with existing)
  - Small images (< 800px height):  1-2 UAVs pasted (total 2-3 with existing)
"""

import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

# ============ CONFIG ============
BASE_DIR = "/home/spark/Arnav/datasets/uav_sod_combined2/final_comb"
OUTPUT_BASE = "/home/spark/Arnav/datasets/uav_sod_combined2/synthetic_copypaste"

SPLITS = {
    "train": {"target_count": 5000},
    "valid": {"target_count": 1000},
    "test":  {"target_count": 1000},
}

# Size-adaptive paste counts
LARGE_IMG_THRESHOLD = 800   # height in px
LARGE_PASTE_RANGE = (3, 4)  # paste count for large images
SMALL_PASTE_RANGE = (1, 2)  # paste count for small images

SCALE_RANGE = (0.5, 2.0)    # relative to existing UAVs in target
MIN_CROP_PX = 10            # minimum crop dimension
CROP_PADDING = 5            # context padding around crops
MAX_PLACEMENT_RETRIES = 15

VIZ_SAMPLES = 10            # per split
SEED = 42
# ================================

random.seed(SEED)
np.random.seed(SEED)


def load_coco(ann_path):
    with open(ann_path) as f:
        coco = json.load(f)
    img_id_to_info = {img["id"]: img for img in coco["images"]}
    img_id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_id_to_anns[ann["image_id"]].append(ann)
    cat_ids = [c["id"] for c in coco["categories"]]
    return coco, img_id_to_info, img_id_to_anns, cat_ids


def boxes_overlap(box_a, box_b):
    """Check if two [x, y, w, h] boxes have ANY pixel overlap."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def find_placement(img_h, img_w, paste_w, paste_h, existing_boxes):
    """Find a valid random placement that does not overlap ANY existing box."""
    for _ in range(MAX_PLACEMENT_RETRIES):
        x = random.randint(0, max(0, img_w - paste_w))
        y = random.randint(0, max(0, img_h - paste_h))
        new_box = [x, y, paste_w, paste_h]

        valid = True
        for eb in existing_boxes:
            if boxes_overlap(new_box, eb):
                valid = False
                break
        if valid:
            return x, y
    return None


def paste_crop_onto_image(img, crop, x, y):
    """Hard-paste a crop onto the image at (x, y), handling boundary clipping."""
    ih, iw = img.shape[:2]
    ch, cw = crop.shape[:2]

    src_x1 = max(0, -x)
    src_y1 = max(0, -y)
    src_x2 = min(cw, iw - x)
    src_y2 = min(ch, ih - y)

    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return img

    img[dst_y1:dst_y2, dst_x1:dst_x2] = crop[src_y1:src_y2, src_x1:src_x2]
    return img


def draw_viz(img, boxes, out_path):
    """Draw bounding boxes on image for visualization."""
    viz = img.copy()
    for bx, by, bw, bh in boxes:
        cv2.rectangle(viz, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 255, 0), 2)
    cv2.imwrite(out_path, viz)


def build_crop_bank(split_dir, img_id_to_info, img_id_to_anns):
    """Extract UAV crops from all annotated images in a split."""
    crops = []
    for img_id, anns in tqdm(img_id_to_anns.items(), desc="  Building crop bank"):
        if not anns:
            continue
        info = img_id_to_info[img_id]
        img_path = os.path.join(split_dir, info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        for ann in anns:
            bx, by, bw, bh = ann["bbox"]
            bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)
            if bw < MIN_CROP_PX or bh < MIN_CROP_PX:
                continue

            px1 = max(0, bx - CROP_PADDING)
            py1 = max(0, by - CROP_PADDING)
            px2 = min(w, bx + bw + CROP_PADDING)
            py2 = min(h, by + bh + CROP_PADDING)

            crop = img[py1:py2, px1:px2].copy()
            crops.append({
                "crop": crop,
                "bbox_w": bw,
                "bbox_h": bh,
                "source_id": img_id,
            })
    return crops


def process_split(split_name, target_count):
    """Generate synthetic images for one split."""
    split_dir = os.path.join(BASE_DIR, split_name)
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    out_dir = os.path.join(OUTPUT_BASE, split_name)
    viz_dir = os.path.join(out_dir, "_viz_samples")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Processing split: {split_name} (target: {target_count} images)")
    print(f"{'='*50}")

    coco, img_id_to_info, img_id_to_anns, cat_ids = load_coco(ann_path)
    category_id = cat_ids[0]

    # All image IDs that have annotations (can paste more onto them)
    # Also include negatives (no annotations — paste fresh UAVs)
    all_img_ids = [img["id"] for img in coco["images"]]

    crop_bank = build_crop_bank(split_dir, img_id_to_info, img_id_to_anns)
    print(f"  Crop bank size: {len(crop_bank)}")

    if not crop_bank:
        print(f"  WARNING: No crops found for {split_name}, skipping.")
        return

    # Median bbox size for images with no existing UAVs
    all_bw = [c["bbox_w"] for c in crop_bank]
    all_bh = [c["bbox_h"] for c in crop_bank]
    median_bw = int(np.median(all_bw))
    median_bh = int(np.median(all_bh))
    print(f"  Median crop size: {median_bw}x{median_bh}")

    out_images = []
    out_annotations = []
    next_img_id = max(img["id"] for img in coco["images"]) + 1
    next_ann_id = (max(ann["id"] for ann in coco["annotations"]) + 1) if coco["annotations"] else 1
    synth_count = 0
    viz_count = 0

    # Shuffle and cycle through images until we hit target_count
    random.shuffle(all_img_ids)
    idx = 0

    with tqdm(total=target_count, desc=f"  Generating {split_name}") as pbar:
        while synth_count < target_count:
            img_id = all_img_ids[idx % len(all_img_ids)]
            idx += 1

            # If we've cycled through all images, reshuffle
            if idx % len(all_img_ids) == 0:
                random.shuffle(all_img_ids)

            info = img_id_to_info[img_id]
            img_path = os.path.join(split_dir, info["file_name"])
            img = cv2.imread(img_path)
            if img is None:
                continue
            ih, iw = img.shape[:2]

            existing_anns = img_id_to_anns.get(img_id, [])
            existing_boxes = [ann["bbox"] for ann in existing_anns]

            # Reference size for scale matching
            if existing_boxes:
                ref_w = np.mean([b[2] for b in existing_boxes])
                ref_h = np.mean([b[3] for b in existing_boxes])
            else:
                ref_w, ref_h = median_bw, median_bh

            # Size-adaptive paste count
            if ih >= LARGE_IMG_THRESHOLD:
                n_paste = random.randint(*LARGE_PASTE_RANGE)
            else:
                n_paste = random.randint(*SMALL_PASTE_RANGE)

            new_boxes = []
            all_boxes = list(existing_boxes)

            for _ in range(n_paste):
                donor = random.choice(crop_bank)
                if donor["source_id"] == img_id:
                    continue

                crop = donor["crop"]
                ch, cw = crop.shape[:2]

                scale_factor = random.uniform(*SCALE_RANGE)
                target_w = max(MIN_CROP_PX, int(ref_w * scale_factor))
                target_h = max(MIN_CROP_PX, int(ref_h * scale_factor))

                crop_aspect = cw / ch
                if crop_aspect > 1:
                    paste_w = target_w
                    paste_h = max(MIN_CROP_PX, int(target_w / crop_aspect))
                else:
                    paste_h = target_h
                    paste_w = max(MIN_CROP_PX, int(target_h * crop_aspect))

                if paste_w > iw * 0.5 or paste_h > ih * 0.5:
                    continue

                resized_crop = cv2.resize(crop, (paste_w, paste_h), interpolation=cv2.INTER_LINEAR)

                result = find_placement(ih, iw, paste_w, paste_h, all_boxes)
                if result is None:
                    continue
                px, py = result

                img = paste_crop_onto_image(img, resized_crop, px, py)
                new_box = [px, py, paste_w, paste_h]
                new_boxes.append(new_box)
                all_boxes.append(new_box)

            if not new_boxes:
                continue

            # Save synthetic image
            base, ext = os.path.splitext(info["file_name"])
            # Add unique suffix to avoid duplicates when cycling
            synth_name = f"synth_{synth_count:05d}_{base}{ext}"
            cv2.imwrite(os.path.join(out_dir, synth_name), img)

            out_images.append({
                "id": next_img_id,
                "file_name": synth_name,
                "width": iw,
                "height": ih,
            })

            # Keep original annotations (re-ID'd)
            for ann in existing_anns:
                out_annotations.append({
                    "id": next_ann_id,
                    "image_id": next_img_id,
                    "category_id": category_id,
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": 0,
                })
                next_ann_id += 1

            # Add pasted annotations
            for box in new_boxes:
                bx, by, bw, bh = box
                out_annotations.append({
                    "id": next_ann_id,
                    "image_id": next_img_id,
                    "category_id": category_id,
                    "bbox": [float(bx), float(by), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                })
                next_ann_id += 1

            next_img_id += 1
            synth_count += 1
            pbar.update(1)

            if viz_count < VIZ_SAMPLES:
                draw_viz(img, all_boxes, os.path.join(viz_dir, synth_name))
                viz_count += 1

    # Write output COCO JSON
    out_coco = {
        "images": out_images,
        "annotations": out_annotations,
        "categories": coco["categories"],
    }
    out_json_path = os.path.join(out_dir, "_annotations.coco.json")
    with open(out_json_path, "w") as f:
        json.dump(out_coco, f)

    # Stats
    img_ann_count = defaultdict(int)
    for ann in out_annotations:
        img_ann_count[ann["image_id"]] += 1
    ann_counts = defaultdict(int)
    for count in img_ann_count.values():
        ann_counts[count] += 1

    print(f"\n  === {split_name} Results ===")
    print(f"  Synthetic images: {synth_count}")
    print(f"  Total annotations: {len(out_annotations)}")
    print(f"  UAVs per image distribution:")
    for n, c in sorted(ann_counts.items()):
        print(f"    {n} UAVs: {c} images")
    print(f"  Output: {out_dir}")
    print(f"  COCO JSON: {out_json_path}")
    print(f"  Viz samples: {viz_dir}")


def main():
    for split_name, cfg in SPLITS.items():
        process_split(split_name, cfg["target_count"])
    print("\n\nAll done!")


if __name__ == "__main__":
    main()
