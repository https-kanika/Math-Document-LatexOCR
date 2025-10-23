import os
import json
import random
import shutil
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Convert annotations to YOLO format with train/val/test split')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing annotations and images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for YOLO dataset')
    parser.add_argument('--ann_folder', type=str, default='annotations', help='Folder name for annotations')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder name for images')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup paths
    BASE_IN = Path(args.input_dir)
    BASE_OUT = Path(args.output_dir)
    ANN_DIR = BASE_IN / args.ann_folder
    IMG_DIR = BASE_IN / args.img_folder
    
    # Create output directories
    LBL_DIR = BASE_OUT / "labels"
    OUT_IMG_DIR = BASE_OUT / "images"
    
    # Create folders for train/val/test splits
    for split in ["train", "val", "test"]:
        for p in [LBL_DIR/split, OUT_IMG_DIR/split]:
            p.mkdir(parents=True, exist_ok=True)

    # Modified class mapping - combined symbol types into one class
    class_map = {
        "text": 0,
        "equation": 1,
        "symbol": 2,  
        "text_with_inline_symbol": 2  
    }
    
    # Create class names list for YAML (only unique classes)
    class_names = ["text", "equation", "symbol"]  # Fixed order matching class_map values

    def normalize_bbox(bbox, W, H):
        x1, y1, x2, y2 = bbox
        x_c = ((x1 + x2) / 2) / W
        y_c = ((y1 + y2) / 2) / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        return x_c, y_c, w, h

    # Get all JSON files
    files = sorted([f for f in os.listdir(ANN_DIR) if f.endswith(".json")])
    
    # Shuffle and split 70% train, 15% validation, 15% test
    random.seed(42)
    random.shuffle(files)
    
    train_split = int(len(files) * 0.7)
    val_split = int(len(files) * 0.85)
    
    train_files = files[:train_split]
    val_files = files[train_split:val_split]
    test_files = files[val_split:]
    
    print(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files")

    def convert_list(list_files, split_name):
        processed = 0
        for fname in list_files:
            jpath = ANN_DIR / fname
            with open(jpath) as f:
                data = json.load(f)
            W, H = data["width"], data["height"]
            lines = []

            for el in data.get("elements", []):
                t = el["type"]

                # Case 1: Simple types (text and equation)
                if t in ("text", "equation"):
                    cls = class_map[t]
                    bbox = el["bbox"]
                    lines.append(f"{cls} {' '.join(map(lambda x: f'{x:.6f}', normalize_bbox(bbox, W, H)))}")
                
                # Case 2: Symbols (both standalone symbols and symbols within text)
                elif t == "symbol":
                    cls = class_map[t]
                    bbox = el["bbox"]
                    lines.append(f"{cls} {' '.join(map(lambda x: f'{x:.6f}', normalize_bbox(bbox, W, H)))}")

                # Case 3: text_with_inline_symbol - extract only the symbol part
                elif t == "text_with_inline_symbol":
                    # Only add the symbol bbox, not the text parts
                    if "symbol_bbox" in el:
                        x_c, y_c, w, h = normalize_bbox(el["symbol_bbox"], W, H)
                        lines.append(f"{class_map['symbol']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                    
                    # Add text parts as text class
                    if "first_text_bbox" in el:
                        x_c, y_c, w, h = normalize_bbox(el["first_text_bbox"], W, H)
                        lines.append(f"{class_map['text']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                    if "second_text_bbox" in el:
                        x_c, y_c, w, h = normalize_bbox(el["second_text_bbox"], W, H)
                        lines.append(f"{class_map['text']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

            # write YOLO label file
            out_label = LBL_DIR / split_name / fname.replace(".json", ".txt")
            out_label.write_text("\n".join(lines))

            # copy image to correct split
            base_img = fname.replace(".json", ".jpg")
            if not (IMG_DIR / base_img).exists():
                base_img = fname.replace(".json", ".png")
            src = IMG_DIR / base_img
            dst = OUT_IMG_DIR / split_name / base_img
            if src.exists():
                shutil.copy(src, dst)
                processed += 1
            else:
                print(f"⚠️ Missing image for {fname}")
                
        return processed

    # Process each split
    train_processed = convert_list(train_files, "train")
    val_processed = convert_list(val_files, "val")
    test_processed = convert_list(test_files, "test")
    
    # Create dataset.yaml file with simplified class structure
    with open(BASE_OUT / "dataset.yaml", 'w') as f:
        f.write(f"# YOLOv8 dataset configuration\n")
        f.write(f"path: {BASE_OUT.absolute()}  # dataset root directory\n")
        f.write(f"train: images/train  # train images relative to 'path'\n")
        f.write(f"val: images/val  # val images relative to 'path'\n")
        f.write(f"test: images/test  # test images relative to 'path'\n\n")
        
        f.write("# Classes (simplified to 3 classes)\n")
        f.write("names:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"  {i}: '{class_name}'\n")
    
    print(f"✅ Conversion Done.")
    print(f"   - Train: {train_processed} images")
    print(f"   - Val:   {val_processed} images")
    print(f"   - Test:  {test_processed} images")
    print(f"   - YAML:  {BASE_OUT / 'dataset.yaml'}")
    print(f"   - Classes: text (0), equation (1), symbol (2)")

if __name__ == "__main__":
    main()