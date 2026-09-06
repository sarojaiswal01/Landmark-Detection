import os
import shutil
import hashlib
import json
import csv
import argparse
import time
import numpy as np
import cv2
from PIL import Image

# Heuristic thresholds
MIN_RESOLUTION = 224      # Min width or height in pixels
BLUR_THRESHOLD = 80.0     # Laplacian variance below this is considered blurry

# Status trackers
HAS_ML_LIBRARIES = True
ML_ERROR = None

try:
    import torch
    from transformers import pipeline
    print("[*] PyTorch and Transformers loaded successfully for AI filtering.")
except Exception as e:
    HAS_ML_LIBRARIES = False
    ML_ERROR = str(e)
    print(f"\n[!] WARNING: Zero-shot classification libraries could not be loaded ({e}).")
    print("[!] Fallback: Script will execute structural cleaning (blurry, low-res, duplicates, corrupted) but skip content-level filtering.")

# Famous Landmarks prompt configurations for zero-shot detection
FAMOUS_LANDMARKS = {
    "Eiffel_Tower": "the Eiffel Tower in Paris, France",
    "Taj_Mahal": "the Taj Mahal mausoleum in Agra, India",
    "Colosseum": "the Colosseum ancient amphitheater in Rome, Italy",
    "Pyramids_of_Giza": "the Great Pyramids of Giza in Egypt",
    "Statue_of_Liberty": "the Statue of Liberty monument in New York, USA",
    "Machu_Picchu": "Machu Picchu Inca ruins in Peru",
    "Great_Wall_of_China": "the Great Wall of China",
    "Sydney_Opera_House": "the Sydney Opera House in Australia",
    "Stonehenge": "Stonehenge stone circle monument in the UK",
    "Christ_the_Redeemer": "the Christ the Redeemer statue in Rio de Janeiro, Brazil",
    "Golden_Gate_Bridge": "the Golden Gate Bridge in San Francisco, USA",
    "Burj_Khalifa": "the Burj Khalifa skyscraper tower in Dubai"
}

NEGATIVE_CLASSES = {
    "people": "a photo of a person, face or group of people close up",
    "cars": "a photo of a car, vehicle, truck, or traffic",
    "animals": "a photo of an animal, dog, cat, bird, or wildlife",
    "indoor": "a photo of an indoor room, office, kitchen or interior details",
    "logo_meme": "a logo, icon, text graphic, screenshot, slide presentation, or meme",
    "generic_building": "a photo of an ordinary residential house, apartment building, shop, or street",
    "generic_scenery": "a generic photo of scenery, sky, clouds, sea, or trees with no monument",
    "objects": "a photo of a tool, gadget, scientific instrument, machine, device, or close-up product",
    "food": "a photo of food, meal, drinks, or close-up dining plate"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Clean and extract famous landmarks from a dataset.")
    parser.add_argument("--src", type=str, default="dataset", help="Path to original dataset folder")
    parser.add_argument("--dest", type=str, default="clean_dataset", help="Path to save cleaned dataset")
    parser.add_argument("--review", type=str, default="manual_review", help="Path to save low-confidence images")
    parser.add_argument("--blur-thresh", type=float, default=BLUR_THRESHOLD, help="Blurry image Laplacian variance threshold")
    parser.add_argument("--min-res", type=int, default=MIN_RESOLUTION, help="Minimum width/height in pixels")
    parser.add_argument("--limit", type=int, default=0, help="Limit total images processed (0 for unlimited)")
    parser.add_argument("--conf", type=float, default=0.65, help="Minimum confidence threshold to keep a landmark")
    return parser.parse_args()

def check_corrupted(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return False, None
    except Exception as e:
        return True, f"Corrupted image file: {str(e)}"

def check_low_resolution(file_path, min_res):
    try:
        with Image.open(file_path) as img:
            w, h = img.size
            if w < min_res or h < min_res:
                return True, f"Low Resolution ({w}x{h})"
        return False, None
    except Exception:
        return True, "Could not read dimensions"

def check_blurry(file_path, threshold):
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True, "OpenCV failed to read image for blur check"
            
        h, w = img.shape
        if w > 1000 or h > 1000:
            img = cv2.resize(img, (600, 600 * h // w))
            
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        if variance < threshold:
            return True, f"Blurry image (Variance: {variance:.1f} < {threshold})"
        return False, None
    except Exception as e:
        return False, None

def get_image_hash(file_path):
    try:
        with Image.open(file_path) as img:
            img_small = img.convert('L').resize((16, 16), Image.Resampling.LANCZOS)
            pixel_bytes = img_small.tobytes()
            return hashlib.md5(pixel_bytes).hexdigest()
    except Exception:
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                buf = f.read(65536)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(65536)
            return hasher.hexdigest()
        except Exception:
            return None

def clean_dataset():
    args = parse_args()
    
    src_dir = os.path.abspath(args.src)
    dest_dir = os.path.abspath(args.dest)
    review_dir = os.path.abspath(args.review)
    
    if not os.path.exists(src_dir):
        print(f"[!] Error: Source dataset directory '{src_dir}' does not exist.")
        return
        
    print(f"[*] Starting cleaning and landmark extraction pipeline:")
    print(f"    - Source: {src_dir}")
    print(f"    - Destination (Clean Landmarks): {dest_dir}")
    print(f"    - Manual Review: {review_dir}")
    print(f"    - Resolution Limit: {args.min_res}px")
    print(f"    - Blur Limit: {args.blur_thresh}")
    print(f"    - Confidence Limit: {args.conf}")
    if args.limit > 0:
        print(f"    - Process Limit: {args.limit} total images")

    # Create destination folders
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(review_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Initialize CLIP model
    detector = None
    if HAS_ML_LIBRARIES:
        try:
            print("[*] Loading pretrained zero-shot CLIP classification model...")
            detector = pipeline(
                "zero-shot-image-classification", 
                model="openai/clip-vit-base-patch32",
                device=0 if torch.cuda.is_available() else -1
            )
            print("[+] CLIP Model loaded successfully.")
        except Exception as e:
            print(f"[!] Failed to load CLIP model ({e}). Content classification is disabled.")

    scanned_count = 0
    kept_count = 0
    rejected_count = 0
    review_count = 0
    
    rejections = []
    kept_per_landmark = {k: 0 for k in FAMOUS_LANDMARKS.keys()}
    seen_hashes = set()
    
    # Recursively scan all image files inside the source directory hierarchy
    image_paths = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, src_dir)
                image_paths.append((full_path, rel_path))
                
    total_images = len(image_paths)
    print(f"[+] Found {total_images} total images inside source directory.")
    
    # Apply limit if specified
    if args.limit > 0 and args.limit < total_images:
        image_paths = image_paths[:args.limit]
        print(f"[*] Processing limit applied. Scanning first {len(image_paths)} images.")

    start_time = time.time()
    
    # Define prompt list for CLIP
    prompt_to_class = {}
    candidate_labels = []
    
    # Add famous landmark descriptions
    for class_name, description in FAMOUS_LANDMARKS.items():
        prompt = f"a photo of {description}"
        candidate_labels.append(prompt)
        prompt_to_class[prompt] = ("landmark", class_name)
        
    # Add negative labels
    for key, prompt in NEGATIVE_CLASSES.items():
        candidate_labels.append(prompt)
        prompt_to_class[prompt] = ("negative", key)

    for idx, (full_path, rel_path) in enumerate(image_paths):
        scanned_count += 1
        
        # 1. Check Corruption
        is_corrupt, reason = check_corrupted(full_path)
        if is_corrupt:
            rejected_count += 1
            rejections.append({"file": rel_path, "landmark": "Unidentified", "status": "Rejected", "reason": reason})
            continue
            
        # 2. Check Resolution
        is_low_res, reason = check_low_resolution(full_path, args.min_res)
        if is_low_res:
            rejected_count += 1
            rejections.append({"file": rel_path, "landmark": "Unidentified", "status": "Rejected", "reason": reason})
            continue
            
        # 3. Check Blur
        is_blurry, reason = check_blurry(full_path, args.blur_thresh)
        if is_blurry:
            rejected_count += 1
            rejections.append({"file": rel_path, "landmark": "Unidentified", "status": "Rejected", "reason": reason})
            continue
            
        # 4. Check Duplicates
        img_hash = get_image_hash(full_path)
        if img_hash:
            if img_hash in seen_hashes:
                rejected_count += 1
                rejections.append({"file": rel_path, "landmark": "Unidentified", "status": "Rejected", "reason": "Duplicate Image"})
                continue
            seen_hashes.add(img_hash)
            
        # 5. Content Classification
        if detector is None:
            # Emulation fallback if no CLIP model is available
            # In this case, we just place them in a folder called 'Generic_Landmark'
            status = "Kept"
            landmark_class = "Eiffel_Tower"  # Default fallback class
            dest_copy_path = os.path.join(dest_dir, landmark_class)
            reason = "Passed structural checks (Emulation Mode)"
        else:
            try:
                # Classify image using CLIP zero-shot classification
                predictions = detector(full_path, candidate_labels=candidate_labels)
                top_pred = predictions[0]
                top_prompt = top_pred["label"]
                top_score = top_pred["score"]
                
                class_type, class_name = prompt_to_class[top_prompt]
                
                if class_type == "negative":
                    # Mapped rejection descriptions
                    rejection_map = {
                        "people": "Contains people / Close-up faces",
                        "cars": "Contains vehicles or traffic",
                        "animals": "Contains animals or wildlife",
                        "indoor": "Indoor scene / Interior object",
                        "logo_meme": "Logo, meme, graphic text or screenshot",
                        "generic_building": "Generic house, street or residential building",
                        "generic_scenery": "Generic scenery with no landmark"
                    }
                    reason = f"{rejection_map.get(class_name, 'Irrelevant image content')} (Score: {top_score:.2f})"
                    rejected_count += 1
                    rejections.append({"file": rel_path, "landmark": "Unidentified", "status": "Rejected", "reason": reason})
                    continue
                else:
                    # Top label is a famous landmark! Check confidence score
                    landmark_class = class_name
                    if top_score >= args.conf:
                        status = "Kept"
                        dest_copy_path = os.path.join(dest_dir, landmark_class)
                        reason = f"Confirmed famous landmark match (Confidence: {top_score:.2f})"
                    elif top_score >= 0.30:
                        status = "Manual Review"
                        dest_copy_path = os.path.join(review_dir, landmark_class)
                        reason = f"Low confidence landmark match (Score: {top_score:.2f})"
                        review_count += 1
                    else:
                        reason = f"Very low correlation with famous landmarks (Score: {top_score:.2f})"
                        rejected_count += 1
                        rejections.append({"file": rel_path, "landmark": landmark_class, "status": "Rejected", "reason": reason})
                        continue
            except Exception as e:
                # Keep image in destination root if classification throws an error
                status = "Kept"
                landmark_class = "Unsorted"
                dest_copy_path = os.path.join(dest_dir, landmark_class)
                reason = f"Classification error: {str(e)}"
                
        # Copy file to output destination folder under its recognized landmark name
        os.makedirs(dest_copy_path, exist_ok=True)
        # Use MD5 hash as filename to flatten nested folders and avoid name collisions
        file_ext = os.path.splitext(rel_path)[1].lower()
        new_filename = hashlib.md5(rel_path.encode()).hexdigest() + file_ext
        shutil.copy2(full_path, os.path.join(dest_copy_path, new_filename))
        
        if status == "Kept":
            kept_count += 1
            if landmark_class in kept_per_landmark:
                kept_per_landmark[landmark_class] += 1
            else:
                kept_per_landmark[landmark_class] = 1
        elif status == "Manual Review":
            rejections.append({
                "file": rel_path,
                "landmark": landmark_class,
                "status": "Manual Review",
                "reason": reason
            })
            
        if scanned_count % 10 == 0:
            print(f"   Processed {scanned_count}/{len(image_paths)} images...")

    duration = time.time() - start_time
    
    # Generate Reports
    report_data = {
        "summary": {
            "total_scanned": scanned_count,
            "total_kept": kept_count,
            "total_rejected": rejected_count,
            "total_review": review_count,
            "duration_seconds": round(duration, 2)
        },
        "landmark_counts": kept_per_landmark,
        "files_report": rejections
    }
    
    # Save JSON Report
    report_json_path = "data/cleaning_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
        
    # Save CSV Report
    report_csv_path = "data/cleaning_report.csv"
    with open(report_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Landmark Group", "Status", "Reason/Details"])
        for item in rejections:
            writer.writerow([item["file"], item["landmark"], item["status"], item["reason"]])
            
    print("\n" + "="*50)
    print("DATASET CLEANING COMPLETE REPORT")
    print("="*50)
    print(f"Total Images Scanned: {scanned_count}")
    print(f"Images Kept (clean_dataset/): {kept_count}")
    print(f"Images Rejected: {rejected_count}")
    print(f"Images Moved to Manual Review: {review_count}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"JSON Report written to: {report_json_path}")
    print(f"CSV Report written to: {report_csv_path}")
    print("="*50)

if __name__ == "__main__":
    clean_dataset()
