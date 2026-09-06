import os
from PIL import Image

def validate_dataset():
    dest_dir = os.path.abspath("dataset")
    if not os.path.exists(dest_dir):
        print("[!] Dataset directory not found.")
        return
        
    print("="*60)
    print("DATASET INTEGRITY VALIDATOR")
    print("="*60)
    
    total_checked = 0
    corrupt_deleted = 0
    class_counts = {}
    
    for root, dirs, files in os.walk(dest_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, file)
                total_checked += 1
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                    # Also try to open and resize to verify decoding
                    with Image.open(filepath) as img:
                        img.resize((224, 224))
                except Exception as e:
                    print(f"    [!] Corrupt image detected and removed: {file} ({e})")
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
                    corrupt_deleted += 1
                    
    # Recalculate distribution
    for item in os.listdir(dest_dir):
        item_path = os.path.join(dest_dir, item)
        if os.path.isdir(item_path):
            img_files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_counts[item] = len(img_files)
            
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total checked: {total_checked}")
    print(f"Corrupt deleted: {corrupt_deleted}")
    print(f"Valid images remaining: {total_checked - corrupt_deleted}")
    print(f"Total active classes: {len(class_counts)}")
    print("="*60)
    
    # Print top 5 and bottom 5 classes by image count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    print("\n[Min Images Classes - Top 5]:")
    for name, count in sorted_classes[:5]:
        print(f"  - {name}: {count} images")
    print("\n[Max Images Classes - Top 5]:")
    for name, count in sorted_classes[-5:]:
        print(f"  - {name}: {count} images")

if __name__ == "__main__":
    validate_dataset()
