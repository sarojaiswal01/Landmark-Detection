import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Delete rejected/unknown images from the source dataset based on the cleaning report.")
    parser.add_argument("--report", type=str, default="data/cleaning_report.json", help="Path to cleaning report JSON file")
    parser.add_argument("--src", type=str, default="dataset", help="Path to raw dataset folder")
    args = parser.parse_args()

    report_path = os.path.abspath(args.report)
    src_dir = os.path.abspath(args.src)

    if not os.path.exists(report_path):
        print(f"[!] Error: Cleaning report '{report_path}' does not exist.")
        return

    print(f"[*] Reading report file: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    files_log = report.get("files_report", [])
    deleted_count = 0
    errors_count = 0

    rejections = [item for item in files_log if item.get("status") == "Rejected"]
    total_rejections = len(rejections)
    
    if total_rejections == 0:
        print("[+] No rejected files logged in the report. Nothing to clean.")
        return

    print(f"[*] Starting cleanup. Found {total_rejections} rejected/unknown images to delete from '{src_dir}'.")

    for idx, item in enumerate(rejections):
        rel_path = item.get("file")
        full_path = os.path.join(src_dir, rel_path)
        
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
                deleted_count += 1
            except Exception as e:
                print(f"[!] Failed to delete {full_path}: {e}")
                errors_count += 1

    print("\n" + "="*50)
    print("DATASET PURGE CLEANUP COMPLETE")
    print("="*50)
    print(f"Total Rejected Images Logged: {total_rejections}")
    print(f"Successfully Deleted: {deleted_count}")
    if errors_count > 0:
        print(f"Failed to Delete (errors): {errors_count}")
    print("="*50)

if __name__ == "__main__":
    main()
