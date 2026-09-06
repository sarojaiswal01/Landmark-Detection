import os
import hashlib
import urllib.request
import json
import socket
import time
import argparse
from PIL import Image
import concurrent.futures
import threading

# Timeout settings
socket.setdefaulttimeout(15)

LANDMARKS_QUERIES = {
    # --- India ---
    "Taj_Mahal": "Taj Mahal Agra",
    "India_Gate": "India Gate New Delhi",
    "Gateway_of_India": "Gateway of India Mumbai",
    "Red_Fort": "Red Fort Delhi",
    "Qutub_Minar": "Qutub Minar Delhi",
    "Lotus_Temple": "Lotus Temple Delhi",
    "Charminar": "Charminar Hyderabad",
    "Mysore_Palace": "Mysore Palace",
    "Hawa_Mahal": "Hawa Mahal Jaipur",
    "Konark_Sun_Temple": "Konark Sun Temple",
    "Victoria_Memorial": "Victoria Memorial Kolkata",
    "Gol_Gumbaz": "Gol Gumbaz Bijapur",
    "Brihadeeswarar_Temple": "Brihadeeswarar Temple Thanjavur",
    "Sanchi_Stupa": "Sanchi Stupa",
    "Ajanta_Caves": "Ajanta Caves",
    "Ellora_Caves": "Ellora Caves",
    "Humayuns_Tomb": "Humayun Tomb Delhi",
    "Meenakshi_Temple": "Meenakshi Temple Madurai",
    "Golden_Temple": "Golden Temple Amritsar",
    "Nalanda_Ruins": "Nalanda Ruins Bihar",
    # --- Asia ---
    "Great_Wall_of_China": "Great Wall of China",
    "Forbidden_City": "Forbidden City Beijing",
    "Mount_Fuji": "Mount Fuji Japan",
    "Angkor_Wat": "Angkor Wat Cambodia",
    "Petronas_Twin_Towers": "Petronas Twin Towers Kuala Lumpur",
    "Marina_Bay_Sands": "Marina Bay Sands Singapore",
    "Borobudur_Temple": "Borobudur Temple Java",
    "Shwedagon_Pagoda": "Shwedagon Pagoda Myanmar",
    "Taipei_101": "Taipei 101 Taiwan",
    "Gyeongbokgung_Palace": "Gyeongbokgung Palace Seoul",
    # --- Europe ---
    "Eiffel_Tower": "Eiffel Tower Paris",
    "Big_Ben": "Big Ben London",
    "Colosseum": "Colosseum Rome",
    "Leaning_Tower_of_Pisa": "Leaning Tower of Pisa",
    "Arc_de_Triomphe": "Arc de Triomphe Paris",
    "Louvre_Museum": "Louvre Museum Paris",
    "Notre_Dame_Cathedral": "Notre-Dame Cathedral Paris",
    "Tower_Bridge": "Tower Bridge London",
    "Buckingham_Palace": "Buckingham Palace London",
    "Acropolis_of_Athens": "Acropolis Athens Parthenon",
    # --- North America ---
    "Statue_of_Liberty": "Statue of Liberty New York",
    "Golden_Gate_Bridge": "Golden Gate Bridge San Francisco",
    "Empire_State_Building": "Empire State Building New York",
    "White_House": "White House Washington DC",
    "Mount_Rushmore": "Mount Rushmore",
    "Space_Needle": "Space Needle Seattle",
    "CN_Tower": "CN Tower Toronto",
    "Chichen_Itza": "Chichen Itza Mexico",
    # --- South America ---
    "Christ_the_Redeemer": "Christ the Redeemer Rio de Janeiro",
    "Machu_Picchu": "Machu Picchu Peru",
    "Iguazu_Falls": "Iguazu Falls",
    "Moai_Statues": "Moai Easter Island statues",
    # --- Middle East ---
    "Burj_Khalifa": "Burj Khalifa Dubai",
    "Sheikh_Zayed_Grand_Mosque": "Sheikh Zayed Grand Mosque Abu Dhabi",
    "Petra": "Petra Jordan Treasury",
    "Dome_of_the_Rock": "Dome of the Rock Jerusalem",
    # --- Africa ---
    "Pyramids_of_Giza": "Pyramids of Giza Egypt",
    "Great_Sphinx": "Great Sphinx Giza",
    "Table_Mountain": "Table Mountain Cape Town",
    "Hassan_II_Mosque": "Hassan II Mosque Casablanca",
    # --- Oceania ---
    "Sydney_Opera_House": "Sydney Opera House",
    "Sydney_Harbour_Bridge": "Sydney Harbour Bridge",
    "Uluru": "Uluru Ayers Rock Australia",
    "Sky_Tower": "Sky Tower Auckland New Zealand",
    # --- New Landmarks ---
    "Stonehenge": "Stonehenge monument Wiltshire UK",
    "Neuschwanstein_Castle": "Neuschwanstein Castle Bavaria Germany",
    "Sagrada_Familia": "Sagrada Familia Barcelona Spain",
    "Hagia_Sophia": "Hagia Sophia Istanbul Turkey",
    "Amer_Fort": "Amer Fort Jaipur Rajasthan India",
    "Hampi_Ruins": "Hampi ruins Karnataka India",
    "Khajuraho_Temples": "Khajuraho Group of Monuments Madhya Pradesh India",
    "Tokyo_Tower": "Tokyo Tower Japan",
    "Burj_Al_Arab": "Burj Al Arab hotel Dubai",
    "Merlion": "Merlion statue Singapore",
    "Victoria_Falls": "Victoria Falls waterfall",
    "Blue_Mosque": "Sultan Ahmed Mosque Blue Mosque Istanbul Turkey",
    "Kremlin": "Moscow Kremlin Russia",
    "Sydney_Tower": "Sydney Tower eye Australia",
    "Grand_Palace": "Grand Palace Bangkok Thailand",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Download clean landmark dataset from Wikimedia Commons.")
    parser.add_argument("--dest", type=str, default="dataset", help="Output directory")
    parser.add_argument("--limit", type=int, default=15, help="Number of images to download per landmark")
    return parser.parse_args()

def get_wikimedia_thumb_url(url, width=500):
    if not url.startswith("https://upload.wikimedia.org/wikipedia/commons/"):
        return url
    
    parts = url.split('/')
    if len(parts) >= 8 and parts[4] == 'commons' and parts[5] != 'thumb':
        filename = parts[-1]
        hash1 = parts[5]
        hash2 = parts[6]
        thumb_url = f"https://upload.wikimedia.org/wikipedia/commons/thumb/{hash1}/{hash2}/{filename}/{width}px-{filename}"
        return thumb_url
        
    return url

def get_image_urls(query, limit):
    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query"
        f"&generator=search&gsrsearch={encoded_query}&gsrnamespace=6"
        f"&gsrlimit=100&prop=imageinfo&iiprop=url&format=json"
    )
    
    headers = {
        'User-Agent': 'LandmarkLensAIDownloader/1.0 (contact: info@landmarklens.ai) Python-urllib'
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        pages = data.get("query", {}).get("pages", {})
        urls = []
        
        for page_id, page_info in pages.items():
            imageinfo = page_info.get("imageinfo", [])
            if imageinfo:
                img_url = imageinfo[0].get("url")
                if img_url and img_url.lower().endswith(('.jpg', '.jpeg', '.png')):
                    thumb_url = get_wikimedia_thumb_url(img_url)
                    urls.append(thumb_url)
                    
        return urls
    except Exception as e:
        print(f"[!] Error querying API for query '{query}': {e}")
        return []

def download_image(url, dest_path, seen_hashes, lock=None):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            img_data = response.read()
            
        hasher = hashlib.md5()
        hasher.update(img_data)
        img_hash = hasher.hexdigest()
        
        if lock:
            lock.acquire()
        try:
            if img_hash in seen_hashes:
                return False, "Duplicate Content"
            seen_hashes.add(img_hash)
        finally:
            if lock:
                lock.release()
                
        with open(dest_path, 'wb') as f:
            f.write(img_data)
            
        with Image.open(dest_path) as img:
            img.verify()
            
        return True, None
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False, str(e)

def download_landmark(landmark_class, query, dest_dir, limit, seen_hashes, lock):
    landmark_folder = os.path.join(dest_dir, landmark_class)
    os.makedirs(landmark_folder, exist_ok=True)
    
    existing_images = [f for f in os.listdir(landmark_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    existing_count = len(existing_images)
    
    if existing_count >= limit:
        print(f"[*] Skipping {landmark_class}: Already has {existing_count} images (limit: {limit})")
        return 0
        
    needed = limit - existing_count
    print(f"[*] Fetching images for: {landmark_class} (has {existing_count}, downloading {needed} more)...")
    img_urls = get_image_urls(query, limit * 3)
    
    downloaded = 0
    for idx, url in enumerate(img_urls):
        if downloaded >= needed:
            break
            
        dest_file = os.path.join(landmark_folder, f"{landmark_class}_{existing_count + downloaded + 1}.jpg")
        success, err_reason = download_image(url, dest_file, seen_hashes, lock)
        if success:
            downloaded += 1
            print(f"    [+] Saved: {landmark_class} {existing_count + downloaded}/{limit}")
            
    print(f"    Finished {landmark_class}: Downloaded {downloaded} new images.")
    return downloaded

def main():
    args = parse_args()
    dest_dir = os.path.abspath(args.dest)
    os.makedirs(dest_dir, exist_ok=True)
    
    print("="*60)
    print("FAMOUS LANDMARKS DATASET DOWNLOADER (MULTITHREADED)")
    print("="*60)
    print(f"Destination folder: {dest_dir}")
    print(f"Target limit per landmark: {args.limit} images")
    print("="*60)
    
    start_time = time.time()
    seen_hashes = set()
    
    # Pre-populate seen_hashes with existing images to prevent inter-run duplicates
    print("[*] Building cache of existing image hashes...")
    for root, dirs, files in os.walk(dest_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'rb') as f:
                        img_data = f.read()
                    hasher = hashlib.md5()
                    hasher.update(img_data)
                    seen_hashes.add(hasher.hexdigest())
                except Exception:
                    pass
    print(f"[+] Loaded {len(seen_hashes)} existing image hashes.")
    
    lock = threading.Lock()
    total_new = 0
    
    # Download landmarks in parallel using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                download_landmark, 
                landmark_class, 
                query, 
                dest_dir, 
                args.limit, 
                seen_hashes, 
                lock
            ): landmark_class for landmark_class, query in LANDMARKS_QUERIES.items()
        }
        
        for future in concurrent.futures.as_completed(futures):
            landmark_class = futures[future]
            try:
                downloaded_count = future.result()
                total_new += downloaded_count
            except Exception as e:
                print(f"[!] Error downloading landmark '{landmark_class}': {e}")
        
    duration = time.time() - start_time
    print("\n" + "="*60)
    print("DATASET DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Total New Images Downloaded: {total_new}")
    print(f"Total Duration: {duration:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()
