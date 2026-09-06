import os
import urllib.request
import json
import socket
import time
import argparse
from PIL import Image
import concurrent.futures
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Timeout settings
socket.setdefaulttimeout(10)

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

# Setup Thread-Safe Keep-Alive Connection Pool with Automatic Retries & Backoff
session = requests.Session()
retries = Retry(
    total=4,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    raise_on_status=False
)
adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

def get_wikimedia_thumb_url(url, width=500):
    if not url.startswith("https://upload.wikimedia.org/wikipedia/commons/"):
        return url
    
    parts = url.split('/')
    if len(parts) >= 8 and parts[4] == 'commons' and parts[5] != 'thumb':
        filename = parts[-1]
        hash1 = parts[5]
        hash2 = parts[6]
        return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{hash1}/{hash2}/{filename}/{width}px-{filename}"
        
    return url

def get_image_urls(query):
    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query"
        f"&generator=search&gsrsearch={encoded_query}&gsrnamespace=6"
        f"&gsrlimit=100&prop=imageinfo&iiprop=url&format=json"
    )
    
    headers = {
        'User-Agent': 'LandmarkLensAIDownloader/2.0 (contact: info@landmarklens.ai)'
    }
    
    try:
        # Use session to query API too for speed
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
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
    except Exception:
        pass
    return []

def extract_filename(url):
    parts = url.split('/')
    if 'thumb' in parts:
        return parts[-2]
    return parts[-1]

def download_single_image(url, dest_path):
    headers = {
        'User-Agent': 'LandmarkLensAIDownloader/2.0 (contact: info@landmarklens.ai)'
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            with Image.open(dest_path) as img:
                img.verify()
            return True
    except Exception:
        pass
        
    if os.path.exists(dest_path):
        try:
            os.remove(dest_path)
        except Exception:
            pass
    return False

def main():
    dest_dir = os.path.abspath("dataset")
    os.makedirs(dest_dir, exist_ok=True)
    
    limit = 100 # Download exactly 100 images per landmark
    print(f"[*] Starting high-speed Keep-Alive download of {limit} images for {len(LANDMARKS_QUERIES)} landmarks...")
    
    # 1. Fetch URLs in parallel using session
    print("[*] Querying Wikimedia Commons API for all landmarks in parallel...")
    landmark_urls = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = {
            executor.submit(get_image_urls, query): class_name 
            for class_name, query in LANDMARKS_QUERIES.items()
        }
        for future in concurrent.futures.as_completed(futures):
            class_name = futures[future]
            urls = future.result()
            
            # Deduplicate by filename
            unique_urls = []
            seen_filenames = set()
            for u in urls:
                filename = extract_filename(u).lower()
                if filename not in seen_filenames:
                    seen_filenames.add(filename)
                    unique_urls.append(u)
            
            landmark_urls[class_name] = unique_urls[:limit]
            print(f"    [+] {class_name}: Found {len(unique_urls)} unique URLs (kept {len(landmark_urls[class_name])})")

    # 2. Flatten tasks for global parallel downloading
    download_tasks = []
    skipped_count = 0
    for class_name, urls in landmark_urls.items():
        class_folder = os.path.join(dest_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)
        for idx, url in enumerate(urls):
            dest_file = os.path.join(class_folder, f"{class_name}_{idx + 1}.jpg")
            
            # Check if file already exists and is non-empty
            if os.path.exists(dest_file) and os.path.getsize(dest_file) > 1024:
                skipped_count += 1
                continue
                
            download_tasks.append((url, dest_file, class_name))
            
    print(f"\n[*] Skipping {skipped_count} already downloaded files.")
    print(f"[*] Queueing {len(download_tasks)} downloads in keep-alive thread pool...")
    
    # 3. Run downloads in parallel (12 worker threads with keep-alive is highly efficient and rate-limit safe)
    start_time = time.time()
    total_downloaded = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(download_single_image, url, dest_path): (url, dest_path, class_name)
            for url, dest_path, class_name in download_tasks
        }
        
        for future in concurrent.futures.as_completed(futures):
            url, dest_path, class_name = futures[future]
            try:
                success = future.result()
                if success:
                    total_downloaded += 1
                    if total_downloaded % 100 == 0:
                        print(f"    [Progress] Downloaded {total_downloaded}/{len(download_tasks)} images...")
            except Exception:
                pass
                
    duration = time.time() - start_time
    print("\n" + "="*60)
    print("KEEP-ALIVE DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Total New Images Saved: {total_downloaded}")
    print(f"Duration: {duration:.2f} seconds ({total_downloaded / duration:.2f} images/sec)")
    print("="*60)

if __name__ == "__main__":
    main()
