import os
import csv
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# ==========================================
# 사용자 설정
# ==========================================
# 발급받으신 PSA API 토큰
API_TOKEN = "BZMMcOXj70bcuqNY0mT3GSIUu32MELkecSKnXH70qhnGmtCSZDhz1fd-9j88ikeTUBWDnuC5gXXuYcEgxs1dKiT-Ng1lQgvQ7Yie-vBuq6aX_al1oBjJEznUV59GnsLa4URBKD4SO1s-6rBELv0Y4jJQoLuzSi6LcuOl5o6KZc-0WILQw6PQeSe_eVe_PgVNTY4vEBqVeUu6L1XSJKAXezW6wNZ5C2UhDdOsTSzVvdPHSO-39XTKVZVf4BkujupqCtlM__Q0WcWkYsq5faclc37D4dz--2LgPurzSqNKCrR6Rx3J"

# 저장할 최상위 디렉토리 생성
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# 긁어올 목표 Spec ID 리스트 (잭슨 홀리데이, 코비 브라이언트)
TARGET_SPEC_IDS = ["8598088", "364602"]

HEADERS = {
    "Authorization": f"bearer {API_TOKEN}",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*"
}

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"이미지 다운로드 에러: {url} -> {e}")
    return False

def scrape_spec_data(spec_id):
    print(f"\n🚀 시작: Spec ID {spec_id} 데이터 수집 중...")
    
    csv_file_path = os.path.join(RAW_DATA_DIR, f"psa_data_{spec_id}.csv")
    img_folder = os.path.join(RAW_DATA_DIR, f"images_{spec_id}")
    os.makedirs(img_folder, exist_ok=True)
    
    # 1. Spec 페이지에 직접 접근하여 기본 정보 구하기
    url = f"https://www.psacard.com/spec/psa/{spec_id}?gt=SINGLE_GRADED"
    res = requests.get(url, headers=HEADERS)
    
    if res.status_code != 200:
        print(f"❌ 접속 실패 (Status: {res.status_code})")
        return

    soup = BeautifulSoup(res.text, 'html.parser')
    
    # 대표 고화질 이미지 찾기 (cloudfront 링크)
    images = []
    for img_tag in soup.find_all('img'):
        src = img_tag.get('src')
        if src and "cloudfront.net/spec" in src:
            images.append(src)
            
    # 거래(옥션) 내역에서 Cert Number 추출 시도 (이베이 썸네일 등)
    sales = []
    
    # PSA는 거래 내역이나 APR(Auction Price Realized)에 일부 Cert Number가 노출됨
    # 이 스크립트는 HTML 내 특정 텍스트 또는 렌더링된 세일즈 내역 썸네일을 기반으로 수집합니다.
    # Note: 완벽한 대량 Cert 조회를 위해서는 PSA Public API 'Search' 엔드포인트를 써야 하지만 
    # 현재 토큰의 권한 범위를 알 수 없으므로 우선 페이지 파싱 + 이미지 다운로드를 진행합니다.
    
    sales_containers = soup.find_all(string=lambda text: "Lot No." in text if text else False)
    for t in sales_containers:
        sales.append(str(t).strip())

    cert_data = []
    
    print(f"📸 대표/고화질 이미지 {len(images)}개 발견 완료")
    
    # CSV 저장 준비
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Spec_ID', 'Image_File', 'Source_URL', 'Memo'])
        
        # 이미지 다운로드 및 CSV 기록
        for idx, img_url in enumerate(images):
            # 이미지 파일명 추출
            parsed_url = urlparse(img_url)
            file_name_ext = os.path.basename(parsed_url.path)
            if not file_name_ext.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                file_name_ext = f"image_{idx}.jpg"
                
            save_name = f"psa_{spec_id}_{idx}_{file_name_ext}"
            save_path = os.path.join(img_folder, save_name)
            
            print(f"  ⬇️ 다운로드 중: {save_name}...")
            
            if download_image(img_url, save_path):
                writer.writerow([spec_id, save_name, img_url, 'HQ Spec Image'])
                
        # 판매 내역 더미 기록
        for s in sales:
            writer.writerow([spec_id, 'N/A', '', s])
            
    print(f"✅ Spec ID {spec_id} 작업 완료! (저장 폴더: {img_folder})")
    time.sleep(2)  # 밴 방지용 휴식

if __name__ == "__main__":
    print(f"데이터 저장 경로: {RAW_DATA_DIR}\n")
    for s_id in TARGET_SPEC_IDS:
        scrape_spec_data(s_id)
        
    print("\n🎉 모든 수집이 완료되었습니다!")
