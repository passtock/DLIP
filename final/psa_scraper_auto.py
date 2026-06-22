import os
import csv
import time
from bs4 import BeautifulSoup
from seleniumbase import SB
import re

# ==========================================
# 1. 설정 및 경로 준비
# ==========================================
SPEC_ID = "4063735"  # 테스트할 타겟 스펙 아이디 (예: 저스틴 허버트)
TARGET_GRADES = [8, 9, 10]
CLICK_LOAD_MORE_COUNT = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "test")
os.makedirs(TEST_DATA_DIR, exist_ok=True)
CSV_FILE_PATH = os.path.join(TEST_DATA_DIR, "card_cllection_list.csv")

def append_to_csv(item_name, grade, cert_number):
    file_exists = os.path.isfile(CSV_FILE_PATH)
    with open(CSV_FILE_PATH, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Item Name", "Grade", "Cert Number"])
        writer.writerow([item_name, f"PSA {grade}", cert_number])
        
def parse_and_extract_certs(html, current_item_name, current_grade, saved_certs):
    """현재 화면에 렌더링된 HTML을 분석하여 Cert 번호들을 추출합니다."""
    soup = BeautifulSoup(html, 'html.parser')
    extracted_count = 0
    
    # 전략 1: Sales History 등의 테이블(td)나 링크(a)에서 /cert/ 가 포함된 8자리 숫자 찾기
    # PSA 홈페이지 구조상 링크(href)에 cert 번호가 포함되어 있거나, 텍스트 자체로 존재합니다.
    
    # 정규식: 7자리~8자리 숫자로 이루어진 단어
    cert_pattern = re.compile(r'\b\d{7,8}\b')
    
    # 1. 모든 링크에서 cert 번호 추출 시도
    for a in soup.find_all('a'):
        href = a.get('href', '')
        if '/cert/' in href:
            match = cert_pattern.search(href)
            if match:
                cert = match.group(0)
                if cert not in saved_certs:
                    saved_certs.add(cert)
                    append_to_csv(current_item_name, current_grade, cert)
                    extracted_count += 1
                    
    # 2. 혹은 화면에 노출된 텍스트 테이블에서 추출 시도 (위에서 못 찾았을 경우 대비)
    if extracted_count == 0:
        # 화면의 모든 텍스트 중에서 연속된 7~8자리 숫자를 cert로 간주 (너무 많을 수 있으니 표나 특정 영역으로 한정 가능)
        # 현재는 가장 확실한 링크(/cert/) 기반으로 추출하도록 우선순위를 두었습니다.
        pass
        
    return extracted_count

def run_automated_scraper():
    print(f"🚀 PSA 자동 수집 봇 시작 (Spec ID: {SPEC_ID})")
    print(f"저장 경로: {CSV_FILE_PATH}\n")
    saved_certs = set()
    
    with SB(uc=True, headless=False) as sb:
        # 최초 1회만 메인 페이지로 접속 (URL 파라미터 제외)
        url = f"https://www.psacard.com/spec/psa/{SPEC_ID}"
        print(f"접속 중: {url}")
        sb.uc_open_with_reconnect(url, 5)
        sb.uc_gui_click_captcha()
        sb.sleep(5)
        
        # Datadome API 차단 에러 감지 및 수동 개입 (매우 중요!)
        html = sb.get_page_source()
        if "contact support if this error continues" in html or "Please try again" in html:
            print("\n🚨 [경고] PSA의 강력한 API 방어막(Datadome)에 의해 페이지 데이터가 로딩되지 않았습니다.")
            print("🚨 봇 차단은 뚫었으나 내부 데이터 통신이 막힌 상태입니다.")
            print("👉 **해결 방법**: 지금 열려있는 팝업 브라우저에서 '새로고침(F5)'을 몇 번 누르시거나, 캡차가 뜨면 직접 풀어주세요.")
            print("👉 화면에 정상적으로 'Sales History'나 카드가 보이면, 그때 아래에서 엔터를 눌러주세요!")
            input("준비가 완료되면 [Enter] 키를 누르세요...")
        
        for grade in TARGET_GRADES:
            print(f"\n======================================")
            print(f"진행 중: PSA {grade} 등급")
            print(f"======================================")
            
            # 1. 등급 드롭다운 클릭 (회원님이 주신 XPath 사용)
            try:
                # 회원님이 주신 특정 XPath (바뀔 수 있으므로 예외처리)
                dropdown_xpath = '//*[@id="radix-_R_qrb52npfiv5uiupiivb_"]/span[2]/svg'
                # 혹시 동적 ID가 바뀌었을 경우를 대비한 범용 CSS Selector (PSA 8, 9, 10 등이 적힌 버튼)
                fallback_selector = "button[aria-haspopup='menu']:contains('PSA')"
                
                if sb.is_element_visible(dropdown_xpath):
                    sb.click(dropdown_xpath)
                else:
                    sb.click(fallback_selector)
                    
                sb.sleep(1)
                
                # 드롭다운 메뉴 안에서 해당 등급(PSA 8, 9, 10) 클릭
                grade_menu_item = f"div[role='menuitem']:contains('PSA {grade}')"
                sb.click(grade_menu_item)
                sb.sleep(4) # 데이터 로딩 대기
            except Exception as e:
                print(f"⚠️ 등급 드롭다운을 클릭하는 데 실패했습니다: {e}")
                print("수동으로 드롭다운에서 등급을 선택하시고 엔터를 눌러주세요.")
                input("선택 후 [Enter] 키를 누르세요...")
            
            item_name = "Unknown Item"
            soup = BeautifulSoup(sb.get_page_source(), 'html.parser')
            h1 = soup.find('h1')
            if h1:
                item_name = h1.text.strip()

            # 2. '더 보기' 버튼 10회 누르기 (회원님이 주신 XPath 반영)
            print(f"✔ '더 보기' 버튼 {CLICK_LOAD_MORE_COUNT}회 연속 클릭 시작...")
            for i in range(CLICK_LOAD_MORE_COUNT):
                try:
                    button_xpath = '//*[@id="main"]/div[2]/div[2]/button'
                    fallback_button = "button:contains('Show More')"
                    
                    if sb.is_element_visible(button_xpath):
                        sb.scroll_to(button_xpath)
                        sb.click(button_xpath)
                    else:
                        sb.scroll_to(fallback_button)
                        sb.click(fallback_button)
                        
                    print(f"  [{i+1}/{CLICK_LOAD_MORE_COUNT}] 더 보기 버튼 클릭 완료")
                    sb.sleep(2.5) # 새로운 목록이 로드될 때까지 대기
                except Exception as e:
                    print(f"  더 이상 누를 수 있는 버튼이 없거나 페이지 끝에 도달했습니다.")
                    break
            
            print(f"✔ 리스트 펼치기 완료. 현재 화면의 HTML에서 Cert 번호를 수집합니다.")
            final_html = sb.get_page_source()
            count = parse_and_extract_certs(final_html, item_name, grade, saved_certs)
            
            if count == 0:
                print(f"⚠️ Cert 번호를 찾지 못했습니다.")
                print(f"현재 띄워진 브라우저 화면에서 Cert 번호가 화면에 보이는 상태인지 확인해주세요.")
            else:
                print(f"✅ PSA {grade} 등급에서 {count}개의 데이터를 성공적으로 스크랩하여 CSV에 저장했습니다.")
            
    print("\n🎉 모든 등급 크롤링 완료!")

if __name__ == "__main__":
    run_automated_scraper()
