import requests
from bs4 import BeautifulSoup
import csv
import time
import re

# ——— 설정 ———
BASE_URL    = 'https://jnfarm.jeonnam.go.kr'
LIST_PATH   = '/farm/property/propertyList.do'
DETAIL_PATH = '/farm/property/propertyView.do'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

# 최종 CSV 컬럼 순서
FIELDNAMES = [
    '용지종류', '공부지목', '실지목', '판매구분', '매매/임대가',
    '면적', '매물주소', '판매자 이름', '판매자 연락처',
    '담당자 이름(부서)', '담당자 연락처', '진행현황', '등록일', '특이사항'
]

session = requests.Session()
session.headers.update(HEADERS)

results = []

for page in range(1, 117):
    # 1) 목록 페이지 요청
    params = {
        'menuCd':        'farm005001',
        'currentPageNo': page,
        'nPageSize':     10,
        'category':      '001',
        'searchCity':    '',
        'searchArea':    '',
        'searchType':    '',
        'transactState': '',
        'type':          '',
    }
    resp = session.get(BASE_URL + LIST_PATH, params=params)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    # 2) 목록 테이블
    table = soup.find('table', class_='job_tableStyle_center')
    if not table:
        print(f"[경고] {page}페이지에서 목록 테이블을 찾지 못함.")
        continue

    for tr in table.select('tbody > tr'):
        # 3) “진행 중”인 항목만
        status = tr.select_one('td:last-child')
        if not status or '진행 중' not in status.get_text():
            continue

        # 4) onclick 에서 propertyId 파싱
        onclick_txt = tr.get('onclick', '')
        m = re.search(r"fn_view\('(\d+)'\)", onclick_txt)
        if not m:
            print(f"  ⚠️ ID 파싱 실패: {onclick_txt}")
            continue
        property_id = m.group(1)

        # 5) 상세 페이지 요청
        detail_params = {
            'menuCd':     'farm005001',
            'propertyId': property_id,
        }
        dresp = session.get(BASE_URL + DETAIL_PATH, params=detail_params)
        dresp.raise_for_status()
        dsoup = BeautifulSoup(dresp.text, 'html.parser')

        # 6) 세부정보 테이블에서 <th>–<td> 매핑
        data = {}
        # HTML 상 클래스가 pgm_tableStyle_center 인 테이블의 tbody>tr 순회
        for tr2 in dsoup.select('table.pgm_tableStyle_center tbody tr'):
            ths = tr2.find_all('th')
            tds = tr2.find_all('td')
            for idx, th in enumerate(ths):
                key = th.get_text(strip=True)
                # td 가 존재하면, 없으면 빈 문자열
                cell = tds[idx] if idx < len(tds) else None
                if not cell:
                    val = ''
                else:
                    # 용지종류는 span 여러 개가 있을 수 있으므로 특별 처리
                    if key == '용지종류':
                        spans = cell.find_all('span')
                        if spans:
                            val = '/'.join(s.get_text(strip=True) for s in spans)
                        else:
                            val = cell.get_text(strip=True)
                    else:
                        val = cell.get_text(strip=True)
                data[key] = val

        # 7) FIELDNAMES 순으로 정렬하여 결과에 추가
        record = {fld: data.get(fld, '') for fld in FIELDNAMES}
        results.append(record)

        time.sleep(0.2)  # 상세 페이지 과부하 방지

    print(f"페이지 {page} 완료 – 누적 {len(results)}건")
    time.sleep(1)  # 목록 페이지 과도 요청 방지

# 8) CSV 저장
with open('농지.csv', 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(results)

print("✅ 크롤링 완료: 농지.csv 생성됨")