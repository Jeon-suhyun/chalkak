"""공공 용어사전 CSV → data/*.csv 로 변환 (찰칵사전 용어 DB)

실행:  python build_terms.py
출력 열: term, category, easy, source, official, plain
  easy     : 팀이 쓴 쉬운 설명 (공공 사전은 비워 둠 → 앱에서 AI가 공식 정의를 쉽게 풀어 줌)
  official : 기관이 쓴 공식 정의 원문
  plain    : 기관이 제공한 '쉬운 용어' (심평원만 있음)
"""
import csv
import glob
import os
import re

os.makedirs("data", exist_ok=True)

# 20대가 이미 아는 일상어라 서류마다 빨간 상자가 너무 많이 생기는 단어 (팀이 검토해서 늘리거나 줄이세요)
STOP = set("""
전세 입찰 회사 환불 합병 한옥 하천 토지 토사 측량 취득 최고 차량 지도 주식 제방 인증 유지 어도 시도 세법 선박 석재
산지 사도 빌딩 빈집 부표 법인 면적 매매 도로 녹지 권리 군도 국세 구도 공지 공원 계약 건축 거실 구획 갱지 각지 농지
공부 층고 초본 지형 매립 가각 주택 부지 원룸 운하 염전 연체 연식 획지 알선 리츠 물건 개인정보 단독주택 공청회
외래 일반의 병리과 이비인후과 종합병원 정신건강의학과 직업환경의학과 한방재활의학과 침구과 오십견
""".split())


def read(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return list(csv.DictReader(raw.decode(enc).splitlines()))
        except UnicodeDecodeError:
            pass
    raise ValueError(f"인코딩을 알 수 없음: {path}")


def clean_def(s):
    s = re.sub(r"<\s*출처[^>]*>", "", s or "")
    s = s.strip().strip('"').strip()
    return re.sub(r"\s+", " ", s)


def variants(name):
    """'(의료급여) 수급권자' → ['의료급여 수급권자', '수급권자']
       '체당금(체불 임금 등 대지급금)' → ['체당금', '체불 임금 등 대지급금']
       '구상권(求償權)' → ['구상권']  (한자 풀이는 버림)"""
    name = re.sub(r"\s+", " ", name.replace("\t", " ")).strip()
    out = []
    m = re.match(r"^\(([^)]*)\)\s*(.+)$", name)
    if m:
        out += [f"{m[1]} {m[2]}" if " " in name else m[1] + m[2], m[2]]
    else:
        m = re.match(r"^(.+?)\s*\(([^)]*)\)\s*$", name)
        if m:
            out.append(m[1])
            if not re.search(r"[一-鿿]", m[2]):
                out.append(m[2])
        else:
            out.append(name)
    return [v.strip() for v in out if len(v.strip()) >= 2 and v.strip() not in STOP]


def write(path, rows):
    seen, uniq = set(), []
    for r in rows:
        if r["term"] not in seen:
            seen.add(r["term"])
            uniq.append(r)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["term", "category", "easy", "source", "official", "plain"])
        w.writeheader()
        w.writerows(uniq)
    print(f"{path}: {len(uniq)}개")


def find(pattern):
    files = glob.glob(pattern)
    return files[0] if files else None


# 1) 한국자산관리공사 온비드 용어사전 → 부동산
f = find("raw_data/*온비드*용어사전*.csv")
if f:
    rows = []
    for r in read(f):
        d = clean_def(r["용어설명"])
        for t in variants(r["용어"]):
            rows.append({"term": t, "category": "부동산", "easy": "", "official": d, "plain": "",
                         "source": "한국자산관리공사 온비드 용어사전(공공데이터포털)"})
    write("data/onbid.csv", rows)

# 2) 건강보험심사평가원 건강보험 용어사전 → 병원
f = find("raw_data/*건강보험심사평가원*용어사전*.csv")
if f:
    rows = []
    for r in read(f):
        d = clean_def(r["설명세부내용"])
        plain = (r.get("쉬운용어단어명") or "").strip()
        for t in variants(r["단어명"]):
            rows.append({"term": t, "category": "병원", "easy": "", "official": d, "plain": plain,
                         "source": "건강보험심사평가원 건강보험 용어사전(공공데이터포털, 공공누리 제1유형)"})
    write("data/hira.csv", rows)

# 3) 금융위원회 금융용어사전 → 보험(보험 관련) / 금융  — 파일을 받으면 열 이름 확인 후 연결
f = find("raw_data/*금융위원회*금융용어*.csv")
if f:
    data = read(f)
    print("금융위원회 사전 열:", list(data[0].keys()))
    tcol = next(c for c in data[0] if "용어" in c and "설명" not in c and "내용" not in c)
    dcol = next(c for c in data[0] if c != tcol and ("설명" in c or "내용" in c or "정의" in c))
    rows = []
    for r in data:
        d = clean_def(r[dcol])
        for t in variants(r[tcol]):
            cat = "보험" if ("보험" in t or "보험" in d[:80]) else "금융"
            rows.append({"term": t, "category": cat, "easy": "", "official": d, "plain": "",
                         "source": "금융위원회 금융용어사전(공공데이터포털)"})
    write("data/fsc.csv", rows)
