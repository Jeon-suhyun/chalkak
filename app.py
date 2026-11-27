"""
찰칵사전 - 서류 속 어려운 용어를 빨갛게 표시하고, 누르면 쉬운 설명을 보여 주는 Streamlit 앱

실행:
    pip install -r requirements.txt
    streamlit run app.py

OpenAI 키(선택): .streamlit/secrets.toml 파일에  OPENAI_API_KEY = "sk-..."  한 줄
                (또는 환경변수 OPENAI_API_KEY)
키가 없어도 사전 매칭 → 빨간 표시 → 클릭 설명은 동작합니다.

용어 사전: terms.csv + data/ 폴더의 *.csv 를 모두 읽습니다.
           열은 term, category, easy, source 네 개 (category: 부동산/근로/보험/병원)
"""
import glob
import hashlib
import io
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

DISPLAY_W = 800   # 화면에 보여 줄 이미지 폭(px). 빨간 상자·클릭 좌표도 이 크기 기준
OCR_W = 1600      # OCR은 더 큰 이미지로 읽어야 작은 글씨를 놓치지 않음
MODEL = "gpt-4o-mini"  # 사용 가능한 모델 이름은 OpenAI 문서에서 확인 후 바꾸세요
KIND_TO_CAT = {"부동산": "부동산", "알바·근로": "근로", "보험": "보험", "병원": "병원"}
DISCLAIMER = "※ 법률·의료 자문이 아닙니다. 중요한 결정은 전문가와 상담하세요."
AI_SOURCE = "AI 생성 (공식 사전 미수록)"


# ---------------------------------------------------------------- 1. 용어 사전
@st.cache_data
def load_terms():
    """terms.csv + data/*.csv 를 하나의 용어 DB로 합침"""
    files = [f for f in ["terms.csv"] + sorted(glob.glob("data/*.csv")) if os.path.exists(f)]
    cols = ["term", "category", "easy", "source", "official", "plain", "basis"]
    def read(f):  # 엑셀에서 'CSV(쉼표로 분리)'로 저장하면 cp949가 되므로 둘 다 읽음
        try:
            return pd.read_csv(f, encoding="utf-8-sig", dtype=str)
        except UnicodeDecodeError:
            return pd.read_csv(f, encoding="cp949", dtype=str)

    frames = [read(f).reindex(columns=cols) for f in files]
    df = pd.concat(frames, ignore_index=True).fillna("")
    df["term"] = df["term"].astype(str).str.strip()
    df = df[(df["easy"] != "") | (df["official"] != "")]  # 설명이 하나도 없는 용어는 버림
    df = df[df["term"].str.len() >= 2]  # 한 글자 용어는 오탐이 너무 많음
    return df.reset_index(drop=True)


def terms_for(all_terms, cat):
    """같은 용어가 여러 사전에 있으면 고른 서류 종류(cat)의 설명을 먼저 쓰고,
    긴 용어부터 정렬 ('우선변제권' 안의 '변제'가 먼저 잡히지 않도록)"""
    # 우선순위: 팀이 쓴 쉬운 설명이 있는 것 → 고른 서류 종류와 같은 분야
    df = all_terms.assign(_pri=(all_terms["easy"] == "").astype(int) * 2 + (all_terms["category"] != cat).astype(int))
    df = df.sort_values("_pri", kind="stable").drop_duplicates("term")
    df = df.sort_values("term", key=lambda s: s.str.len(), ascending=False, kind="stable")
    return df.drop(columns="_pri").reset_index(drop=True)


# ---------------------------------------------------------------- 2. 글자 읽기
@st.cache_resource
def get_ocr():
    import easyocr  # 첫 실행 때 한국어 모델을 내려받으므로 몇 분 걸릴 수 있음

    return easyocr.Reader(["ko", "en"], gpu=False)


def scale_to(img, width):
    return img.resize((width, round(img.height * width / img.width)))


@st.cache_data(show_spinner="사진에서 글자를 읽는 중… (처음 한 번은 오래 걸려요)")
def read_image(file_bytes):
    """사진 → (화면용 이미지, [(글자, (x0,y0,x1,y1)), ...], 평균 인식 신뢰도)"""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes))).convert("RGB")  # 휴대폰 사진 회전 보정
    ocr_img = img if img.width <= OCR_W else scale_to(img, OCR_W)
    show = scale_to(img, DISPLAY_W)
    s = DISPLAY_W / ocr_img.width  # OCR 좌표 → 화면 좌표
    lines, confs = [], []
    for box, text, conf in get_ocr().readtext(np.array(ocr_img)):  # EasyOCR에는 numpy 배열로 넘김
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        lines.append((text, (min(xs) * s, min(ys) * s, max(xs) * s, max(ys) * s)))
        confs.append(conf)
    return show, lines, (sum(confs) / len(confs) if confs else 0.0)


@st.cache_data(show_spinner="PDF를 여는 중…")
def read_pdf(file_bytes, page_no):
    """PDF 한 쪽 → (화면용 이미지, 글자 목록, 글자 정보 있음 여부, 인식 신뢰도)"""
    import fitz  # PyMuPDF

    page = fitz.open(stream=file_bytes, filetype="pdf")[page_no]
    if not page.get_text().strip():  # 스캔 PDF → 글자 정보가 없으므로 큰 해상도로 그려서 OCR
        z = OCR_W / page.rect.width
        pix = page.get_pixmap(matrix=fitz.Matrix(z, z))
        img, lines, q = read_image(pix.tobytes("png"))
        return img, lines, False, q
    zoom = DISPLAY_W / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    lines = []
    for blk in page.get_text("dict")["blocks"]:
        for line in blk.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            x0, y0, x1, y1 = line["bbox"]
            lines.append((text, (x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom)))
    return img, lines, True, 1.0


# ---------------------------------------------------------------- 3. 용어 찾기
def overlaps(a, b):
    """두 상자가 '절반 이상' 겹치면 같은 자리로 봄 ('우선변제권' 속 '변제' 같은 중복 방지).
    살짝 닿기만 하는 위아래 줄 상자는 겹침으로 보지 않음 (글씨가 작고 줄 간격이 좁은 서류 대비)"""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return small > 0 and ix * iy / small > 0.5


def add_hit(hits, term, box, row):
    # 같은 용어는 처음 나온 곳 한 번만 표시 (화면이 빨간 상자로 뒤덮이지 않게)
    if any(h["term"] == term for h in hits):
        return
    if not any(overlaps(box, h["box"]) for h in hits):
        hits.append({"term": term, "box": box, "row": row})


SHORT = 2  # 이 길이 이하 용어는 단어 첫머리에서만 찾음 ('가상계약서' 속 '상계' 같은 오탐 방지)


def short_ok(text, start):
    """앞 글자가 한글이 아니어야(=단어의 시작이어야) 짧은 용어로 인정"""
    return start == 0 or not re.match(r"[가-힣]", text[start - 1])


def find_terms_pdf(file_bytes, page_no, terms):
    """글자 정보가 있는 PDF는 PyMuPDF가 단어의 정확한 위치를 알려 줌 (상자가 어긋나지 않음)"""
    import fitz

    page = fitz.open(stream=file_bytes, filetype="pdf")[page_no]
    page_text = re.sub(r"\s+", "", page.get_text())
    zoom = DISPLAY_W / page.rect.width
    words = page.get_text("words")  # (x0, y0, x1, y1, 단어, ...)
    hits = []
    for row in terms.to_dict("records"):
        if re.sub(r"\s+", "", row["term"]) not in page_text:  # 없는 용어는 위치 검색 생략 (사전이 커도 빠름)
            continue
        if len(row["term"]) <= SHORT:  # 짧은 용어: 그 용어로 시작하는 단어만, 단어 앞부분 폭만큼 상자
            for x0, y0, x1, y1, w, *_ in words:
                for m in re.finditer(re.escape(row["term"]), w):
                    if short_ok(w, m.start()):
                        cw = (x1 - x0) / max(len(w), 1)
                        add_hit(hits, row["term"], ((x0 + cw * m.start()) * zoom - 2, y0 * zoom - 2,
                                                    (x0 + cw * m.end()) * zoom + 2, y1 * zoom + 2), row)
            continue
        for r in page.search_for(row["term"]):
            add_hit(hits, row["term"], (r.x0 * zoom - 2, r.y0 * zoom - 2, r.x1 * zoom + 2, r.y1 * zoom + 2), row)

    # PDF에 '【 갑 구 】', '계약전 알릴 의무'처럼 글자 사이 공백이 사전과 다르면 정확 검색이 실패함
    # → 아직 못 찾은 용어만 줄 단위로, 공백을 허용해서 다시 찾음
    located = {h["term"] for h in hits}
    missing = [r for r in terms.to_dict("records")
               if r["term"] not in located and re.sub(r"\s+", "", r["term"]) in page_text]
    if missing:
        lines = []
        for blk in page.get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                text = "".join(span["text"] for span in line["spans"])
                x0, y0, x1, y1 = line["bbox"]
                lines.append((text, (x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom)))
        for h in find_terms(lines, pd.DataFrame(missing)):
            add_hit(hits, h["term"], h["box"], h["row"])
    return hits


def fuzzy_terms(lines, terms, hits):
    """OCR이 한 글자를 잘못 읽은 경우(예: '계약갱신요구권'→'계약갱신요구궐')를 구제.
    네 글자 이상 용어만, 정확히 한 글자만 다를 때, 그 글자 묶음이 다른 사전 용어가 아닐 때만 인정"""
    known = {re.sub(r"\s+", "", t) for t in terms["term"]}
    found = {h["term"] for h in hits}
    line_info = []
    for text, box in lines:
        idx = [j for j, ch in enumerate(text) if not ch.isspace()]
        compact = "".join(text[j] for j in idx)
        line_info.append((text, box, idx, compact, bigrams(compact)))
    for row in terms.to_dict("records"):
        key = re.sub(r"\s+", "", row["term"])
        if len(key) < 4 or row["term"] in found:
            continue
        kb = bigrams(key)
        for text, (x0, y0, x1, y1), idx, compact, cb in line_info:
            if len(kb & cb) < len(kb) - 2:  # 빠른 거르기: 두 글자 묶음이 거의 다 겹쳐야 후보
                continue
            for i in range(len(compact) - len(key) + 1):
                w = compact[i:i + len(key)]
                if w not in known and sum(a != b for a, b in zip(w, key)) == 1:
                    tn = max(len(text), 1)
                    s0, s1 = idx[i], idx[i + len(key) - 1] + 1
                    add_hit(hits, row["term"], (x0 + (x1 - x0) * s0 / tn - 2, y0 - 2,
                                                x0 + (x1 - x0) * s1 / tn + 2, y1 + 2), row)
                    break
            if any(h["term"] == row["term"] for h in hits):
                break
    return hits


def find_terms(lines, terms, fuzzy=False):
    """사진(OCR) 글자 줄에서 용어를 찾아 [{term, box, row}] 반환.
    OCR은 띄어쓰기를 자주 틀리므로 글자 사이 공백을 허용해서 찾고,
    줄 안 위치는 글자 수 비율로 계산 (한글은 글자 폭이 거의 같아 잘 맞음)"""
    all_text = re.sub(r"\s+", "", "".join(t for t, _ in lines))
    cands = [r for r in terms.to_dict("records") if re.sub(r"\s+", "", r["term"]) in all_text]
    hits = []
    for text, (x0, y0, x1, y1) in lines:
        compact = re.sub(r"\s+", "", text)
        tn = max(len(text), 1)
        for row in cands:
            key = re.sub(r"\s+", "", row["term"])
            if key not in compact:
                continue
            # 글자 사이 공백을 허용해서 원문에서 위치를 찾음 → 상자 위치를 모든 용어가 같은 기준으로 계산
            pat = r"\s*".join(map(re.escape, key))
            for m in re.finditer(pat, text):
                if len(key) <= SHORT and not short_ok(text, m.start()):
                    continue
                add_hit(hits, row["term"], (x0 + (x1 - x0) * m.start() / tn - 2, y0 - 2,
                                            x0 + (x1 - x0) * m.end() / tn + 2, y1 + 2), row)
    if fuzzy:  # 사진(OCR)일 때만
        hits = fuzzy_terms(lines, terms, hits)
    return hits


def draw(img, hits, selected=None):
    out = img.copy()
    d = ImageDraw.Draw(out)
    for h in hits:
        color = (0, 90, 255) if h["term"] == selected else (230, 30, 30)
        d.rectangle(h["box"], outline=color, width=3)
    return out


# ---------------------------------------------------------------- 4. 서류 속 기간 계산
DATE = r"(\d{4})\s*[년.\-/]\s*(\d{1,2})\s*[월.\-/]\s*(\d{1,2})\s*일?"
PERIOD = re.compile(DATE + r"\s*(?:부터|~|-|–|에서)\s*" + DATE)


def find_periods(doc_text):
    """'2026년 12월 1일부터 2027년 5월 31일까지' 같은 기간을 찾아 길이를 계산.
    날짜 계산은 AI가 자주 틀리므로 코드가 직접 해서 AI에게 사실로 넘김"""
    from datetime import date

    out = []
    for line in doc_text.splitlines():
        for m in PERIOD.finditer(line):
            try:
                a = date(int(m[1]), int(m[2]), int(m[3]))
                b = date(int(m[4]), int(m[5]), int(m[6]))
            except ValueError:
                continue
            if b <= a:
                continue
            days = (b - a).days + 1
            months = (b.year - a.year) * 12 + (b.month - a.month) + (1 if b.day >= a.day - 1 else 0)
            label = re.sub(r"(은|는|이|가)$", "", line[:m.start()].strip(" :：.0123456789")) or "기간"
            out.append(f"{label}: {a:%Y.%m.%d}~{b:%Y.%m.%d} = {days}일(약 {months}개월), "
                       f"{'1년 이상' if days >= 365 else '1년 미만'}")
    return out


# ---------------------------------------------------------------- 5. 근거 검색 (RAG)
def bigrams(s):
    s = re.sub(r"\s+", "", str(s))
    return {s[i:i + 2] for i in range(len(s) - 1)}


def retrieve(question, all_terms, cat, k=5):
    """질문과 관련된 용어를 사전에서 찾음.
    점수 = 질문에 용어가 그대로 있으면 큰 점수 + 글자 두 개씩 겹치는 정도 + 고른 서류 종류면 가산"""
    q = re.sub(r"\s+", "", question)
    rows = all_terms.to_dict("records")
    # 사전 용어 이름에 너무 흔하게 나오는 두 글자(예: '보험', '건강')는 겹쳐도 관련 근거로 치지 않음
    df = Counter(b for row in rows for b in bigrams(row["term"]))
    common = {b for b, n in df.items() if n >= max(10, len(rows) * 0.01)}
    qb = bigrams(question) - common
    scored = []
    for row in rows:
        key = re.sub(r"\s+", "", row["term"])
        score = 10.0 if key in q else 0.0
        tb = bigrams(row["term"]) - common
        score += 3.0 * len(qb & tb) / max(len(tb), 1)
        score += 1.0 * len(qb & bigrams(row["easy"] or row["official"][:150])) / max(len(qb), 1)
        if score >= 1.5:  # 아무 관련 없는 용어는 버림 (부분 일치면 용어 글자의 절반 이상이 겹쳐야 함)
            scored.append((score + (0.5 if row["category"] == cat else 0), row))
    scored.sort(key=lambda x: -x[0])
    if scored:  # 1등보다 한참 낮은 점수는 근거에서 뺌 (관련 없는 용어가 섞이지 않게)
        scored = [x for x in scored if x[0] >= scored[0][0] * 0.4]
    seen, out = set(), []
    for _, row in scored:
        if row["term"] not in seen:
            seen.add(row["term"])
            out.append(row)
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------- 5. AI 기능
def openai_client():
    key = os.environ.get("OPENAI_API_KEY")
    try:
        key = st.secrets.get("OPENAI_API_KEY", key)
    except Exception:  # secrets.toml 이 없으면 예외가 날 수 있음
        pass
    if not key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=key)


def ai_find_more(doc_text, known, kind):
    """사전에 없는 어려운 용어를 AI가 추가로 고름"""
    client = openai_client()
    if client is None:
        return None
    prompt = (
        f"다음은 '{kind}' 관련 서류에서 읽은 글이다. 처음 이런 서류를 보는 20대가 이해하기 어려운 "
        f"법률·금융·의료 용어를 최대 {8 if len(doc_text) < 1500 else 15}개 골라라. 글에 나온 표기 그대로 쓰고, "
        f"이미 찾은 용어 {sorted(known)} 는 제외한다. 각 용어를 이 서류의 맥락에 맞게 중학생도 "
        '이해할 수 있도록 두 문장 이내로 설명하라. JSON으로만 답하라: {"terms":[{"term":"","easy":""}]}\n\n'
        + doc_text[:6000]
    )
    res = client.chat.completions.create(
        model=MODEL,
        temperature=0,  # 같은 서류면 같은 결과가 나오도록
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    items = json.loads(res.choices[0].message.content).get("terms", [])
    return [{"term": i["term"].strip(), "category": KIND_TO_CAT[kind], "easy": i["easy"],
             "source": AI_SOURCE, "official": "", "plain": ""} for i in items if i.get("term") and i.get("easy")]


@st.cache_data(show_spinner=False)
def ai_easy(term, official, kind, line):
    """공공 사전의 어려운 공식 정의를, 그 정의만 근거로 쉬운 말로 풀어 씀 (출처 근거형 설명)"""
    client = openai_client()
    if client is None:
        return None
    res = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "user", "content": (
            f"아래 [공식 정의]만 근거로 '{term}'의 뜻을 '{kind}' 서류를 처음 보는 20대가 이해하도록 "
            "쉬운 말 두 문장 이내로 설명하라. 공식 정의에 없는 내용은 덧붙이지 마라. "
            "[서류 속 문장]이 있으면 이 서류에서 무슨 뜻인지 한 문장 덧붙여도 된다.\n\n"
            f"[공식 정의]\n{official[:1200]}\n\n[서류 속 문장]\n{line or '(없음)'}")}],
    )
    return res.choices[0].message.content.strip()


def ai_answer(question, doc_text, refs, kind, facts=()):
    client = openai_client()
    if client is None:
        return "OpenAI 키를 설정하면 질문 기능을 쓸 수 있어요."
    ctx = "\n".join(f"- {r['term']}: {r['easy'] or r['official'][:300]} (출처: {r['source']})"
                    for r in refs) or "(관련 용어 없음)"
    res = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                f"너는 '{kind}' 서류를 처음 보는 20대에게 용어와 조항을 쉽게 설명하는 도우미다. "
                "근거는 [서류 내용], [계산된 사실], [용어 사전]뿐이다. 아래 순서대로 생각하고 3~5문장 존댓말(~요, ~습니다)로 답하라.\n"
                "1) 먼저 [서류 내용]에서 질문과 관련된 조항(계약 기간, 금액, 수습, 날짜 등)을 모두 찾는다. "
                "찾았으면 조항 번호와 원문 표현을 따옴표로 그대로 인용한다(숫자·기준일을 바꾸지 마라). "
                "정말 하나도 없을 때만 '이 서류에는 ~에 대한 조항이 없습니다'라고 말한다. 서류 제목·계약 종류도 서류에서 확인되는 사실이다.\n"
                "2) 찾은 조항, [계산된 사실], [용어 사전]의 조건을 맞춰 보고 결론을 낼 수 있으면 첫 문장에 결론부터 말한다. "
                "'만약 ~라면' 같은 가정으로 돌려 말하지 마라.\n"
                "3) 서류만으로 정말 판단할 수 없을 때만 첫 문장을 '이 서류만으로는 알 수 없습니다'로 시작하고, "
                "무엇을 어디서 확인해야 하는지(예: 업무가 단순노무인지, 청약서, 보험회사) 말한다. 결론을 낸 답에 이 문장을 섞지 마라.\n"
                "지킬 것: [용어 사전] 설명은 조건(무엇이 1년 이상인지, 몇 개월 이내인지 등)을 바꾸거나 빼지 말고 옮기되, 서류 원문처럼 따옴표로 인용하지 마라. "
                "용어의 정의는 그 말이 무엇인지만 알려 줄 뿐 이 서류에 해당한다는 뜻이 아니니 일반화해 단정하지 마라. "
                "근거가 부족하면 추측하지 마라. 사기 여부 판정이나 법률·의료 판단은 하지 말고, 필요하면 전문가 상담을 권하라.")},
            {"role": "user", "content": f"[용어 사전]\n{ctx}\n\n[계산된 사실]\n{chr(10).join(facts) or '(없음)'}\n\n[서류 내용]\n{doc_text[:12000]}\n\n[질문]\n{question}"},
        ],
    )
    return res.choices[0].message.content


# ---------------------------------------------------------------- 6. 화면
def show_term_card(term, row, kind="", doc_text=""):
    with st.container(border=True):
        st.subheader(f"🔎 {term}")
        if row.get("easy"):
            st.write(row["easy"])
        elif row.get("official"):
            line = next((l for l in doc_text.splitlines() if term in l), "")
            with st.spinner("공식 정의를 쉽게 풀어 쓰는 중…"):
                easy = ai_easy(term, row["official"], kind, line)
            if easy:
                st.write(easy)
                st.caption("🤖 AI가 아래 공식 정의만 근거로 쉽게 풀어 쓴 설명이에요.")
            else:
                st.write(row["official"])
        if row.get("plain"):
            st.caption(f"쉬운 말로는 '{row['plain']}'(이)라고 해요.")
        if row.get("official") and row.get("easy") != row.get("official"):
            with st.expander("공식 정의 원문 보기"):
                st.write(row["official"])
        if row["source"] == AI_SOURCE:
            st.caption("⚠️ 출처: AI가 만든 설명이에요. 공식 사전에는 없는 용어라 한 번 더 확인해 주세요.")
        else:
            st.caption(f"출처: {row['source']}" + (f" · 근거: {row['basis']}" if row.get("basis") else ""))
        st.caption(DISCLAIMER)


def main():
    from streamlit_image_coordinates import streamlit_image_coordinates

    st.set_page_config(page_title="찰칵사전", page_icon="📄", layout="centered")
    all_terms = load_terms()
    st.title("📄 찰칵사전")
    st.caption("어려운 서류 용어, 찍으면 풀려요")

    kind = st.radio("서류 종류", list(KIND_TO_CAT), index=1, horizontal=True)
    cat = KIND_TO_CAT[kind]
    terms = terms_for(all_terms, cat)
    st.info("🔒 이름·주민번호·주소·전화번호는 가린 뒤 올려 주세요. 올린 파일은 저장하지 않아요.")

    tab_cam, tab_file = st.tabs(["📷 사진 찍기", "📁 사진·PDF 올리기"])
    with tab_cam:
        shot = st.camera_input("서류가 화면에 꽉 차게, 기울지 않게 찍어 주세요")
    with tab_file:
        up = st.file_uploader("파일 선택", type=["png", "jpg", "jpeg", "pdf"])

    src = shot or up
    if src is None:
        st.stop()

    data = src.getvalue()
    page_no = 0
    if src.name.lower().endswith(".pdf"):
        import fitz

        n_pages = fitz.open(stream=data, filetype="pdf").page_count
        if n_pages > 1:
            page_no = st.number_input("쪽", 1, n_pages, 1) - 1
        img, lines, has_text, quality = read_pdf(data, page_no)
    else:
        img, lines, quality = read_image(data)
        has_text = False

    # 다른 파일이나 다른 쪽을 열면 이전 선택·AI 결과·대화를 지움
    doc_id = f"{hashlib.md5(data).hexdigest()}:{page_no}"
    if st.session_state.get("doc_id") != doc_id:
        for k in ("selected", "extra", "chat"):
            st.session_state.pop(k, None)
        st.session_state["doc_id"] = doc_id

    if not has_text and (len(lines) < 3 or quality < 0.4):
        st.warning("글자가 잘 안 읽혔어요. 밝은 곳에서 서류가 화면에 꽉 차게, 기울지 않게 다시 찍어 주세요.")

    finder = (lambda df: find_terms_pdf(data, page_no, df)) if has_text else (lambda df: find_terms(lines, df, fuzzy=True))
    doc_text = "\n".join(t for t, _ in lines)
    # 챗봇용 글: 글자 정보가 있는 PDF는 모든 쪽을 넘김(보고 있는 쪽을 맨 앞에).
    # 예) 등기부 2쪽에 있는 근저당도 1쪽을 보면서 물어볼 수 있게. 스캔 PDF·사진은 지금 쪽만.
    chat_text = doc_text
    if has_text and src.name.lower().endswith(".pdf"):
        _pdf = fitz.open(stream=data, filetype="pdf")
        if _pdf.page_count > 1:
            order = [page_no] + [i for i in range(_pdf.page_count) if i != page_no]
            chat_text = "\n\n".join(f"[{i + 1}쪽]\n{_pdf[i].get_text().strip()}" for i in order)
    hits = finder(terms)

    # AI로 더 찾기 (버튼을 눌렀을 때만 → API 비용 절약)
    if st.button("🤖 AI로 어려운 용어 더 찾기"):
        with st.spinner("AI가 서류를 읽는 중…"):
            extra = ai_find_more(doc_text, {h["term"] for h in hits}, kind)
        if extra is None:
            st.warning("OpenAI 키를 설정하면 AI로 용어를 더 찾을 수 있어요.")
        else:
            st.session_state["extra"] = extra
    if st.session_state.get("extra"):
        hits += [h for h in finder(pd.DataFrame(st.session_state["extra"]))
                 if not any(overlaps(h["box"], o["box"]) for o in hits)]

    if hits:
        st.write(f"**{len(hits)}개** 용어를 찾았어요. 빨간 상자를 눌러 보세요.")
    else:
        st.write("사전에 있는 용어를 찾지 못했어요. 'AI로 어려운 용어 더 찾기'를 눌러 보거나 아래에 물어보세요.")
    selected = st.session_state.get("selected")
    shown = draw(img, hits, selected)
    try:  # 화면(휴대폰) 폭에 맞춰 이미지를 줄여서 보여 줌
        click = streamlit_image_coordinates(shown, key="doc", use_column_width="always")
    except TypeError:  # 구버전 컴포넌트
        click = streamlit_image_coordinates(shown, key="doc", width=img.width)

    if click:
        # 줄여서 보여 준 만큼 클릭 좌표를 원래 이미지 크기로 되돌림
        scale = img.width / click["width"] if click.get("width") else 1
        cx, cy = click["x"] * scale, click["y"] * scale
        for h in hits:
            x0, y0, x1, y1 = h["box"]
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                if st.session_state.get("selected") != h["term"]:
                    st.session_state["selected"] = h["term"]
                    st.rerun()

    # 상자를 누르기 어려울 때를 위한 버튼 목록
    cols = st.columns(4)
    for i, name in enumerate(dict.fromkeys(h["term"] for h in hits)):
        if cols[i % 4].button(name, key=f"b{i}"):
            st.session_state["selected"] = name
            st.rerun()

    if selected:
        row = next((h["row"] for h in hits if h["term"] == selected), None)
        if row:
            show_term_card(selected, row, kind, doc_text)

    # ---- 질문 챗봇: 사전에서 근거를 찾아(RAG) 서류 내용과 함께 AI에게 넘김
    st.divider()
    chat = st.session_state.setdefault("chat", [])
    for msg in chat:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])
            if msg.get("refs"):
                st.caption("근거: " + " · ".join(f"{r['term']}({r['source']})" for r in msg["refs"]))

    q = st.chat_input("표시되지 않은 것도 물어보세요 (예: 수습기간에도 최저임금 받아?)")
    if q:
        refs = retrieve(q, all_terms, cat)  # 화면에 '근거'로 보여 줄 용어
        ctx_rows = refs + [h["row"] for h in hits if h["row"] not in refs][:10]  # AI 참고용 (서류 속 용어 포함)
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("답을 찾는 중…"):
                ans = ai_answer(q, chat_text, ctx_rows, kind, find_periods(chat_text))
            st.write(ans)
            used = refs
            if used:
                st.caption("근거: " + " · ".join(f"{r['term']}({r['source']})" for r in used))
            st.caption(DISCLAIMER)
        chat += [{"role": "user", "text": q},
                 {"role": "assistant", "text": ans + "\n\n" + DISCLAIMER, "refs": refs[:5]}]


if __name__ == "__main__":
    main()
