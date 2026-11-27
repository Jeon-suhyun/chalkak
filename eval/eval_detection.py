"""
찰칵사전 검증: 서류 10건에서 정답 용어를 몇 개 표시하는지 측정

실행 (chalkak 폴더에서):
    C:\\venvs\\chalkak\\python.exe eval\\eval_detection.py          (사전 매칭만)
    C:\\venvs\\chalkak\\python.exe eval\\eval_detection.py --ai     (사전 + 'AI로 더 찾기', OpenAI 키 필요, 몇 센트)
    ... --heldout            (확인 측정: 보완 과정에서 보지 않은 서류, eval/gold_heldout.json)

- 앱(app.py)과 똑같은 함수로 용어를 찾는다
- --ai 를 주면 앱의 'AI로 어려운 용어 더 찾기'와 같은 방식으로 AI가 고른 용어를 더해서 측정
- 결과: eval/eval_results(_ai).md (표), eval/eval_results(_ai).csv
- 통과 기준: 그 서류의 정답 용어를 '빠짐없이' 표시 (예선 자료 10쪽 기준)
"""
import csv
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import app  # noqa: E402  (main()은 실행되지 않음)

DOCS = os.path.join(ROOT, "테스트자료")
GOLD = os.path.join(ROOT, "eval", "gold_terms.json")
GOLD_HELDOUT = os.path.join(ROOT, "eval", "gold_heldout.json")


def norm(s):
    return re.sub(r"[\s·‧ㆍ・]", "", s)


CAT_TO_KIND = {v: k for k, v in app.KIND_TO_CAT.items()}


def detect(case, terms, use_ai=False):
    """앱과 같은 순서: 글자 읽기 → 사전 매칭 → (선택) AI가 고른 용어를 서류에서 다시 찾아 표시"""
    path = os.path.join(DOCS, case["file"])
    data = open(path, "rb").read()
    units, quality = [], None  # (finder, 글자 목록) 쪽마다
    if case["pages"]:  # PDF
        for p in case["pages"]:
            img, lines, has_text, q = app.read_pdf(data, p - 1)
            finder = (lambda df, p=p: app.find_terms_pdf(data, p - 1, df)) if has_text \
                else (lambda df, lines=lines: app.find_terms(lines, df, fuzzy=True))
            units.append((finder, lines))
            quality = q if quality is None else min(quality, q)
    else:  # 사진
        img, lines, q = app.read_image(data)
        units.append((lambda df, lines=lines: app.find_terms(lines, df, fuzzy=True), lines))
        quality = q
    found = []
    for finder, _ in units:
        found += [h["term"] for h in finder(terms)]
    ai_added = []
    if use_ai:
        doc_text = "\n".join(t for _, lines in units for t, _ in lines)
        extra = app.ai_find_more(doc_text, set(found), CAT_TO_KIND[case["category"]]) or []
        if extra:
            import pandas as pd
            df = pd.DataFrame(extra)
            for finder, _ in units:
                ai_added += [h["term"] for h in finder(df)]  # 서류에서 위치를 찾은 것만 (앱과 동일)
    return sorted(set(found) | set(ai_added)), quality, sorted(set(ai_added) - set(found))


def matched(gold, found):
    """정답 g를 찾은 것으로 보는 경우: 같은 용어이거나, 찾은 용어가 g를 포함(예: 정답 '근저당권' ← 찾은 '근저당권설정')"""
    g = norm(gold)
    return any(norm(t) == g or g in norm(t) for t in found)


def main():
    use_ai = "--ai" in sys.argv
    if use_ai and app.openai_client() is None:
        sys.exit("OpenAI 키를 찾을 수 없어요 (.streamlit/secrets.toml 확인)")
    heldout = "--heldout" in sys.argv
    suffix = ("_heldout" if heldout else "") + ("_ai" if use_ai else "")
    gold_path = GOLD_HELDOUT if heldout else GOLD
    cases = json.load(open(gold_path, encoding="utf-8"))["cases"]
    all_terms = app.load_terms()
    rows, total_g, total_hit, passed = [], 0, 0, 0
    for c in cases:
        t0 = time.time()
        terms = app.terms_for(all_terms, c["category"])
        try:
            found, q, ai_added = detect(c, terms, use_ai)
            err = ""
        except Exception as e:  # 한 건이 실패해도 나머지는 계속
            found, q, ai_added, err = [], None, [], f"{type(e).__name__}: {e}"
        hit = [g for g in c["gold"] if matched(g, found)]
        miss = [g for g in c["gold"] if g not in hit]
        ok = not miss and not err
        passed += ok
        total_g += len(c["gold"])
        total_hit += len(hit)
        extra = [t for t in found if not any(matched(g, [t]) for g in c["gold"])]
        rows.append({
            "id": c["id"], "서류": c["name"], "종류": c["kind"],
            "정답": len(c["gold"]), "찾음": len(hit), "재현율": f"{len(hit) / len(c['gold']):.0%}",
            "통과": "O" if ok else "X", "놓친 용어": ", ".join(miss) or "-",
            "정답 외 표시 수": len(extra), "AI가 더한 용어": ", ".join(ai_added) or "-", "인식 신뢰도": "" if q is None else f"{q:.2f}",
            "시간(초)": f"{time.time() - t0:.1f}", "오류": err,
        })
        print(f"[{c['id']}] {c['name']}: {len(hit)}/{len(c['gold'])} {'통과' if ok else '미통과'}"
              + (f"  놓침: {', '.join(miss)}" if miss else "") + (f"  오류: {err}" if err else ""))

    os.makedirs("eval", exist_ok=True)
    with open(f"eval/eval_results{suffix}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    mode = ("사전 + AI 추가 추출" if use_ai else "사전 매칭만") + (" · 확인 측정(held-out)" if heldout else "")
    summary = (f"[{mode}] 통과 {passed}/{len(cases)}건 · 정답 용어 재현율 {total_hit}/{total_g} "
               f"({total_hit / total_g:.0%}) · 용어 DB {len(all_terms)}개 · 측정 {time.strftime('%Y-%m-%d %H:%M')}")
    cols = ["id", "서류", "종류", "찾음", "정답", "재현율", "통과", "놓친 용어", "정답 외 표시 수", "인식 신뢰도"]
    if use_ai:
        cols.insert(8, "AI가 더한 용어")
    md = ["# 찰칵사전 용어 표시 검증 결과", "", f"**{summary}**", "",
          "- 통과 기준: 서류의 정답 용어를 빠짐없이 표시 (예선 계획: 10건 중 9건 이상)",
          f"- 정답 용어는 평가 전에 서류 본문만 보고 정해 `eval/{os.path.basename(gold_path)}`에 고정",
          f"- 측정 모드: {mode}", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    open(f"eval/eval_results{suffix}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + summary)
    print(f"결과 파일: eval/eval_results{suffix}.md, eval/eval_results{suffix}.csv")


if __name__ == "__main__":
    main()
