import streamlit as st
import pandas as pd
import json
import uuid
from datetime import datetime
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection

# =============================================================================
# [PART 1] 시스템 설정 및 데이터 로직 (구글 시트 연동형)
# =============================================================================

st.set_page_config(page_title="MNK 성과관리 시스템", layout="wide")

# 구글 시트 연결 (Secrets 설정을 기반으로 자동으로 연결됩니다)
conn = st.connection("gsheets", type=GSheetsConnection)

YEAR_OPTIONS = [str(y) for y in range(datetime.now().year + 1, datetime.now().year - 5, -1)]
PART_ORDER = ["마케팅", "디자인컷", "콘티", "모델링", "애니메이션", "편집"]

if 'opened_gid' not in st.session_state:
    st.session_state.opened_gid = None
if 'temp_workers' not in st.session_state:
    st.session_state.temp_workers = []

# 데이터 로드/저장 함수
def load_data():
    try:
        df = conn.read(worksheet="Data", ttl=0)
        return df.dropna(how="all")
    except:
        return pd.DataFrame()

def load_config():
    try:
        df_cfg = conn.read(worksheet="Config", ttl=0)
        config_raw = df_cfg.iloc[0].to_dict()
        return {
            "diff_weights": json.loads(config_raw["diff_weights"]),
            "cont_weights": json.loads(config_raw["cont_weights"]),
            "penalty_rate": float(config_raw["penalty_rate"]),
            "main_color": config_raw["main_color"]
        }
    except:
        return {
            "diff_weights": {"S": 2.0, "A": 1.5, "B": 1.0, "C": 0.7},
            "cont_weights": {"상": 1.2, "중": 1.0, "하": 0.8},
            "penalty_rate": 0.05,
            "main_color": "#00FFD1"
        }

def save_and_stay(df, gid=None):
    # 구글 시트 'Data' 워크시트 업데이트
    conn.update(worksheet="Data", data=df)
    st.session_state.opened_gid = gid
    st.cache_data.clear()
    st.rerun()

def save_config(config):
    cfg_df = pd.DataFrame([{
        "diff_weights": json.dumps(config["diff_weights"]),
        "cont_weights": json.dumps(config["cont_weights"]),
        "penalty_rate": config["penalty_rate"],
        "main_color": config["main_color"]
    }])
    conn.update(worksheet="Config", data=cfg_df)
    st.cache_data.clear()

config = load_config()
all_df = load_data()

# 1-4. 핵심 점수 계산 엔진 (수정됨: 디자인컷 균등 배분 로직)
def run_score_engine(project_df, p_diff, p_total_edits, cfg):
    df = project_df.copy()
    if len(df) == 0: return df
    
    # 0. 데이터 초기화 및 보정 (이전 계산값 삭제 및 NaN 처리)
    df['공통수정분'] = 0.0
    if '제외횟수' not in df.columns:
        df['제외횟수'] = 0.0
    df['제외횟수'] = pd.to_numeric(df['제외횟수'], errors='coerce').fillna(0)
    df['수정횟수'] = pd.to_numeric(df['수정횟수'], errors='coerce').fillna(0)

    # [로직 A/B/C] 점수입력 산정 (기존 로직 유지)
    mkt_mask = df['파트'] == "마케팅"
    mkt_sum = df.loc[mkt_mask, '점수입력'].sum()
    design_mask = df['파트'] == "디자인컷"
    design_count = design_mask.sum()
    rem_pool = max(0, 100.0 - mkt_sum)

    if design_count > 0:
        design_unit = rem_pool / design_count
        df.loc[design_mask, '점수입력'] = round(design_unit, 2)
    else:
        prod_mask = ~df['파트'].isin(["마케팅", "디자인컷"])
        if prod_mask.sum() > 0:
            total_cont_w = sum([cfg["cont_weights"].get(row['기여도'], 1.0) for _, row in df[prod_mask].iterrows()])
            unit = rem_pool / total_cont_w if total_cont_w > 0 else 0
            for idx in df[prod_mask].index:
                cw = cfg["cont_weights"].get(df.at[idx, '기여도'], 1.0)
                df.at[idx, '점수입력'] = round(unit * cw, 2)

    # [로직 D] 사용자 요청 정밀 공통수정 배분 수식 (최종형)
    total_n = len(df)
    p_total_edits = float(p_total_edits)
    
    # 1단계: 제외자(Excluders) 몫 먼저 계산 및 할당
    exclude_mask = df['제외횟수'] > 0
    non_exclude_mask = df['제외횟수'] == 0
    
    sum_allocated_to_excluders = 0.0
    
    for idx in df[exclude_mask].index:
        my_ex = df.at[idx, '제외횟수']
        # 수식: (총수정 - 본인제외분) / 전체인원
        my_share = max(0, (p_total_edits - my_ex) / total_n)
        df.at[idx, '공통수정분'] = my_share
        sum_allocated_to_excluders += my_share

    # 2단계: 미제외자(Non-Excluders) 몫 계산
    # 수식: (총수정 - 제외자들이 가져간 합계) / 미제외 인원수
    remaining_pool = max(0, p_total_edits - sum_allocated_to_excluders)
    non_exclude_count = non_exclude_mask.sum()
    
    if non_exclude_count > 0:
        non_exclude_share = remaining_pool / non_exclude_count
        df.loc[non_exclude_mask, '공통수정분'] = non_exclude_share

    # [로직 E] 최종 점수 및 감점 확정
    for idx in df.index:
        raw_val = df.at[idx, '점수입력']
        # 최종 개인별 수정 책임 = (계산된 공통수정분) + (개인 수정횟수)
        total_resp = max(0, df.at[idx, '공통수정분'] + df.at[idx, '수정횟수'])
        
        df.at[idx, '기본점수'] = round(raw_val, 2)
        penalty_val = round(raw_val * (total_resp * cfg["penalty_rate"]), 2)
        df.at[idx, '감점점수'] = penalty_val
        
        final_calc = max(0, raw_val - penalty_val) * cfg["diff_weights"].get(p_diff, 1.0)
        df.at[idx, '최종점수'] = round(final_calc, 2)
        # 화면 표시를 위해 소수점 정리
        df.at[idx, '공통수정분'] = round(df.at[idx, '공통수정분'], 4)

    return df

config = load_config()
all_df = load_data()

# =============================================================================
# [PART 2] UI 스타일 정의 (CSS)
# =============================================================================
st.markdown(f"""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * {{ font-family: '{config.get('font_family', 'Pretendard')}', sans-serif; }}
    .stApp {{ background-color: #1E1E26; color: #f0f2f6; }}
    .metric-card {{ background-color: #2D2D3A; padding: 15px; border-radius: 10px; border-left: 5px solid {config.get('main_color', '#E84D4D')}; margin-bottom: 10px; }}
    .metric-label {{ font-size: 12px; color: #aaa; }}
    .metric-value {{ font-weight: 700; font-size: 20px; color: {config.get('main_color', '#E84D4D')}; }}
    .score-style {{ color: {config.get('main_color', '#E84D4D')}; font-weight: 800; font-size: 15px; text-align: center; }}
    .header-style {{ background-color: #262730; padding: 10px; border-radius: 5px; font-weight: bold; font-size: 14px; text-align: center; border-bottom: 2px solid #444; }}
    hr {{ border: 0; height: 1px; background: #333; margin: 20px 0; }}
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# [PART 3] 메인 화면 구성 (TABS)
# =============================================================================
tabs = st.tabs(["📝 작업 등록", "🗂️ 프로젝트 관리", "📈 통계 대시보드", "⚙️ 설정"])

# -----------------------------------------------------------------------------
# [TAB 0] 작업 등록
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("1️⃣ 프로젝트 기본 정보")
    with st.container(border=True):
        c_y, c_m, c1, c2, c3, c4 = st.columns([1, 0.8, 1.5, 0.8, 1.2, 0.8])
        p_year = c_y.selectbox("연도 설정", YEAR_OPTIONS, key="reg_y")
        p_month = c_m.selectbox("월 설정", list(range(1, 13)), index=datetime.now().month-1, key="reg_m")
        p_name = c1.text_input("프로젝트 명 설정", placeholder="예: 엠엔케이", key="reg_n")
        p_diff = c2.selectbox("난이도 설정", list(config["diff_weights"].keys()), index=2, key="reg_d")
        p_cat = c3.text_input("분류 설정", placeholder="예: 영상 혹은 디자인컷", key="reg_c")
        p_edits = c4.number_input("전체 수정횟수 설정", min_value=0, step=1, key="reg_e")
    
    st.write("")
    st.subheader("2️⃣ 프로젝트 참여 작업자 기본 정보")
    with st.container(border=True):
        w1, w2, w3, w4 = st.columns([1.5, 1.5, 1, 1.5])
        part = w1.selectbox("파트 선택", PART_ORDER, index=0, key="reg_wp")
        name = w2.text_input("작업자 명", placeholder="이름 기입", key="reg_wn")
        is_special = part in ["마케팅", "디자인컷"]
        cont = w3.selectbox("기여도", ["상", "중", "하"], index=1, disabled=is_special, key="reg_wc")
        m_score = w4.number_input("마케팅 점수기입", min_value=0.0, disabled=(part != "마케팅"), key="reg_ms")
        
        if st.button("➕ 명단에 추가", use_container_width=True):
            if name:
                new_entry = {
                    "이름": name, "파트": part, 
                    "기여도": "-" if is_special else cont, 
                    "점수입력": m_score if part=="마케팅" else 0.0, 
                    "수정횟수": 0, "worker_id": str(uuid.uuid4())
                }
                st.session_state.temp_workers.append(new_entry)
                st.rerun()
            else: st.warning("작업자 이름을 입력해주세요.")

    if st.session_state.temp_workers:
        st.write("---")
        st.markdown("### 📋 현재 추가된 명단")
        t_df = pd.DataFrame(st.session_state.temp_workers)
        st.dataframe(t_df[["파트", "이름", "기여도", "점수입력"]], use_container_width=True, hide_index=True)
        
        c_del, c_save = st.columns([1, 4])
        if c_del.button("🔄 목록 초기화"):
            st.session_state.temp_workers = []
            st.rerun()
        if c_save.button("🚀 프로젝트 최종 저장 및 점수 발행", type="primary", use_container_width=True):
            final_df = run_score_engine(t_df, p_diff, p_edits, config)
            gid = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{p_name}"
            final_df[['연도','월','프로젝트명','난이도','분류','프로젝트_수정횟수','group_id','등록일시']] = [
                p_year, p_month, p_name, p_diff, p_cat, p_edits, gid, datetime.now().strftime("%Y-%m-%d %H:%M")
            ]
            all_df = pd.concat([load_data(), final_df], ignore_index=True)
            st.session_state.temp_workers = []
            save_and_stay(all_df, gid)

# -----------------------------------------------------------------------------
# [TAB 1] 프로젝트 관리 (수정됨: 월 변경 기능 추가)
# -----------------------------------------------------------------------------
with tabs[1]:
    if not all_df.empty:
        st.subheader("🔍 프로젝트 통합 검색 및 필터")
        def get_chosung(text):
            CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
            result = ""
            for char in str(text):
                if '가' <= char <= '힣':
                    char_code = ord(char) - ord('가')
                    result += CHOSUNG_LIST[char_code // 588]
                else: result += char
            return result

        with st.container(border=True):
            search_query = st.text_input("🔎 검색 (프로젝트명 또는 작업자 이름)", placeholder="초성 검색 가능", key="pm_search_main")
            f1, f2, f3, f4 = st.columns(4)
            sel_y = f1.selectbox("📅 연도", ["전체"] + sorted(all_df['연도'].unique().tolist(), reverse=True), key="mg_f_y")
            sel_d = f2.selectbox("📊 난이도", ["전체"] + list(config["diff_weights"].keys()), key="mg_f_d")
            sel_q = f3.selectbox("📆 분기", ["전체", "1분기", "2분기", "3분기", "4분기"], key="mg_f_q")
            cat_list = sorted(all_df['분류'].dropna().unique().astype(str).tolist())
            sel_c = f4.selectbox("📁 분류", ["전체"] + cat_list, key="mg_f_c")

        filtered_df = all_df.copy()
        if sel_y != "전체": filtered_df = filtered_df[filtered_df['연도'] == sel_y]
        if sel_d != "전체": filtered_df = filtered_df[filtered_df['난이도'] == sel_d]
        if sel_q != "전체": 
            filtered_df['temp_q'] = filtered_df['월'].apply(lambda x: f"{(int(x)-1)//3 + 1}분기")
            filtered_df = filtered_df[filtered_df['temp_q'] == sel_q]
        if sel_c != "전체": filtered_df = filtered_df[filtered_df['분류'] == sel_c]

        if search_query:
            query_qs = get_chosung(search_query.replace(" ", ""))
            matched_gids = []
            for gid in filtered_df['group_id'].unique():
                g_rows = all_df[all_df['group_id'] == gid]
                combined = (str(g_rows.iloc[0]['프로젝트명']) + "".join(g_rows['이름'].astype(str))).replace(" ", "")
                if search_query.replace(" ", "").lower() in combined.lower() or query_qs in get_chosung(combined):
                    matched_gids.append(gid)
            filtered_df = filtered_df[filtered_df['group_id'].isin(matched_gids)]

        st.write(f"✅ 검색 결과: {len(filtered_df['group_id'].unique())}건")

        for gid in filtered_df['group_id'].unique():
            g_df = all_df[all_df['group_id'] == gid].copy()
            g_df['파트'] = pd.Categorical(g_df['파트'], categories=PART_ORDER, ordered=True)
            g_df = g_df.sort_values('파트')
            first = g_df.iloc[0]
            is_expanded = st.session_state.get('opened_gid') == gid
            
            with st.expander(f"📂 [{first['연도']}/{first['월']}월] {first['프로젝트명']} | {first['난이도']} | {first['분류']}", expanded=is_expanded):
                st.markdown("##### ⚙️ 프로젝트 정보 설정")
                with st.container(border=True):
                    mc = st.columns([3, 1.2, 1.0, 1.1, 1.2, 1.2, 1, 0.5])
                    en = mc[0].text_input("프로젝트명", value=first['프로젝트명'], key=f"en_{gid}")
                    ey = mc[1].selectbox("연도", YEAR_OPTIONS, index=YEAR_OPTIONS.index(str(first['연도'])), key=f"ey_{gid}")
                    
                    # [월 변경 기능 반영]
                    month_list = list(range(1, 13))
                    em = mc[2].selectbox("월", month_list, index=month_list.index(int(first['월'])), key=f"em_{gid}")
                    
                    ed = mc[3].selectbox("난이도", list(config["diff_weights"].keys()), index=list(config["diff_weights"].keys()).index(first['난이도']), key=f"ed_{gid}")
                    ec = mc[4].text_input("분류", value=first['분류'], key=f"ec_{gid}")
                    ee = mc[5].number_input("전체 수정횟수", min_value=0, value=int(first['프로젝트_수정횟수']), key=f"ee_{gid}")
                    
                    mc[6].markdown('<div style="margin-top:28px;"></div>', unsafe_allow_html=True)
                    is_del_ok = mc[7].checkbox("🗑️", key=f"del_chk_{gid}", label_visibility="collapsed")
                    
                    if mc[6].button("삭제", key=f"del_group_{gid}", disabled=not is_del_ok, use_container_width=True):
                        all_df = all_df[all_df['group_id'] != gid]
                        save_and_stay(all_df, None)

                    if st.button("💾 프로젝트 업데이트", key=f"up_btn_{gid}", use_container_width=True, type="primary"):
                        mask = all_df['group_id'] == gid
                        all_df.loc[mask, ['프로젝트명','연도','월','난이도','분류','프로젝트_수정횟수']] = [en, ey, em, ed, ec, ee]
                        st.session_state['opened_gid'] = gid
                        all_df.update(run_score_engine(all_df[mask], ed, ee, config))
                        save_and_stay(all_df, gid)

                st.divider()
                st.markdown("##### 👥 참여 작업자 관리")
                
                # 헤더 설정 (11개 컬럼)
                cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                headers = ["파트", "이름", "점수/기여도", "기본점수", "감점", "최종점수", "공통수정", "제외횟수", "개인수정", "수정조절", "삭제"]
                for col, text in zip(cols, headers):
                    col.markdown(f'<div class="header-style" style="font-size:11px; text-align:center;">{text}</div>', unsafe_allow_html=True)
                
                for _, row in g_df.iterrows():
                    wid = row['worker_id']
                    target_mask = all_df['worker_id'] == wid
                    if not target_mask.any(): continue
                    ridx = all_df[target_mask].index[0]
                    
                    r = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                    
                    # 1. 파트 및 이름 변경
                    new_p = r[0].selectbox("P", PART_ORDER, index=PART_ORDER.index(row['파트']), key=f"p_{wid}", label_visibility="collapsed")
                    new_n = r[1].text_input("N", value=row['이름'], key=f"n_{wid}", label_visibility="collapsed")
                    
                    # 2. 점수입력 및 기여도 입력
                    need_update = False
                    if row['파트'] == "마케팅":
                        new_val = r[2].number_input("V", value=float(row['점수입력']), key=f"v_{wid}", label_visibility="collapsed")
                        if new_val != row['점수입력']:
                            all_df.at[ridx, '점수입력'] = new_val
                            need_update = True
                    elif row['파트'] == "디자인컷":
                        r[2].markdown('<div style="text-align:center; margin-top:8px; font-size:12px; color:#aaa;">자동배분</div>', unsafe_allow_html=True)
                    else:
                        cl = ["상", "중", "하"]
                        current_c = row['기여도'] if row['기여도'] in cl else "중"
                        new_c = r[2].selectbox("C", cl, index=cl.index(current_c), key=f"c_{wid}", label_visibility="collapsed")
                        if new_c != row['기여도']:
                            all_df.at[ridx, '기여도'] = new_c
                            need_update = True

                    if new_p != row['파트'] or new_n != row['이름']:
                        all_df.at[ridx, '파트'] = new_p
                        all_df.at[ridx, '이름'] = new_n
                        need_update = True

                    # 3. 점수 정보 출력
                    r[3].markdown(f'<div class="score-style">{row["기본점수"]:,.1f}</div>', unsafe_allow_html=True)
                    r[4].markdown(f'<div class="score-style">-{row["감점점수"]:,.1f}</div>', unsafe_allow_html=True)
                    r[5].markdown(f'<div class="score-style" style="font-size:15px; color:#00FFD1;">{row["최종점수"]:,.1f}</div>', unsafe_allow_html=True)
                    
                    # 4. 공통수정분 표시
                    comm_edits = row.get("공통수정분", 0)
                    r[6].markdown(f'<div style="text-align:center; margin-top:8px; font-size:12px; color:#888;">{comm_edits:,.2f}회</div>', unsafe_allow_html=True)

                    # 5. [중요] 제외횟수 드롭다운 및 데이터 꼬임 방지 로직
                    max_proj_edits = int(first['프로젝트_수정횟수'])
                    exclude_options = list(range(max_proj_edits + 1))
                    
                    val_ex = row.get('제외횟수', 0)
                    curr_ex = int(val_ex) if pd.notna(val_ex) else 0
                    if curr_ex > max_proj_edits: curr_ex = 0
                    
                    new_ex = r[7].selectbox("EX", exclude_options, index=exclude_options.index(curr_ex), key=f"ex_{wid}", label_visibility="collapsed")
                    
                    if new_ex != curr_ex:
                        # 수정 전의 깨끗한 상태를 유지하기 위해 DF의 해당 값만 먼저 변경
                        all_df.at[ridx, '제외횟수'] = float(new_ex)
                        need_update = True

                    # 변경사항이 있을 경우 엔진 가동 및 저장 (동기화 핵심)
                    if need_update:
                        this_project_mask = all_df['group_id'] == gid
                        project_subset = all_df[this_project_mask].copy()
                        calculated_subset = run_score_engine(project_subset, ed, ee, config)
                        all_df.loc[this_project_mask, :] = calculated_subset
                        save_and_stay(all_df, gid)

                    # 6. 개인수정 횟수 및 조절 버튼
                    r[8].markdown(f'<div style="text-align:center; margin-top:8px; font-size:14px; font-weight:bold; color:#E84D4D;">{row["수정횟수"]}회</div>', unsafe_allow_html=True)

                    btn_c = r[9].columns([1, 1])
                    if btn_c[0].button("➖", key=f"mn_{wid}", use_container_width=True):
                        all_df.at[ridx, '수정횟수'] = max(0, row['수정횟수'] - 1)
                        # 즉시 그룹 재계산 반영
                        this_project_mask = all_df['group_id'] == gid
                        all_df.loc[this_project_mask, :] = run_score_engine(all_df[this_project_mask], ed, ee, config)
                        save_and_stay(all_df, gid)
                    if btn_c[1].button("➕", key=f"pl_{wid}", use_container_width=True):
                        all_df.at[ridx, '수정횟수'] += 1
                        # 즉시 그룹 재계산 반영
                        this_project_mask = all_df['group_id'] == gid
                        all_df.loc[this_project_mask, :] = run_score_engine(all_df[this_project_mask], ed, ee, config)
                        save_and_stay(all_df, gid)

                    # 7. 삭제
                    del_c = r[10].columns([0.4, 0.6])
                    is_row_del = del_c[0].checkbox("", key=f"cw_{wid}", label_visibility="collapsed")
                    if del_c[1].button("🗑️", key=f"dw_{wid}", disabled=not is_row_del, use_container_width=True):
                        all_df = all_df[all_df['worker_id'] != wid]
                        # 삭제 후 남은 인원들에 대해 다시 계산
                        remaining_mask = all_df['group_id'] == gid
                        if remaining_mask.any():
                            all_df.loc[remaining_mask, :] = run_score_engine(all_df[remaining_mask], ed, ee, config)
                        save_and_stay(all_df, gid)

                st.markdown("---")
                st.markdown("➕ **중간 투입 작업자 추가**")
                with st.container(border=True):
                    ac1, ac2, ac3, ac4 = st.columns([1, 1, 1, 1])
                    new_worker_p = ac1.selectbox("파트 선택", PART_ORDER, key=f"new_p_{gid}")
                    new_worker_n = ac2.text_input("이름 입력", placeholder="작업자 이름", key=f"new_n_{gid}")
                    if new_worker_p == "마케팅":
                        new_worker_v = ac3.number_input("점수입력", value=0.0, key=f"new_v_{gid}")
                        new_worker_c = "-"
                    elif new_worker_p == "디자인컷":
                        ac3.markdown('<div style="margin-top:35px; color:#aaa; font-size:12px;">마케팅 제외 자동균등배분</div>', unsafe_allow_html=True)
                        new_worker_v = 0.0
                        new_worker_c = "-"
                    else:
                        new_worker_c = ac3.selectbox("기여도 설정", ["상", "중", "하"], index=1, key=f"new_c_{gid}")
                        new_worker_v = 0.0
                    
                    ac4.markdown('<div style="margin-top:28px;"></div>', unsafe_allow_html=True)
                    if ac4.button("현재 프로젝트에 추가", key=f"add_btn_{gid}", use_container_width=True, type="secondary"):
                        if new_worker_n:
                            new_row = {
                                'worker_id': str(uuid.uuid4()), 'group_id': gid, '프로젝트명': first['프로젝트명'],
                                '연도': first['연도'], '월': first['월'], '난이도': first['난이도'], '분류': first['분류'],
                                '프로젝트_수정횟수': first['프로젝트_수정횟수'], '파트': new_worker_p, '이름': new_worker_n,
                                '기여도': new_worker_c, '점수입력': new_worker_v, '수정횟수': 0, '등록일시': first['등록일시']
                            }
                            all_df = pd.concat([all_df, pd.DataFrame([new_row])], ignore_index=True)
                            all_df.update(run_score_engine(all_df[all_df['group_id'] == gid], ed, ee, config))
                            save_and_stay(all_df, gid)
                        else: st.error("이름을 입력해야 추가가 가능합니다.")

# -----------------------------------------------------------------------------
# [TAB 2] 통계 대시보드
# -----------------------------------------------------------------------------
with tabs[2]:
    if all_df.empty:
        st.info("📊 통계를 생성할 데이터가 없습니다.")
    else:
        # 1. 필터링 (정렬 및 기본 설정)
        dff = all_df.copy()
        dff['분기'] = dff['월'].apply(lambda x: f"{(int(x)-1)//3 + 1}분기")
        st.subheader("🔍 데이터 필터링")
        with st.container(border=True):
            f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
            sel_y_st = f1.selectbox("📅 연도", ["전체"] + sorted(dff['연도'].unique().tolist(), reverse=True), key="stat_y")
            cat_list_st = sorted(dff['분류'].dropna().unique().astype(str).tolist())
            sel_cat_st = f2.selectbox("📁 작업 분류", ["전체"] + cat_list_st, key="stat_cat")
            chart_m = f3.selectbox("📊 분석 기준", ["작업자별", "파트별", "난이도별", "월별", "분기별"], key="stat_mode")
            if sel_y_st != "전체": dff = dff[dff['연도'] == sel_y_st]
            if sel_cat_st != "전체": dff = dff[dff['분류'] == sel_cat_st]
            target_col = {"작업자별":"이름", "파트별":"파트", "난이도별":"난이도", "월별":"월", "분기별":"분기"}[chart_m]
            detail_filter = f4.multiselect("🔍 상세 필터", sorted(dff[target_col].unique().astype(str).tolist()), key="stat_detail")
            if detail_filter: dff = dff[dff[target_col].astype(str).isin(detail_filter)]

        # 유틸리티 함수
        def format_score(val):
            return str(int(val)) if val == int(val) else f"{val:.2f}"

        def get_rgba(hex_color, opacity):
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'

        # 2. 요약 카드 4개
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        proj_count = dff['group_id'].nunique()
        avg_weighted = dff['최종점수'].mean() if not dff.empty else 0
        proj_uniq = dff.drop_duplicates('group_id')
        avg_edits = proj_uniq['프로젝트_수정횟수'].mean() if proj_count > 0 else 0
        p_rank = dff.groupby('이름')['수정횟수'].sum().sort_values(ascending=False)
        top_info = f"{p_rank.index[0]} / {p_rank.values[0]}회" if not p_rank.empty and p_rank.values[0] > 0 else "- / 0회"
        
        m_c1.markdown(f'<div class="metric-card"><div class="metric-label">총 프로젝트</div><div class="metric-value">{proj_count}건</div></div>', unsafe_allow_html=True)
        m_c2.markdown(f'<div class="metric-card"><div class="metric-label">가중점수 평균</div><div class="metric-value">{format_score(avg_weighted)}점</div></div>', unsafe_allow_html=True)
        m_c3.markdown(f'<div class="metric-card"><div class="metric-label">수정횟수 평균</div><div class="metric-value">{format_score(avg_edits)}회</div></div>', unsafe_allow_html=True)
        m_c4.markdown(f'<div class="metric-card"><div class="metric-label">최다 수정</div><div class="metric-value" style="font-size:1.1em;">{top_info}</div></div>', unsafe_allow_html=True)

        # 3. 점수 그래프 영역
        main_chart_spot = st.container()

        # 4. 점수 그래프의 상세 디자인 설정 창 (3번 바로 아래 위치)
        with st.expander("🎨 그래프 상세 디자인 설정", expanded=False):
            cl, cm, cr = st.columns([1.2, 1, 1.2])
            with cl:
                c_type = st.radio("📈 그래프 형태", ["막대형", "선형"], horizontal=True, key="ds_type")
                f_size = st.slider("🟦 전체 글자 크기", 10, 35, 14, key="ds_font")
                thickness = st.slider("📏 그래프 두께", 0.1, 1.0, 0.7, key="ds_thick")
            with cm:
                f_color = st.color_picker("가중점수 색상", "#00FFD1", key="ds_c2")
                b_color = st.color_picker("기본점수 색상", "#555555", key="ds_c1")
            with cr:
                d_type = st.selectbox("✨ 디자인 타입", ["기본형", "타입 A"], key="ds_d_type")
                pattern = st.selectbox("🏁 막대 내부 패턴", ["없음", "/", "\\", "x", "."], key="ds_p")

        with main_chart_spot:
            if not dff.empty:
                chart_df = dff.groupby(target_col)[['기본점수', '최종점수']].sum().reset_index()
                if chart_m == "월별": chart_df[target_col] = chart_df[target_col].apply(lambda x: f"{x}월")
                chart_df['base_text'] = chart_df['기본점수'].apply(format_score)
                chart_df['final_text'] = chart_df['최종점수'].apply(format_score)

                fig = go.Figure()
                fixed_font = dict(size=f_size, color="white")
                p_map = {"없음":None, "/":"/", "\\":"\\", "x":"x", ".":"."}
                
                if c_type == "막대형":
                    b_mode = 'overlay' if d_type == "타입 A" else 'group'
                    fig.add_trace(go.Bar(
                        x=chart_df[target_col], y=chart_df['기본점수'], name='기본점수',
                        marker=dict(color=chart_df['기본점수'], colorscale=[[0, get_rgba(b_color, 0.1)], [1, b_color]], pattern_shape=p_map.get(pattern)),
                        text=chart_df['base_text'], textposition='inside' if d_type == "타입 A" else 'outside',
                        textfont=fixed_font, insidetextfont=fixed_font, outsidetextfont=fixed_font,
                        constraintext='none', cliponaxis=False, width=0.8 if d_type == "타입 A" else None, opacity=0.7 if d_type == "타입 A" else 1.0
                    ))
                    fig.add_trace(go.Bar(
                        x=chart_df[target_col], y=chart_df['최종점수'], name='가중점수',
                        marker=dict(color=chart_df['최종점수'], colorscale=[[0, get_rgba(f_color, 0.1)], [1, f_color]], pattern_shape=p_map.get(pattern)),
                        text=chart_df['final_text'], textposition='outside',
                        textfont=fixed_font, insidetextfont=fixed_font, outsidetextfont=fixed_font,
                        constraintext='none', cliponaxis=False, width=0.5 if d_type == "타입 A" else None
                    ))
                    fig.update_layout(barmode=b_mode, bargap=1.0 - thickness)
                else:
                    line_shape = 'spline' if d_type == "타입 A" else 'linear'
                    fig.add_trace(go.Scatter(x=chart_df[target_col], y=chart_df['기본점수'], name='기본점수', mode='lines+markers+text',
                        line=dict(color=b_color, width=thickness*15, shape=line_shape),
                        text=chart_df['base_text'], textposition='top center', textfont=fixed_font))
                    fig.add_trace(go.Scatter(x=chart_df[target_col], y=chart_df['최종점수'], name='가중점수', mode='lines+markers+text',
                        line=dict(color=f_color, width=thickness*15, shape=line_shape),
                        text=chart_df['final_text'], textposition='bottom center', textfont=fixed_font))
                
                fig.update_layout(template="plotly_dark", height=600, font=dict(size=f_size), coloraxis_showscale=False,
                                  margin=dict(t=80, b=50, l=50, r=50), uniformtext=dict(mode=False))
                st.plotly_chart(fig, use_container_width=True)

        # 5. 프로젝트별/작업자별 최다 수정 막대그래프 (나란히 배치)
        st.divider()
        st.subheader("📈 수정 횟수 분석")
        top_chart_spot = st.container()
        
        # 6. 최다 수정 그래프 전용 설정 창
        with st.expander("🎨 그래프 상세 디자인 설정", expanded=False):
            tc1, tc2, tc3 = st.columns([1.2, 1, 1.2])
            with tc1:
                t_f_size = tc1.slider("🟦 글자 크기", 10, 35, 14, key="top_f_size")
                t_thick = tc1.slider("📏 막대 두께", 0.1, 1.0, 0.6, key="top_thick")
            with tc2:
                t_color_p = tc2.color_picker("프로젝트 막대 색상", "#E84D4D", key="top_cp")
                t_color_w = tc2.color_picker("작업자 막대 색상", "#FFA500", key="top_cw")
            with tc3:
                t_pattern = tc3.selectbox("🏁 막대 패턴", ["없음", "/", "\\", "x", "."], key="top_pat")

        with top_chart_spot:
            if not dff.empty:
                col1, col2 = st.columns(2)
                
                # [수정 로직 1] 수정 횟수가 1회 이상인 데이터만 필터링 후 상위 5개 추출
                top_proj = dff.drop_duplicates('group_id')
                top_proj = top_proj[top_proj['프로젝트_수정횟수'] > 0].nlargest(5, '프로젝트_수정횟수')
                
                top_worker = dff.groupby('이름')['수정횟수'].sum().reset_index()
                top_worker = top_worker[top_worker['수정횟수'] > 0].nlargest(5, '수정횟수')
                
                t_font_cfg = dict(size=t_f_size, color="white")
                p_map = {"없음":None, "/":"/", "\\":"\\", "x":"x", ".":"."}

                with col1:
                    st.markdown("##### 프로젝트 최다 수정")
                    if not top_proj.empty:
                        fig_p = go.Figure(go.Bar(
                            x=top_proj['프로젝트명'], y=top_proj['프로젝트_수정횟수'],
                            marker=dict(color=top_proj['프로젝트_수정횟수'], colorscale=[[0, get_rgba(t_color_p, 0.2)], [1, t_color_p]], pattern_shape=p_map.get(t_pattern)),
                            text=top_proj['프로젝트_수정횟수'], textposition='outside',
                            textfont=t_font_cfg, insidetextfont=t_font_cfg, outsidetextfont=t_font_cfg,
                            constraintext='none', cliponaxis=False, width=t_thick
                        ))
                        # [수정 로직 2] Y축 기준선 및 그리드 활성화
                        fig_p.update_layout(
                            template="plotly_dark", height=400, margin=dict(t=50, b=50, l=50, r=30),
                            xaxis=dict(tickfont=dict(size=t_f_size)),
                            yaxis=dict(
                                title="수정 횟수",
                                tickfont=dict(size=t_f_size),
                                showgrid=True,        # 그리드 표시
                                gridcolor='rgba(255,255,255,0.1)', # 연한 그리드 색상
                                zeroline=True,        # 0점 기준선 표시
                                zerolinecolor='white' # 기준선 색상
                            ),
                            uniformtext=dict(mode=False)
                        )
                        st.plotly_chart(fig_p, use_container_width=True)
                    else:
                        st.info("수정 내역이 있는 프로젝트가 없습니다.")

                with col2:
                    st.markdown("##### 작업자 최다 수정")
                    if not top_worker.empty:
                        fig_w = go.Figure(go.Bar(
                            x=top_worker['이름'], y=top_worker['수정횟수'],
                            marker=dict(color=top_worker['수정횟수'], colorscale=[[0, get_rgba(t_color_w, 0.2)], [1, t_color_w]], pattern_shape=p_map.get(t_pattern)),
                            text=top_worker['수정횟수'], textposition='outside',
                            textfont=t_font_cfg, insidetextfont=t_font_cfg, outsidetextfont=t_font_cfg,
                            constraintext='none', cliponaxis=False, width=t_thick
                        ))
                        # [수정 로직 2] Y축 기준선 및 그리드 활성화
                        fig_w.update_layout(
                            template="plotly_dark", height=400, margin=dict(t=50, b=50, l=50, r=30),
                            xaxis=dict(tickfont=dict(size=t_f_size)),
                            yaxis=dict(
                                title="총 수정 횟수",
                                tickfont=dict(size=t_f_size),
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.1)',
                                zeroline=True,
                                zerolinecolor='white'
                            ),
                            uniformtext=dict(mode=False)
                        )
                        st.plotly_chart(fig_w, use_container_width=True)
                    else:
                        st.info("수정 내역이 있는 작업자가 없습니다.")

# -----------------------------------------------------------------------------
# [TAB 3] 설정 (난이도/기여도 동적 관리 및 레이아웃 개편)
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("⚙️ 시스템 환경 설정")
    
    # 상단: 난이도 및 기여도 동적 관리 (2컬럼 레이아웃)
    col_diff, col_cont = st.columns(2)
    
    # --- 1. 난이도 설정 구역 ---
    with col_diff:
        with st.container(border=True):
            st.subheader("📊 난이도 가중치 관리")
            
            # (1) 난이도 추가 UI
            with st.expander("➕ 난이도 항목 추가", expanded=False):
                ad1, ad2, ad3 = st.columns([1, 1, 1])
                new_d_key = ad1.text_input("난이도 명", placeholder="예: A+", key="add_d_k")
                new_d_val = ad2.number_input("가중치", value=1.0, step=0.1, key="add_d_v")
                if ad3.button("추가", key="btn_add_d", use_container_width=True):
                    if new_d_key and new_d_key not in config["diff_weights"]:
                        config["diff_weights"][new_d_key] = new_d_val
                        with open(CONFIG_FILE, 'w', encoding='utf-8') as f: json.dump(config, f, indent=4)
                        st.rerun()
            
            # (2) 기존 난이도 리스트 (수정 및 삭제)
            new_diff_cfg = {}
            for k in list(config["diff_weights"].keys()):
                d_c1, d_c2, d_c3 = st.columns([2, 2, 1])
                d_c1.markdown(f"**{k}**")
                val = d_c2.number_input("가중치", value=float(config["diff_weights"][k]), step=0.1, key=f"edit_d_{k}", label_visibility="collapsed")
                new_diff_cfg[k] = val
                if d_c3.button("🗑️", key=f"del_d_{k}"):
                    del config["diff_weights"][k]
                    with open(CONFIG_FILE, 'w', encoding='utf-8') as f: json.dump(config, f, indent=4)
                    st.rerun()

    # --- 2. 기여도 설정 구역 ---
    with col_cont:
        with st.container(border=True):
            st.subheader("💡 기여도 가중치 관리")
            
            # (1) 기여도 추가 UI
            with st.expander("➕ 기여도 항목 추가", expanded=False):
                ac1, ac2, ac3 = st.columns([1, 1, 1])
                new_c_key = ac1.text_input("기여도 명", placeholder="예: 최상", key="add_c_k")
                new_c_val = ac2.number_input("가중치", value=1.0, step=0.1, key="add_c_v")
                if ac3.button("추가", key="btn_add_c", use_container_width=True):
                    if new_c_key and new_c_key not in config["cont_weights"]:
                        config["cont_weights"][new_c_key] = new_c_val
                        with open(CONFIG_FILE, 'w', encoding='utf-8') as f: json.dump(config, f, indent=4)
                        st.rerun()

            # (2) 기존 기여도 리스트 (수정 및 삭제)
            new_cont_cfg = {}
            for k in list(config["cont_weights"].keys()):
                c_c1, c_c2, c_c3 = st.columns([2, 2, 1])
                c_c1.markdown(f"**{k}**")
                val = c_c2.number_input("가중치", value=float(config["cont_weights"][k]), step=0.1, key=f"edit_c_{k}", label_visibility="collapsed")
                new_cont_cfg[k] = val
                if c_c3.button("🗑️", key=f"del_c_{k}"):
                    del config["cont_weights"][k]
                    with open(CONFIG_FILE, 'w', encoding='utf-8') as f: json.dump(config, f, indent=4)
                    st.rerun()

    # 하단: 기타 시스템 설정 (간결한 가로 배치)
    st.write("")
    with st.container(border=True):
        st.subheader("🎨 시스템 공통 설정")
        s1, s2, s3 = st.columns([1, 1, 1])
        new_penalty = s1.number_input("📉 수정 1회당 감점률", value=float(config["penalty_rate"]), step=0.01)
        new_color = s2.color_picker("🎨 시스템 메인 컬러", value=config["main_color"])
        s3.markdown('<div style="margin-top:32px;"></div>', unsafe_allow_html=True)
        save_btn = s3.button("💾 모든 설정 저장 및 데이터 재계산", type="primary", use_container_width=True)

    # 설정 저장 및 전체 데이터 업데이트 로직
    if save_btn:
        config["diff_weights"] = new_diff_cfg
        config["cont_weights"] = new_cont_cfg
        config["penalty_rate"] = new_penalty
        config["main_color"] = new_color
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        
        if not all_df.empty:
            updated_list = []
            # 전체 프로젝트 순회하며 새로운 가중치로 재계산
            for gid in all_df['group_id'].unique():
                pdf = all_df[all_df['group_id'] == gid].copy()
                diff = pdf.iloc[0]['난이도']
                # 만약 기존 데이터의 난이도가 삭제되었다면 기본값(목록의 첫번째) 적용
                if diff not in config["diff_weights"]:
                    diff = list(config["diff_weights"].keys())[0]
                
                edits = pdf.iloc[0]['프로젝트_수정횟수']
                updated_list.append(run_score_engine(pdf, diff, edits, config))
            
            all_df = pd.concat(updated_list, ignore_index=True)
            save_and_stay(all_df, st.session_state.opened_gid)
        else:
            st.success("설정이 저장되었습니다.")

            st.rerun()
