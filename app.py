import streamlit as st
import pandas as pd
import json
import os
import uuid
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# [PART 1] 시스템 설정 및 데이터 로직
# =============================================================================

# 1-1. 페이지 기본 설정
st.set_page_config(page_title="MNK 성과관리 시스템", layout="wide")

DATA_FILE = "performance_data.csv"
CONFIG_FILE = "system_config.json"
YEAR_OPTIONS = [str(y) for y in range(datetime.now().year, datetime.now().year - 5, -1)]
PART_ORDER = ["마케팅", "콘티", "모델링", "애니메이션", "편집", "디자인컷"]

# 🚀 성능 개선: 페이지네이션 설정
PROJECTS_PER_PAGE = 10

# 1-2. 세션 상태 초기화
if 'opened_gid' not in st.session_state:
    st.session_state.opened_gid = None
if 'temp_workers' not in st.session_state:
    st.session_state.temp_workers = []
if 'temp_project_data' not in st.session_state:
    st.session_state.temp_project_data = {}
if 'cached_year_list' not in st.session_state:
    st.session_state.cached_year_list = None
if 'cached_cat_list' not in st.session_state:
    st.session_state.cached_cat_list = None
if 'cached_project_info' not in st.session_state:
    st.session_state.cached_project_info = None
if 'cached_df' not in st.session_state:
    st.session_state.cached_df = None
if 'last_load_time' not in st.session_state:
    st.session_state.last_load_time = None
# 🚀 페이지네이션 상태
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
# 🚀 검색 인덱스 캐시
if 'search_index' not in st.session_state:
    st.session_state.search_index = None
if 'search_index_timestamp' not in st.session_state:
    st.session_state.search_index_timestamp = None
if 'show_duplicate_warning' not in st.session_state:
    st.session_state.show_duplicate_warning = False
if 'pending_project_data' not in st.session_state:
    st.session_state.pending_project_data = None

# 1-3. 데이터 입출력 함수
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "diff_weights": {"S": 2.0, "A": 1.5, "B": 1.0, "C": 0.8, "D": 0.5},
        "cont_weights": {"상": 1.2, "중": 1.0, "하": 0.8},
        "penalty_rate": 0.1, "main_color": "#E84D4D", "font_family": "Pretendard"
    }

# 🚀 성능 개선: 캐싱된 데이터 로드
def load_data():
    if os.path.exists(DATA_FILE):
        current_mtime = os.path.getmtime(DATA_FILE)
        if st.session_state.cached_df is not None and st.session_state.last_load_time == current_mtime:
            return st.session_state.cached_df.copy()
        
        df = pd.read_csv(DATA_FILE)
        df['연도'] = df['연도'].astype(str)
        if 'worker_id' not in df.columns:
            df['worker_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
        st.session_state.cached_df = df
        st.session_state.last_load_time = current_mtime
        return df
    return pd.DataFrame()

def save_and_stay(df, gid=None):
    if not df.empty:
        df['파트'] = pd.Categorical(df['파트'], categories=PART_ORDER, ordered=True)
        df = df.sort_values(by=['등록일시', 'group_id', '파트'], ascending=[False, True, True])
    df.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
    st.session_state.opened_gid = gid
    # 🚀 캐시 무효화
    st.session_state.cached_df = None
    st.session_state.last_load_time = None
    st.session_state.cached_year_list = None
    st.session_state.cached_cat_list = None
    st.session_state.search_index = None
    st.session_state.search_index_timestamp = None
    st.rerun()

# 1-4. 핵심 점수 계산 엔진
def run_score_engine(project_df, p_diff, p_total_edits, cfg):
    df = project_df.copy()
    if len(df) == 0:
        return df
    
    if '제외횟수' not in df.columns:
        df['제외횟수'] = 0.0
    df['제외횟수'] = pd.to_numeric(df['제외횟수'], errors='coerce').fillna(0)
    df['수정횟수'] = pd.to_numeric(df['수정횟수'], errors='coerce').fillna(0)
    p_total_edits = float(p_total_edits)

    total_personal_edits = df['수정횟수'].sum()
    common_pool = max(0.0, p_total_edits - total_personal_edits)

    df['공통수정분'] = 0.0
    total_workers = len(df)
    
    if total_workers > 0 and common_pool > 0:
        active_mask = df['제외횟수'] < common_pool
        active_workers = active_mask.sum()
        
        if active_workers > 0:
            total_exclude = df.loc[active_mask, '제외횟수'].sum()
            first_distribution = (common_pool - total_exclude) / active_workers
            df.loc[active_mask, '공통수정분'] = first_distribution
            
            for idx in df[active_mask].index:
                exclude_count = df.at[idx, '제외횟수']
                if exclude_count > 0:
                    other_active_workers = active_workers - 1
                    if other_active_workers > 0:
                        bonus_per_other = exclude_count / other_active_workers
                        for other_idx in df[active_mask].index:
                            if other_idx != idx:
                                df.at[other_idx, '공통수정분'] += bonus_per_other

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

    for idx in df.index:
        raw_val = df.at[idx, '점수입력']
        total_resp = max(0, df.at[idx, '공통수정분'] + df.at[idx, '수정횟수'])
        df.at[idx, '기본점수'] = round(raw_val, 2)
        penalty_val = round(raw_val * (total_resp * cfg["penalty_rate"]), 2)
        df.at[idx, '감점점수'] = penalty_val
        final_calc = max(0, raw_val - penalty_val) * cfg["diff_weights"].get(p_diff, 1.0)
        df.at[idx, '최종점수'] = round(final_calc, 2)
        df.at[idx, '공통수정분'] = round(df.at[idx, '공통수정분'], 4)

    return df

# 🚀 성능 개선: 초성 검색 함수 (캐싱 적용)
@st.cache_data
def get_chosung(text):
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    result = ""
    for char in str(text):
        if '가' <= char <= '힣':
            char_code = ord(char) - ord('가')
            result += CHOSUNG_LIST[char_code // 588]
        else:
            result += char
    return result

# 🚀 성능 개선: 검색 인덱스 구축
def build_search_index(df):
    """검색 인덱스 사전 구축"""
    search_index = {}
    for gid, g_df in df.groupby('group_id'):
        project_name = str(g_df.iloc[0]['프로젝트명'])
        worker_names = "".join(g_df['이름'].astype(str))
        combined = (project_name + worker_names).replace(" ", "")
        
        search_index[gid] = {
            'text': combined.lower(),
            'chosung': get_chosung(combined)
        }
    return search_index

# 🚀 성능 개선: 빠른 검색
def fast_search(df, query, search_index):
    """사전 구축된 인덱스를 사용한 빠른 검색"""
    if not query:
        return df
    
    query_lower = query.replace(" ", "").lower()
    query_chosung = get_chosung(query)
    
    matched_gids = [
        gid for gid, index in search_index.items()
        if query_lower in index['text'] or query_chosung in index['chosung']
    ]
    
    return df[df['group_id'].isin(matched_gids)]

config = load_config()
all_df = load_data()

# =============================================================================
# [PART 2] UI 스타일 정의
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
    .status-card {{ background: linear-gradient(135deg, #2D2D3A 0%, #1E1E26 100%); padding: 12px 20px; border-radius: 8px; border: 1px solid #3A3A4A; margin: 5px 0; }}
    .status-label {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
    .status-value {{ font-size: 18px; font-weight: 700; color: {config.get('main_color', '#E84D4D')}; margin-top: 3px; }}
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# [PART 3] 메인 화면 구성
# =============================================================================
tabs = st.tabs(["📝 작업 등록", "🗂️ 프로젝트 관리", "📈 통계 대시보드", "⚙️ 설정"])

# [TAB 0] 작업 등록
with tabs[0]:
    st.subheader("1️⃣ 프로젝트 기본 정보")
    with st.container(border=True):
        c_y, c_m, c1, c2, c3, c4 = st.columns([1, 0.8, 1.5, 0.8, 1.2, 0.8])
        p_year = c_y.selectbox("연도 설정", YEAR_OPTIONS, key="reg_y")
        p_month = c_m.selectbox("월 설정", list(range(1, 13)), index=datetime.now().month-1, key="reg_m")
        p_name = c1.text_input("프로젝트 명 설정", placeholder="예: 엠엔케이", key="reg_n")
        p_diff = c2.selectbox("난이도 설정", list(config["diff_weights"].keys()), index=2, key="reg_d")
        p_cat = c3.selectbox("분류 설정", ["영상", "디자인컷"], key="reg_c")
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
            else:
                st.warning("작업자 이름을 입력해주세요.")

    if st.session_state.temp_workers:
        st.write("---")
        st.markdown("### 📋 현재 추가된 명단")
        t_df = pd.DataFrame(st.session_state.temp_workers)
        st.dataframe(t_df[["파트", "이름", "기여도", "점수입력"]], use_container_width=True, hide_index=True)
        
        c_del, c_save = st.columns([1, 4])
        if c_del.button("🔄 목록 초기화"):
            st.session_state.temp_workers = []
            st.session_state.show_duplicate_warning = False
            st.session_state.pending_project_data = None
            st.rerun()
        
        if c_save.button("🚀 프로젝트 최종 저장 및 점수 발행", type="primary", use_container_width=True):
            # 프로젝트명 중복 체크
            existing_df = load_data()
            is_duplicate = not existing_df.empty and p_name in existing_df['프로젝트명'].values
            
            if is_duplicate:
                # 중복 확인 다이얼로그 표시
                st.session_state.show_duplicate_warning = True
                st.session_state.pending_project_data = {
                    't_df': t_df, 'p_diff': p_diff, 'p_edits': p_edits,
                    'p_year': p_year, 'p_month': p_month, 'p_name': p_name,
                    'p_cat': p_cat
                }
                st.rerun()
            else:
                # 중복이 아니면 바로 저장
                final_df = run_score_engine(t_df, p_diff, p_edits, config)
                gid = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{p_name}"
                final_df[['연도','월','프로젝트명','난이도','분류','프로젝트_수정횟수','group_id','등록일시']] = [
                    p_year, p_month, p_name, p_diff, p_cat, p_edits, gid, datetime.now().strftime("%Y-%m-%d %H:%M")
                ]
                all_df = pd.concat([load_data(), final_df], ignore_index=True)
                st.session_state.temp_workers = []
                save_and_stay(all_df, gid)
        
        # 중복 확인 다이얼로그 (저장 버튼 아래에 표시)
        if st.session_state.show_duplicate_warning:
            st.write("")
            with st.container(border=True):
                st.warning("⚠️ 이미 등록된 프로젝트가 존재합니다. 등록을 계속 하시겠습니까?")
                conf_col1, conf_col2, conf_col3 = st.columns([1, 1, 3])
                
                if conf_col1.button("✅ 예", key="confirm_yes", use_container_width=True, type="primary"):
                    # 저장 진행
                    pending = st.session_state.pending_project_data
                    final_df = run_score_engine(pending['t_df'], pending['p_diff'], pending['p_edits'], config)
                    gid = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{pending['p_name']}"
                    final_df[['연도','월','프로젝트명','난이도','분류','프로젝트_수정횟수','group_id','등록일시']] = [
                        pending['p_year'], pending['p_month'], pending['p_name'], pending['p_diff'], 
                        pending['p_cat'], pending['p_edits'], gid, datetime.now().strftime("%Y-%m-%d %H:%M")
                    ]
                    all_df = pd.concat([load_data(), final_df], ignore_index=True)
                    st.session_state.temp_workers = []
                    st.session_state.show_duplicate_warning = False
                    st.session_state.pending_project_data = None
                    save_and_stay(all_df, gid)
                
                if conf_col2.button("❌ 아니오", key="confirm_no", use_container_width=True):
                    st.session_state.show_duplicate_warning = False
                    st.session_state.pending_project_data = None
                    st.rerun()

# [TAB 1] 프로젝트 관리
with tabs[1]:
    if not all_df.empty:
        st.subheader("📊 데이터 현황")
        
        # 🚀 성능 개선: 캐싱된 연도/분류 목록
        if st.session_state.cached_year_list is None or st.session_state.cached_cat_list is None:
            st.session_state.cached_year_list = sorted(all_df['연도'].unique().tolist(), reverse=True)
            st.session_state.cached_cat_list = sorted(all_df['분류'].dropna().unique().astype(str).tolist())
        
        # 필터링
        with st.container(border=True):
            search_query = st.text_input("🔎 검색 (프로젝트명 또는 작업자 이름)", placeholder="초성 검색 가능", key="pm_search_main")
            f1, f2, f3, f4 = st.columns(4)
            sel_y = f1.selectbox("📅 연도", ["전체"] + st.session_state.cached_year_list, key="mg_f_y")
            sel_d = f2.selectbox("📊 난이도", ["전체"] + list(config["diff_weights"].keys()), key="mg_f_d")
            sel_q = f3.selectbox("📆 분기", ["전체", "1분기", "2분기", "3분기", "4분기"], key="mg_f_q")
            sel_c = f4.selectbox("📁 분류", ["전체"] + st.session_state.cached_cat_list, key="mg_f_c")

        # 🚀 성능 개선: 필터링 최적화
        filtered_df = all_df
        if sel_y != "전체":
            filtered_df = filtered_df[filtered_df['연도'] == sel_y]
        if sel_d != "전체":
            filtered_df = filtered_df[filtered_df['난이도'] == sel_d]
        if sel_q != "전체":
            filtered_df = filtered_df[filtered_df['월'].apply(lambda x: f"{(int(x)-1)//3 + 1}분기") == sel_q]
        if sel_c != "전체":
            filtered_df = filtered_df[filtered_df['분류'] == sel_c]

        # 🚀 성능 개선: 검색 인덱스 구축 (데이터 변경 시에만)
        current_timestamp = st.session_state.last_load_time
        if st.session_state.search_index is None or st.session_state.search_index_timestamp != current_timestamp:
            st.session_state.search_index = build_search_index(filtered_df)
            st.session_state.search_index_timestamp = current_timestamp
        
        # 🚀 성능 개선: 빠른 검색
        if search_query:
            filtered_df = fast_search(filtered_df, search_query, st.session_state.search_index)

        # 🚀 성능 개선: 현황 표시 (drop_duplicates 한 번만 실행)
        with st.container(border=True):
            status_cols = st.columns(5)
            
            unique_projects = filtered_df.drop_duplicates('group_id')
            total_projects = len(unique_projects)
            
            status_cols[0].markdown(f"""
                <div class="status-card">
                    <div class="status-label">총 등록 수</div>
                    <div class="status-value">{total_projects}건</div>
                </div>
            """, unsafe_allow_html=True)
            
            diff_counts = unique_projects['난이도'].value_counts()
            diff_text = " / ".join([f"{k}:{v}" for k, v in diff_counts.items()]) if not diff_counts.empty else "-"
            status_cols[1].markdown(f"""
                <div class="status-card">
                    <div class="status-label">난이도별</div>
                    <div class="status-value" style="font-size:14px;">{diff_text}</div>
                </div>
            """, unsafe_allow_html=True)
            
            quarter_counts = unique_projects['월'].apply(lambda x: f"{(int(x)-1)//3 + 1}분기").value_counts()
            quarter_text = " / ".join([f"{k}:{v}" for k, v in quarter_counts.items()]) if not quarter_counts.empty else "-"
            status_cols[2].markdown(f"""
                <div class="status-card">
                    <div class="status-label">분기별</div>
                    <div class="status-value" style="font-size:14px;">{quarter_text}</div>
                </div>
            """, unsafe_allow_html=True)
            
            cat_counts = unique_projects['분류'].value_counts()
            cat_text = " / ".join([f"{k}:{v}" for k, v in cat_counts.items()]) if not cat_counts.empty else "-"
            status_cols[3].markdown(f"""
                <div class="status-card">
                    <div class="status-label">분류별</div>
                    <div class="status-value" style="font-size:14px;">{cat_text}</div>
                </div>
            """, unsafe_allow_html=True)
            
            year_counts = unique_projects['연도'].value_counts()
            year_text = " / ".join([f"{k}:{v}" for k, v in sorted(year_counts.items(), reverse=True)]) if not year_counts.empty else "-"
            status_cols[4].markdown(f"""
                <div class="status-card">
                    <div class="status-label">연도별</div>
                    <div class="status-value" style="font-size:14px;">{year_text}</div>
                </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("📁 프로젝트 통합 검색 및 필터")

        # 정렬
        def sort_by_difficulty(df):
            diff_order = {k: i for i, k in enumerate(config["diff_weights"].keys())}
            df['_diff_order'] = df['난이도'].map(diff_order).fillna(999)
            return df
        
        project_representatives = filtered_df.drop_duplicates('group_id').copy()
        project_representatives['연도'] = project_representatives['연도'].astype(str)
        project_representatives['월'] = project_representatives['월'].astype(int)
        project_representatives = sort_by_difficulty(project_representatives)
        project_representatives = project_representatives.sort_values(
            by=['연도', '월', '_diff_order'], 
            ascending=[False, False, True]
        )
        
        sorted_gids = project_representatives['group_id'].tolist()

        # 🚀 성능 개선: 페이지네이션
        total_projects_count = len(sorted_gids)
        total_pages = max(1, (total_projects_count + PROJECTS_PER_PAGE - 1) // PROJECTS_PER_PAGE)
        
        # 페이지 범위 검증
        if st.session_state.current_page >= total_pages:
            st.session_state.current_page = max(0, total_pages - 1)
        
        # 페이지네이션 컨트롤
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("◀ 이전", disabled=st.session_state.current_page == 0):
                st.session_state.current_page -= 1
                st.rerun()
        
        with col2:
            st.markdown(f"<div style='text-align:center; padding: 8px;'>페이지 {st.session_state.current_page + 1} / {total_pages} (총 {total_projects_count}개 프로젝트)</div>", unsafe_allow_html=True)
        
        with col3:
            if st.button("다음 ▶", disabled=st.session_state.current_page >= total_pages - 1):
                st.session_state.current_page += 1
                st.rerun()

        # 🚀 성능 개선: 현재 페이지의 프로젝트만 표시
        start_idx = st.session_state.current_page * PROJECTS_PER_PAGE
        end_idx = min(start_idx + PROJECTS_PER_PAGE, total_projects_count)
        visible_gids = sorted_gids[start_idx:end_idx]

# 정렬된 순서로 프로젝트 표시 (페이지네이션 적용)
        for gid in visible_gids:
            # 표시용 데이터: temp_project_data가 있으면 그것을 사용, 없으면 원본 데이터 사용
            if gid in st.session_state.temp_project_data:
                g_df = st.session_state.temp_project_data[gid].copy()
            else:
                g_df = all_df[all_df['group_id'] == gid].copy()
            
            # 원본 데이터: 점수 표시용 (업데이트 버튼을 누르기 전까지는 원본 점수 표시)
            original_g_df = all_df[all_df['group_id'] == gid].copy()
            
            g_df['파트'] = pd.Categorical(g_df['파트'], categories=PART_ORDER, ordered=True)
            g_df = g_df.sort_values('파트')
            original_g_df['파트'] = pd.Categorical(original_g_df['파트'], categories=PART_ORDER, ordered=True)
            original_g_df = original_g_df.sort_values('파트')
            first = g_df.iloc[0]
            p_total_limit = int(first['프로젝트_수정횟수'])
            
            is_expanded = st.session_state.get('opened_gid') == gid
            
            with st.expander(f"📂 [{first['연도']}/{first['월']}월] {first['프로젝트명']} | {first['난이도']} | {first['분류']}", expanded=is_expanded):
                with st.container(border=True):
                    mc = st.columns([3, 1.2, 1.0, 1.1, 1.2, 1.2, 1, 0.5])
                    en = mc[0].text_input("프로젝트명", value=first['프로젝트명'], key=f"en_{gid}")
                    ey = mc[1].selectbox("연도", YEAR_OPTIONS, index=YEAR_OPTIONS.index(str(first['연도'])), key=f"ey_{gid}")
                    month_list = list(range(1, 13))
                    em = mc[2].selectbox("월", month_list, index=month_list.index(int(first['월'])), key=f"em_{gid}")
                    ed = mc[3].selectbox("난이도", list(config["diff_weights"].keys()), index=list(config["diff_weights"].keys()).index(first['난이도']), key=f"ed_{gid}")
                    ec = mc[4].text_input("분류", value=first['분류'], key=f"ec_{gid}")
                    ee = mc[5].number_input("전체 수정횟수", min_value=0, value=int(first['프로젝트_수정횟수']), key=f"ee_{gid}")
                    mc[6].markdown('<div style="margin-top:28px;"></div>', unsafe_allow_html=True)
                    is_del_ok = mc[7].checkbox("🗑️", key=f"del_chk_{gid}", label_visibility="collapsed")
                    
                    if mc[6].button("삭제", key=f"del_group_{gid}", disabled=not is_del_ok, use_container_width=True):
                        all_df = all_df[all_df['group_id'] != gid]
                        if gid in st.session_state.temp_project_data:
                            del st.session_state.temp_project_data[gid]
                        save_and_stay(all_df, None)

                st.divider()
                cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                headers = ["파트", "이름", "점수/기여도", "기본점수", "감점", "최종점수", "공통수정", "제외횟수", "개인수정", "수정조절", "삭제"]
                for col, text in zip(cols, headers):
                    col.markdown(f'<div class="header-style" style="font-size:11px; text-align:center;">{text}</div>', unsafe_allow_html=True)
                
                current_total_personal = g_df['수정횟수'].sum()
                
                # 원본 데이터를 worker_id로 인덱싱 (점수 표시용)
                original_dict = {row['worker_id']: row for _, row in original_g_df.iterrows()}

                for _, row in g_df.iterrows():
                    wid = row['worker_id']
                    r = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                    new_p = r[0].selectbox("P", PART_ORDER, index=PART_ORDER.index(row['파트']), key=f"p_{wid}", label_visibility="collapsed")
                    new_n = r[1].text_input("N", value=row['이름'], key=f"n_{wid}", label_visibility="collapsed")
                    
                    if row['파트'] == "마케팅":
                        new_val = r[2].number_input("V", value=float(row['점수입력']), key=f"v_{wid}", label_visibility="collapsed")
                        if new_val != row['점수입력'] or new_p != row['파트'] or new_n != row['이름']:
                            if gid not in st.session_state.temp_project_data:
                                st.session_state.temp_project_data[gid] = g_df.copy()
                            ridx = st.session_state.temp_project_data[gid][st.session_state.temp_project_data[gid]['worker_id'] == wid].index[0]
                            st.session_state.temp_project_data[gid].at[ridx, '점수입력'] = new_val
                            st.session_state.temp_project_data[gid].at[ridx, '파트'] = new_p
                            st.session_state.temp_project_data[gid].at[ridx, '이름'] = new_n
                    elif row['파트'] == "디자인컷":
                        r[2].markdown('<div style="text-align:center; margin-top:8px; font-size:12px; color:#aaa;">자동배분</div>', unsafe_allow_html=True)
                        if new_p != row['파트'] or new_n != row['이름']:
                            if gid not in st.session_state.temp_project_data:
                                st.session_state.temp_project_data[gid] = g_df.copy()
                            ridx = st.session_state.temp_project_data[gid][st.session_state.temp_project_data[gid]['worker_id'] == wid].index[0]
                            st.session_state.temp_project_data[gid].at[ridx, '파트'] = new_p
                            st.session_state.temp_project_data[gid].at[ridx, '이름'] = new_n
                    else:
                        cl = ["상", "중", "하"]
                        current_c = row['기여도'] if row['기여도'] in cl else "중"
                        new_c = r[2].selectbox("C", cl, index=cl.index(current_c), key=f"c_{wid}", label_visibility="collapsed")
                        if new_c != row['기여도'] or new_p != row['파트'] or new_n != row['이름']:
                            if gid not in st.session_state.temp_project_data:
                                st.session_state.temp_project_data[gid] = g_df.copy()
                            ridx = st.session_state.temp_project_data[gid][st.session_state.temp_project_data[gid]['worker_id'] == wid].index[0]
                            st.session_state.temp_project_data[gid].at[ridx, '기여도'] = new_c
                            st.session_state.temp_project_data[gid].at[ridx, '파트'] = new_p
                            st.session_state.temp_project_data[gid].at[ridx, '이름'] = new_n

                    # 점수는 원본 데이터에서 가져오기 (업데이트 버튼을 누르기 전까지는 원본 점수 표시)
                    original_row = original_dict.get(wid, row)
                    r[3].markdown(f'<div class="score-style">{original_row["기본점수"]:,.1f}</div>', unsafe_allow_html=True)
                    r[4].markdown(f'<div class="score-style">-{original_row["감점점수"]:,.1f}</div>', unsafe_allow_html=True)
                    r[5].markdown(f'<div class="score-style" style="font-size:15px; color:#00FFD1;">{original_row["최종점수"]:,.1f}</div>', unsafe_allow_html=True)
                    comm_edits = original_row.get("공통수정분", 0)
                    r[6].markdown(f'<div style="text-align:center; margin-top:8px; font-size:12px; color:#888;">{comm_edits:,.2f}회</div>', unsafe_allow_html=True)

                    max_proj_edits = int(first['프로젝트_수정횟수'])
                    exclude_options = list(range(max_proj_edits + 1))
                    val_ex = row.get('제외횟수', 0)
                    curr_ex = int(val_ex) if pd.notna(val_ex) and int(val_ex) <= max_proj_edits else 0
                    new_ex = r[7].selectbox("EX", exclude_options, index=exclude_options.index(curr_ex), key=f"ex_{wid}", label_visibility="collapsed")
                    
                    if new_ex != curr_ex:
                        if gid not in st.session_state.temp_project_data:
                            st.session_state.temp_project_data[gid] = g_df.copy()
                        ridx = st.session_state.temp_project_data[gid][st.session_state.temp_project_data[gid]['worker_id'] == wid].index[0]
                        st.session_state.temp_project_data[gid].at[ridx, '제외횟수'] = float(new_ex)

                    r[8].markdown(f'<div style="text-align:center; margin-top:8px; font-size:14px; font-weight:bold; color:#E84D4D;">{row["수정횟수"]}회</div>', unsafe_allow_html=True)
                    btn_c = r[9].columns([1, 1])
                    if btn_c[0].button("➖", key=f"mn_{wid}", use_container_width=True):
                        if gid not in st.session_state.temp_project_data:
                            st.session_state.temp_project_data[gid] = g_df.copy()
                        ridx = st.session_state.temp_project_data[gid][st.session_state.temp_project_data[gid]['worker_id'] == wid].index[0]
                        st.session_state.temp_project_data[gid].at[ridx, '수정횟수'] = max(0, row['수정횟수'] - 1)
                    
                    can_increase = current_total_personal < p_total_limit
                    if btn_c[1].button("➕", key=f"pl_{wid}", use_container_width=True, disabled=not can_increase):
                        if gid not in st.session_state.temp_project_data:
                            st.session_state.temp_project_data[gid] = g_df.copy()
                        ridx = st.session_state.temp_project_data[gid][st.session_state.temp_project_data[gid]['worker_id'] == wid].index[0]
                        st.session_state.temp_project_data[gid].at[ridx, '수정횟수'] += 1

                    del_c = r[10].columns([0.4, 0.6])
                    is_row_del = del_c[0].checkbox("", key=f"cw_{wid}", label_visibility="collapsed")
                    if del_c[1].button("🗑️", key=f"dw_{wid}", disabled=not is_row_del, use_container_width=True):
                        if gid not in st.session_state.temp_project_data:
                            st.session_state.temp_project_data[gid] = g_df.copy()
                        st.session_state.temp_project_data[gid] = st.session_state.temp_project_data[gid][st.session_state.temp_project_data[gid]['worker_id'] != wid]

                st.divider()
                st.markdown("### ➕ 작업자 추가 등록")
                with st.container(border=True):
                    add_cols = st.columns([1.5, 1.5, 1, 1.5, 1])
                    add_part = add_cols[0].selectbox("파트", PART_ORDER, key=f"add_part_{gid}")
                    add_name = add_cols[1].text_input("이름", placeholder="작업자 이름", key=f"add_name_{gid}")
                    add_is_special = add_part in ["마케팅", "디자인컷"]
                    add_cont = add_cols[2].selectbox("기여도", ["상", "중", "하"], index=1, disabled=add_is_special, key=f"add_cont_{gid}")
                    add_score = add_cols[3].number_input("마케팅 점수", min_value=0.0, disabled=(add_part != "마케팅"), key=f"add_score_{gid}")
                    
                    add_cols[4].markdown('<div style="margin-top:28px;"></div>', unsafe_allow_html=True)
                    if add_cols[4].button("추가", key=f"add_worker_{gid}", use_container_width=True, type="secondary"):
                        if add_name:
                            if gid not in st.session_state.temp_project_data:
                                st.session_state.temp_project_data[gid] = g_df.copy()
                            
                            new_worker = pd.DataFrame([{
                                "이름": add_name, "파트": add_part,
                                "기여도": "-" if add_is_special else add_cont,
                                "점수입력": add_score if add_part == "마케팅" else 0.0,
                                "수정횟수": 0, "제외횟수": 0, "worker_id": str(uuid.uuid4()),
                                "연도": first['연도'], "월": first['월'], "프로젝트명": first['프로젝트명'],
                                "난이도": first['난이도'], "분류": first['분류'],
                                "프로젝트_수정횟수": first['프로젝트_수정횟수'],
                                "group_id": gid, "등록일시": first['등록일시'],
                                "기본점수": 0.0, "감점점수": 0.0, "최종점수": 0.0, "공통수정분": 0.0
                            }])
                            
                            st.session_state.temp_project_data[gid] = pd.concat([
                                st.session_state.temp_project_data[gid], new_worker
                            ], ignore_index=True)
                            st.rerun()
                        else:
                            st.warning("작업자 이름을 입력해주세요.")
                
                st.write("")
                st.divider()
                if st.button("💾 프로젝트 업데이트", key=f"up_btn_{gid}", use_container_width=True, type="primary"):
                    if gid in st.session_state.temp_project_data:
                        updated_df = st.session_state.temp_project_data[gid].copy()
                        all_df = all_df[all_df['group_id'] != gid]
                        updated_df['프로젝트명'] = en
                        updated_df['연도'] = ey
                        updated_df['월'] = em
                        updated_df['난이도'] = ed
                        updated_df['분류'] = ec
                        updated_df['프로젝트_수정횟수'] = ee
                        updated_df = run_score_engine(updated_df, ed, ee, config)
                        all_df = pd.concat([all_df, updated_df], ignore_index=True)
                    else:
                        mask = all_df['group_id'] == gid
                        all_df.loc[mask, ['프로젝트명','연도','월','난이도','분류','프로젝트_수정횟수']] = [en, ey, em, ed, ec, ee]
                        all_df.loc[mask, :] = run_score_engine(all_df[mask], ed, ee, config)
                    
                    st.session_state.opened_gid = gid
                    if gid in st.session_state.temp_project_data:
                        del st.session_state.temp_project_data[gid]
                    save_and_stay(all_df, gid)
    else:
        st.info("📭 등록된 프로젝트가 없습니다. '작업 등록' 탭에서 프로젝트를 추가해주세요.")

# [TAB 2] 통계 대시보드
with tabs[2]:
    if all_df.empty:
        st.info("📊 통계를 생성할 데이터가 없습니다.")
    else:
        dff = all_df.copy()
        dff['실질수정'] = pd.to_numeric(dff['공통수정분'], errors='coerce').fillna(0) + pd.to_numeric(dff['수정횟수'], errors='coerce').fillna(0)
        dff['분기'] = dff['월'].apply(lambda x: f"{(int(x)-1)//3 + 1}분기")

        st.subheader("🔍 데이터 필터링")
        with st.container(border=True):
            f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
            sel_y_st = f1.selectbox("📅 연도", ["전체"] + sorted(dff['연도'].unique().tolist(), reverse=True), key="stat_y")
            cat_list_st = sorted(dff['분류'].dropna().unique().astype(str).tolist())
            sel_cat_st = f2.selectbox("📁 작업 분류", ["전체"] + cat_list_st, key="stat_cat")
            chart_m = f3.selectbox("📊 분석 기준", ["작업자별", "파트별", "난이도별", "월별", "분기별"], key="stat_mode")
            
            if sel_y_st != "전체":
                dff = dff[dff['연도'] == sel_y_st]
            if sel_cat_st != "전체":
                dff = dff[dff['분류'] == sel_cat_st]
            
            target_col = {"작업자별":"이름", "파트별":"파트", "난이도별":"난이도", "월별":"월", "분기별":"분기"}[chart_m]
            detail_filter = f4.multiselect("🔍 상세 필터", sorted(dff[target_col].unique().astype(str).tolist()), key="stat_detail")
            if detail_filter:
                dff = dff[dff[target_col].astype(str).isin(detail_filter)]

        def format_score(val):
            return str(int(val)) if val == int(val) else f"{val:.2f}"
        
        def get_rgba(hex_color, opacity):
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'

        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        proj_count = dff['group_id'].nunique()
        avg_weighted = dff['최종점수'].mean() if not dff.empty else 0
        proj_uniq = dff.drop_duplicates('group_id')
        avg_edits = proj_uniq['프로젝트_수정횟수'].mean() if proj_count > 0 else 0
        
        p_rank = dff.groupby('이름')['실질수정'].sum().sort_values(ascending=False)
        top_info = f"{p_rank.index[0]} / {p_rank.values[0]:,.1f}회" if not p_rank.empty and p_rank.values[0] > 0 else "- / 0회"
        
        m_c1.markdown(f'<div class="metric-card"><div class="metric-label">총 프로젝트</div><div class="metric-value">{proj_count}건</div></div>', unsafe_allow_html=True)
        m_c2.markdown(f'<div class="metric-card"><div class="metric-label">가중점수 평균</div><div class="metric-value">{format_score(avg_weighted)}점</div></div>', unsafe_allow_html=True)
        m_c3.markdown(f'<div class="metric-card"><div class="metric-label">프로젝트 수정평균</div><div class="metric-value">{format_score(avg_edits)}회</div></div>', unsafe_allow_html=True)
        m_c4.markdown(f'<div class="metric-card"><div class="metric-label">최다 수정(공통+개인)</div><div class="metric-value" style="font-size:1.1em;">{top_info}</div></div>', unsafe_allow_html=True)

        main_chart_spot = st.container()
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
                pattern = st.selectbox("🎁 막대 내부 패턴", ["없음", "/", "\\", "x", "."], key="ds_p")

        with main_chart_spot:
            if not dff.empty:
                # 분류가 "디자인컷"인 프로젝트 제외한 기본점수 계산
                chart_df_base = dff[dff['분류'] != '디자인컷'].groupby(target_col)['기본점수'].sum().reset_index()
                chart_df_base.columns = [target_col, '기본점수']
                
                # 전체(디자인컷 포함) 가중점수 계산
                chart_df_final = dff.groupby(target_col)['최종점수'].sum().reset_index()
                chart_df_final.columns = [target_col, '최종점수']
                
                # 두 데이터프레임 병합
                chart_df = pd.merge(chart_df_base, chart_df_final, on=target_col, how='outer').fillna(0)
                
                if chart_m == "월별":
                    chart_df[target_col] = chart_df[target_col].apply(lambda x: f"{x}월")
                chart_df['base_text'] = chart_df['기본점수'].apply(format_score)
                chart_df['final_text'] = chart_df['최종점수'].apply(format_score)
                fig = go.Figure()
                fixed_font = dict(size=f_size, color="white")
                p_map = {"없음":None, "/":"/", "\\":"\\" , "x":"x", ".":"."}
                
                if c_type == "막대형":
                    b_mode = 'overlay' if d_type == "타입 A" else 'group'
                    fig.add_trace(go.Bar(
                        x=chart_df[target_col], 
                        y=chart_df['기본점수'], 
                        name='기본점수', 
                        marker=dict(
                            color=chart_df['기본점수'], 
                            colorscale=[[0, get_rgba(b_color, 0.1)], [1, b_color]], 
                            pattern_shape=p_map.get(pattern)
                        ), 
                        text=chart_df['base_text'], 
                        textposition='inside' if d_type == "타입 A" else 'outside', 
                        textfont=fixed_font, 
                        insidetextfont=fixed_font, 
                        outsidetextfont=fixed_font, 
                        constraintext='none', 
                        cliponaxis=False, 
                        width=0.8 if d_type == "타입 A" else None, 
                        opacity=0.7 if d_type == "타입 A" else 1.0
                    ))
                    fig.add_trace(go.Bar(
                        x=chart_df[target_col], 
                        y=chart_df['최종점수'], 
                        name='가중점수', 
                        marker=dict(
                            color=chart_df['최종점수'], 
                            colorscale=[[0, get_rgba(f_color, 0.1)], [1, f_color]], 
                            pattern_shape=p_map.get(pattern)
                        ), 
                        text=chart_df['final_text'], 
                        textposition='outside', 
                        textfont=fixed_font, 
                        insidetextfont=fixed_font, 
                        outsidetextfont=fixed_font, 
                        constraintext='none', 
                        cliponaxis=False, 
                        width=0.5 if d_type == "타입 A" else None
                    ))
                    fig.update_layout(barmode=b_mode, bargap=1.0 - thickness)
                else:
                    line_shape = 'spline' if d_type == "타입 A" else 'linear'
                    fig.add_trace(go.Scatter(
                        x=chart_df[target_col], 
                        y=chart_df['기본점수'], 
                        name='기본점수', 
                        mode='lines+markers+text', 
                        line=dict(color=b_color, width=thickness*15, shape=line_shape), 
                        text=chart_df['base_text'], 
                        textposition='top center', 
                        textfont=fixed_font
                    ))
                    fig.add_trace(go.Scatter(
                        x=chart_df[target_col], 
                        y=chart_df['최종점수'], 
                        name='가중점수', 
                        mode='lines+markers+text', 
                        line=dict(color=f_color, width=thickness*15, shape=line_shape), 
                        text=chart_df['final_text'], 
                        textposition='bottom center', 
                        textfont=fixed_font
                    ))
                
                fig.update_layout(
                    template="plotly_dark", 
                    height=600, 
                    font=dict(size=f_size), 
                    coloraxis_showscale=False, 
                    margin=dict(t=80, b=50, l=50, r=50), 
                    uniformtext=dict(mode=False)
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("📈 수정 횟수 분석")
        top_chart_spot = st.container()
        with st.expander("🎨 그래프 상세 디자인 설정", expanded=False):
            tc1, tc2, tc3 = st.columns([1.2, 1, 1.2])
            with tc1:
                t_f_size = tc1.slider("🟦 글자 크기", 10, 35, 14, key="top_f_size")
                t_thick = tc1.slider("📏 막대 두께", 0.1, 1.0, 0.6, key="top_thick")
            with tc2:
                t_color_p = tc2.color_picker("프로젝트 막대 색상", "#E84D4D", key="top_cp")
                t_color_w = tc2.color_picker("작업자 막대 색상", "#FFA500", key="top_cw")
            with tc3:
                t_pattern = tc3.selectbox("🎁 막대 패턴", ["없음", "/", "\\", "x", "."], key="top_pat")

        with top_chart_spot:
            if not dff.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 프로젝트 최다 수정")
                    top_proj = dff.drop_duplicates('group_id')
                    top_proj = top_proj[top_proj['프로젝트_수정횟수'] > 0].nlargest(5, '프로젝트_수정횟수')
                    if not top_proj.empty:
                        fig_p = go.Figure(go.Bar(
                            x=top_proj['프로젝트명'], 
                            y=top_proj['프로젝트_수정횟수'], 
                            marker=dict(
                                color=top_proj['프로젝트_수정횟수'], 
                                colorscale=[[0, get_rgba(t_color_p, 0.2)], [1, t_color_p]], 
                                pattern_shape=p_map.get(t_pattern)
                            ), 
                            text=top_proj['프로젝트_수정횟수'], 
                            textposition='outside', 
                            textfont=dict(size=t_f_size, color="white"), 
                            constraintext='none', 
                            cliponaxis=False, 
                            width=t_thick
                        ))
                        fig_p.update_layout(
                            template="plotly_dark", 
                            height=400, 
                            margin=dict(t=50, b=50, l=50, r=30), 
                            xaxis=dict(tickfont=dict(size=t_f_size)), 
                            yaxis=dict(title="수정 횟수", showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=True, zerolinecolor='white'), 
                            uniformtext=dict(mode=False)
                        )
                        st.plotly_chart(fig_p, use_container_width=True)
                    else:
                        st.info("수정 내역이 없습니다.")

                with col2:
                    st.markdown("##### 작업자 최다 수정 (공통+개인)")
                    top_worker = dff.groupby('이름')['실질수정'].sum().reset_index()
                    top_worker = top_worker[top_worker['실질수정'] > 0].nlargest(5, '실질수정')
                    if not top_worker.empty:
                        top_worker['text'] = top_worker['실질수정'].apply(lambda x: f"{x:,.1f}")
                        fig_w = go.Figure(go.Bar(
                            x=top_worker['이름'], 
                            y=top_worker['실질수정'], 
                            marker=dict(
                                color=top_worker['실질수정'], 
                                colorscale=[[0, get_rgba(t_color_w, 0.2)], [1, t_color_w]], 
                                pattern_shape=p_map.get(t_pattern)
                            ), 
                            text=top_worker['text'], 
                            textposition='outside', 
                            textfont=dict(size=t_f_size, color="white"), 
                            constraintext='none', 
                            cliponaxis=False, 
                            width=t_thick
                        ))
                        fig_w.update_layout(
                            template="plotly_dark", 
                            height=400, 
                            margin=dict(t=50, b=50, l=50, r=30), 
                            xaxis=dict(tickfont=dict(size=t_f_size)), 
                            yaxis=dict(title="총 실질수정 횟수", showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=True, zerolinecolor='white'), 
                            uniformtext=dict(mode=False)
                        )
                        st.plotly_chart(fig_w, use_container_width=True)
                    else:
                        st.info("수정 내역이 없습니다.")

# [TAB 3] 설정
with tabs[3]:
    st.header("⚙️ 시스템 환경 설정")
    col_diff, col_cont = st.columns(2)
    
    with col_diff:
        with st.container(border=True):
            st.subheader("📊 난이도 가중치 관리")
            with st.expander("➕ 난이도 항목 추가", expanded=False):
                ad1, ad2, ad3 = st.columns([1, 1, 1])
                new_d_key = ad1.text_input("난이도 명", placeholder="예: A+", key="add_d_k")
                new_d_val = ad2.number_input("가중치", value=1.0, step=0.1, key="add_d_v")
                if ad3.button("추가", key="btn_add_d", use_container_width=True):
                    if new_d_key and new_d_key not in config["diff_weights"]:
                        config["diff_weights"][new_d_key] = new_d_val
                        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                            json.dump(config, f, indent=4)
                        st.rerun()
            
            new_diff_cfg = {}
            for k in list(config["diff_weights"].keys()):
                d_c1, d_c2, d_c3 = st.columns([2, 2, 1])
                d_c1.markdown(f"**{k}**")
                val = d_c2.number_input("가중치", value=float(config["diff_weights"][k]), step=0.1, key=f"edit_d_{k}", label_visibility="collapsed")
                new_diff_cfg[k] = val
                if d_c3.button("🗑️", key=f"del_d_{k}"):
                    del config["diff_weights"][k]
                    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=4)
                    st.rerun()
    
    with col_cont:
        with st.container(border=True):
            st.subheader("💡 기여도 가중치 관리")
            with st.expander("➕ 기여도 항목 추가", expanded=False):
                ac1, ac2, ac3 = st.columns([1, 1, 1])
                new_c_key = ac1.text_input("기여도 명", placeholder="예: 최상", key="add_c_k")
                new_c_val = ac2.number_input("가중치", value=1.0, step=0.1, key="add_c_v")
                if ac3.button("추가", key="btn_add_c", use_container_width=True):
                    if new_c_key and new_c_key not in config["cont_weights"]:
                        config["cont_weights"][new_c_key] = new_c_val
                        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                            json.dump(config, f, indent=4)
                        st.rerun()
            
            new_cont_cfg = {}
            for k in list(config["cont_weights"].keys()):
                c_c1, c_c2, c_c3 = st.columns([2, 2, 1])
                c_c1.markdown(f"**{k}**")
                val = c_c2.number_input("가중치", value=float(config["cont_weights"][k]), step=0.1, key=f"edit_c_{k}", label_visibility="collapsed")
                new_cont_cfg[k] = val
                if c_c3.button("🗑️", key=f"del_c_{k}"):
                    del config["cont_weights"][k]
                    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=4)
                    st.rerun()
    
    st.write("")
    with st.container(border=True):
        st.subheader("🎨 시스템 공통 설정")
        s1, s2, s3 = st.columns([1, 1, 1])
        new_penalty = s1.number_input("📉 수정 1회당 감점율", value=float(config["penalty_rate"]), step=0.01)
        new_color = s2.color_picker("🎨 시스템 메인 컬러", value=config["main_color"])
        s3.markdown('<div style="margin-top:32px;"></div>', unsafe_allow_html=True)
        if s3.button("💾 모든 설정 저장 및 데이터 재계산", type="primary", use_container_width=True):
            config.update({
                "diff_weights": new_diff_cfg, 
                "cont_weights": new_cont_cfg, 
                "penalty_rate": new_penalty, 
                "main_color": new_color
            })
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            
            if not all_df.empty:
                updated_list = []
                for gid in all_df['group_id'].unique():
                    pdf = all_df[all_df['group_id'] == gid].copy()
                    diff = pdf.iloc[0]['난이도']
                    if diff not in config["diff_weights"]:
                        diff = list(config["diff_weights"].keys())[0]
                    updated_list.append(run_score_engine(pdf, diff, pdf.iloc[0]['프로젝트_수정횟수'], config))
                all_df = pd.concat(updated_list, ignore_index=True)
                save_and_stay(all_df, st.session_state.opened_gid)
            else:
                st.rerun()
