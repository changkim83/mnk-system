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

# 구글 시트 연결 (Secrets 설정 기반)
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

def save_to_gsheets(df, config_data=None, gid=None):
    conn.update(worksheet="Data", data=df)
    if config_data:
        cfg_df = pd.DataFrame([{
            "diff_weights": json.dumps(config_data["diff_weights"]),
            "cont_weights": json.dumps(config_data["cont_weights"]),
            "penalty_rate": config_data["penalty_rate"],
            "main_color": config_data["main_color"]
        }])
        conn.update(worksheet="Config", data=cfg_df)
    st.session_state.opened_gid = gid
    st.cache_data.clear()
    st.rerun()

def run_score_engine(project_df, p_diff, p_total_edits, cfg):
    df = project_df.copy()
    if len(df) == 0: return df
    
    df['공통수정분'] = 0.0
    df['제외횟수'] = pd.to_numeric(df.get('제외횟수', 0), errors='coerce').fillna(0)
    df['수정횟수'] = pd.to_numeric(df.get('수정횟수', 0), errors='coerce').fillna(0)

    # 파트별 배분
    mkt_mask = df['파트'] == "마케팅"
    mkt_sum = df.loc[mkt_mask, '점수입력'].sum()
    design_mask = df['파트'] == "디자인컷"
    design_count = design_mask.sum()
    rem_pool = max(0, 100.0 - mkt_sum)

    if design_count > 0:
        df.loc[design_mask, '점수입력'] = round(rem_pool / design_count, 2)
    else:
        prod_mask = ~df['파트'].isin(["마케팅", "디자인컷"])
        total_w = sum([cfg["cont_weights"].get(row['기여도'], 1.0) for _, row in df[prod_mask].iterrows()])
        if total_w > 0:
            unit = rem_pool / total_w
            for idx in df[prod_mask].index:
                df.at[idx, '점수입력'] = round(unit * cfg["cont_weights"].get(df.at[idx, '기여도'], 1.0), 2)

    # 공통수정 배분
    total_n = len(df)
    p_total_edits = float(p_total_edits)
    sum_ex = 0.0
    for idx in df[df['제외횟수'] > 0].index:
        share = max(0, (p_total_edits - df.at[idx, '제외횟수']) / total_n)
        df.at[idx, '공통수정분'] = share
        sum_ex += share
    
    non_ex_mask = df['제외횟수'] == 0
    if non_ex_mask.sum() > 0:
        df.loc[non_ex_mask, '공통수정분'] = max(0, p_total_edits - sum_ex) / non_ex_mask.sum()

    # 최종 점수 확정
    diff_w = cfg["diff_weights"].get(p_diff, 1.0)
    for idx in df.index:
        raw = df.at[idx, '점수입력']
        total_resp = df.at[idx, '공통수정분'] + df.at[idx, '수정횟수']
        penalty = round(raw * (total_resp * cfg["penalty_rate"]), 2)
        df.at[idx, '기본점수'] = round(raw, 2)
        df.at[idx, '감점점수'] = penalty
        df.at[idx, '최종점수'] = round(max(0, raw - penalty) * diff_w, 2)
    return df

config = load_config()
all_df = load_data()

# CSS 스타일 적용
st.markdown(f"""
    <style>
    .stApp {{ background-color: #1E1E26; color: #f0f2f6; }}
    .metric-card {{ background-color: #2D2D3A; padding: 15px; border-radius: 10px; border-left: 5px solid {config['main_color']}; margin-bottom: 10px; }}
    .metric-value {{ font-weight: 700; font-size: 20px; color: {config['main_color']}; }}
    .header-style {{ background-color: #262730; padding: 5px; border-radius: 5px; font-weight: bold; font-size: 12px; text-align: center; border-bottom: 2px solid #444; }}
    .score-style {{ color: {config['main_color']}; font-weight: 800; font-size: 14px; text-align: center; }}
    </style>
    """, unsafe_allow_html=True)

# TABS 구성
tabs = st.tabs(["📝 작업 등록", "🗂️ 프로젝트 관리", "📈 통계 대시보드", "⚙️ 설정"])

# -----------------------------------------------------------------------------
# [TAB 0] 작업 등록
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("1️⃣ 프로젝트 기본 정보")
    with st.container(border=True):
        c_y, c_m, c1, c2, c3, c4 = st.columns([1, 0.8, 1.5, 0.8, 1.2, 0.8])
        p_year = c_y.selectbox("연도", YEAR_OPTIONS, key="reg_y")
        p_month = c_m.selectbox("월", list(range(1, 13)), index=datetime.now().month-1, key="reg_m")
        p_name = c1.text_input("프로젝트 명", placeholder="프로젝트 이름을 입력하세요")
        p_diff = c2.selectbox("난이도", list(config["diff_weights"].keys()))
        p_cat = c3.text_input("분류", placeholder="예: 유튜브, 홍보")
        p_edits = c4.number_input("전체 수정", min_value=0, step=1)

    st.subheader("2️⃣ 작업자 추가")
    with st.container(border=True):
        w1, w2, w3, w4, w5 = st.columns([1.5, 1.2, 1, 1, 0.8])
        w_name = w1.text_input("이름", key="in_name")
        w_part = w2.selectbox("파트", PART_ORDER, key="in_part")
        w_cont = w3.selectbox("기여도", list(config["cont_weights"].keys()), key="in_cont")
        w_indiv = w4.number_input("개별수정", min_value=0, step=1, key="in_indiv")
        if w5.button("추가", use_container_width=True):
            if w_name:
                st.session_state.temp_workers.append({
                    "이름": w_name, "파트": w_part, "기여도": w_cont, 
                    "수정횟수": w_indiv, "제외횟수": 0, "점수입력": 0.0
                })
                st.rerun()

    if st.session_state.temp_workers:
        st.markdown("##### 👥 등록 대기 명단")
        temp_df = pd.DataFrame(st.session_state.temp_workers)
        st.table(temp_df[["이름", "파트", "기여도", "수정횟수"]])
        if st.button("🗑️ 대기 명단 초기화"):
            st.session_state.temp_workers = []
            st.rerun()

    st.divider()
    if st.button("🚀 프로젝트 최종 저장", type="primary", use_container_width=True):
        if not p_name: st.error("프로젝트 명을 입력해주세요."); st.stop()
        if not st.session_state.temp_workers: st.error("작업자를 한 명 이상 추가해주세요."); st.stop()
        
        t_df = pd.DataFrame(st.session_state.temp_workers)
        final_df = run_score_engine(t_df, p_diff, p_edits, config)
        
        gid = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{p_name}"
        final_df['연도'] = p_year
        final_df['월'] = p_month
        final_df['프로젝트명'] = p_name
        final_df['난이도'] = p_diff
        final_df['분류'] = p_cat
        final_df['프로젝트_수정횟수'] = p_edits
        final_df['group_id'] = gid
        final_df['등록일시'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        all_df = pd.concat([all_df, final_df], ignore_index=True)
        st.session_state.temp_workers = []
        save_to_gsheets(all_df, gid=None)

# -----------------------------------------------------------------------------
# [TAB 1] 프로젝트 관리 (수정/삭제)
# -----------------------------------------------------------------------------
with tabs[1]:
    if all_df.empty:
        st.info("등록된 프로젝트가 없습니다.")
    else:
        st.subheader("📂 등록된 프로젝트 목록")
        proj_list = all_df.drop_duplicates('group_id').sort_values('등록일시', ascending=False)
        
        for _, row in proj_list.iterrows():
            with st.expander(f"📌 [{row['연도']}/{row['월']}] {row['프로젝트명']} ({row['분류']})"):
                pdf = all_df[all_df['group_id'] == row['group_id']].copy()
                
                c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1])
                new_name = c1.text_input("프로젝트명", value=row['프로젝트명'], key=f"edit_nm_{row['group_id']}")
                new_diff = c2.selectbox("난이도", list(config["diff_weights"].keys()), index=list(config["diff_weights"].keys()).index(row['난이도']) if row['난이도'] in config["diff_weights"] else 0, key=f"edit_df_{row['group_id']}")
                new_cat = c3.text_input("분류", value=row['분류'], key=f"edit_ct_{row['group_id']}")
                new_edits = c4.number_input("전체 수정", value=int(row['프로젝트_수정횟수']), key=f"edit_ed_{row['group_id']}")
                
                # 삭제 버튼
                if c5.button("🗑️ 프로젝트 삭제", key=f"del_{row['group_id']}", use_container_width=True):
                    all_df = all_df[all_df['group_id'] != row['group_id']]
                    save_to_gsheets(all_df)

                # 개별 작업자 수정
                edited_workers = []
                for i, w_row in pdf.iterrows():
                    wc1, wc2, wc3, wc4, wc5 = st.columns([1, 1, 1, 1, 1])
                    w_n = wc1.text_input("이름", value=w_row['이름'], key=f"wn_{i}")
                    w_p = wc2.selectbox("파트", PART_ORDER, index=PART_ORDER.index(w_row['파트']), key=f"wp_{i}")
                    w_c = wc3.selectbox("기여도", list(config["cont_weights"].keys()), index=list(config["cont_weights"].keys()).index(w_row['기여도']) if w_row['기여도'] in config["cont_weights"] else 0, key=f"wc_{i}")
                    w_s = wc4.number_input("개별수정", value=int(w_row['수정횟수']), key=f"ws_{i}")
                    w_ex = wc5.number_input("제외횟수", value=int(w_row.get('제외횟수', 0)), key=f"we_{i}")
                    edited_workers.append({"이름": w_n, "파트": w_p, "기여도": w_c, "수정횟수": w_s, "제외횟수": w_ex, "점수입력": 0.0})

                if st.button("💾 수정사항 반영", key=f"save_{row['group_id']}"):
                    new_pdf = pd.DataFrame(edited_workers)
                    recalc_df = run_score_engine(new_pdf, new_diff, new_edits, config)
                    recalc_df[['연도','월','프로젝트명','난이도','분류','프로젝트_수정횟수','group_id','등록일시']] = [row['연도'], row['월'], new_name, new_diff, new_cat, new_edits, row['group_id'], row['등록일시']]
                    
                    all_df = all_df[all_df['group_id'] != row['group_id']]
                    all_df = pd.concat([all_df, recalc_df], ignore_index=True)
                    save_to_gsheets(all_df)

# -----------------------------------------------------------------------------
# [TAB 2] 통계 대시보드
# -----------------------------------------------------------------------------
with tabs[2]:
    if all_df.empty:
        st.info("📊 통계를 생성할 데이터가 없습니다.")
    else:
        # 1. 필터링
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

        def format_score(val): return str(int(val)) if val == int(val) else f"{val:.2f}"
        def get_rgba(hex_color, opacity):
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'

        # 2. 요약 카드
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        proj_count = dff['group_id'].nunique()
        avg_weighted = dff['최종점수'].mean() if not dff.empty else 0
        p_rank = dff.groupby('이름')['수정횟수'].sum().sort_values(ascending=False)
        top_info = f"{p_rank.index[0]} / {p_rank.values[0]}회" if not p_rank.empty and p_rank.values[0] > 0 else "- / 0회"
        
        m_c1.markdown(f'<div class="metric-card"><div class="metric-label">총 프로젝트</div><div class="metric-value">{proj_count}건</div></div>', unsafe_allow_html=True)
        m_c2.markdown(f'<div class="metric-card"><div class="metric-label">가중점수 평균</div><div class="metric-value">{format_score(avg_weighted)}점</div></div>', unsafe_allow_html=True)
        m_c3.markdown(f'<div class="metric-card"><div class="metric-label">총 수정횟수</div><div class="metric-value">{int(dff["수정횟수"].sum())}회</div></div>', unsafe_allow_html=True)
        m_c4.markdown(f'<div class="metric-card"><div class="metric-label">최다 수정자</div><div class="metric-value">{top_info}</div></div>', unsafe_allow_html=True)

        # 3. 점수 그래프 & 4. 디자인 설정
        main_chart_spot = st.container()
        with st.expander("🎨 점수 그래프 상세 디자인 설정"):
            cl, cm, cr = st.columns([1.2, 1, 1.2])
            with cl:
                c_type = st.radio("📈 형태", ["막대형", "선형"], horizontal=True)
                f_size = st.slider("🟦 글자 크기", 10, 30, 14)
            with cm:
                f_color = st.color_picker("가중점수 색상", config['main_color'])
                b_color = st.color_picker("기본점수 색상", "#555555")
            with cr:
                d_type = st.selectbox("✨ 디자인", ["기본형", "타입 A"])
                thickness = st.slider("📏 두께", 0.1, 1.0, 0.7)

        with main_chart_spot:
            if not dff.empty:
                chart_df = dff.groupby(target_col)[['기본점수', '최종점수']].sum().reset_index()
                fig = go.Figure()
                fixed_font = dict(size=f_size, color="white")
                if c_type == "막대형":
                    fig.add_trace(go.Bar(x=chart_df[target_col], y=chart_df['기본점수'], name='기본점수', marker_color=b_color, text=chart_df['기본점수'].apply(format_score), textposition='outside', textfont=fixed_font))
                    fig.add_trace(go.Bar(x=chart_df[target_col], y=chart_df['최종점수'], name='가중점수', marker_color=f_color, text=chart_df['최종점수'].apply(format_score), textposition='outside', textfont=fixed_font))
                    fig.update_layout(barmode='group' if d_type=="기본형" else 'overlay', bargap=1.0-thickness)
                else:
                    fig.add_trace(go.Scatter(x=chart_df[target_col], y=chart_df['기본점수'], name='기본점수', mode='lines+markers+text', line=dict(color=b_color, width=thickness*10), text=chart_df['기본점수'].apply(format_score), textposition='top center', textfont=fixed_font))
                    fig.add_trace(go.Scatter(x=chart_df[target_col], y=chart_df['최종점수'], name='가중점수', mode='lines+markers+text', line=dict(color=f_color, width=thickness*10), text=chart_df['최종점수'].apply(format_score), textposition='bottom center', textfont=fixed_font))
                fig.update_layout(template="plotly_dark", height=500, margin=dict(t=50, b=50, l=50, r=50))
                st.plotly_chart(fig, use_container_width=True)

        # 5. 수정 횟수 TOP 5 & 6. 디자인 설정
        st.divider()
        st.subheader("🚩 수정 횟수 TOP 5 분석")
        top_chart_spot = st.container()
        with st.expander("🎨 TOP 5 그래프 상세 디자인 설정"):
            tc1, tc2 = st.columns(2)
            t_f_size = tc1.slider("🟦 TOP 5 글자 크기", 10, 30, 14)
            t_color_p = tc2.color_picker("막대 색상", "#E84D4D")

        with top_chart_spot:
            if not dff.empty:
                col1, col2 = st.columns(2)
                top_p = dff.drop_duplicates('group_id').query("프로젝트_수정횟수 > 0").nlargest(5, '프로젝트_수정횟수')
                top_w = dff.groupby('이름')['수정횟수'].sum().reset_index().query("수정횟수 > 0").nlargest(5, '수정횟수')
                
                with col1:
                    st.markdown("##### 📂 프로젝트별 TOP 5")
                    if not top_p.empty:
                        fig_p = go.Figure(go.Bar(x=top_p['프로젝트명'], y=top_p['프로젝트_수정횟수'], marker_color=t_color_p, text=top_p['프로젝트_수정횟수'], textposition='outside', textfont=dict(size=t_f_size, color="white")))
                        fig_p.update_layout(template="plotly_dark", height=400, yaxis=dict(title="수정 횟수", showgrid=True, zeroline=True, zerolinecolor='white'))
                        st.plotly_chart(fig_p, use_container_width=True)
                with col2:
                    st.markdown("##### 👤 작업자별 TOP 5")
                    if not top_w.empty:
                        fig_w = go.Figure(go.Bar(x=top_w['이름'], y=top_w['수정횟수'], marker_color="#FFA500", text=top_w['수정횟수'], textposition='outside', textfont=dict(size=t_f_size, color="white")))
                        fig_w.update_layout(template="plotly_dark", height=400, yaxis=dict(title="총 수정 횟수", showgrid=True, zeroline=True, zerolinecolor='white'))
                        st.plotly_chart(fig_w, use_container_width=True)

# -----------------------------------------------------------------------------
# [TAB 3] 설정
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("⚙️ 시스템 환경 설정")
    col_diff, col_cont = st.columns(2)
    with col_diff:
        with st.container(border=True):
            st.subheader("📊 난이도 관리")
            with st.expander("➕ 항목 추가"):
                ad1, ad2, ad3 = st.columns([1,1,1])
                nk = ad1.text_input("명칭", key="nk")
                nv = ad2.number_input("가중치", value=1.0, key="nv")
                if ad3.button("추가", key="ab1"):
                    config["diff_weights"][nk] = nv
                    save_to_gsheets(all_df, config_data=config)
            for k in list(config["diff_weights"].keys()):
                r1, r2, r3 = st.columns([2,2,1])
                r1.write(f"**{k}**")
                config["diff_weights"][k] = r2.number_input("값", value=float(config["diff_weights"][k]), key=f"dv_{k}", label_visibility="collapsed")
                if r3.button("🗑️", key=f"dk_{k}"):
                    del config["diff_weights"][k]
                    save_to_gsheets(all_df, config_data=config)

    with col_cont:
        with st.container(border=True):
            st.subheader("💡 기여도 관리")
            with st.expander("➕ 항목 추가"):
                ac1, ac2, ac3 = st.columns([1,1,1])
                ck = ac1.text_input("명칭", key="ck")
                cv = ac2.number_input("가중치", value=1.0, key="cv")
                if ac3.button("추가", key="ab2"):
                    config["cont_weights"][ck] = cv
                    save_to_gsheets(all_df, config_data=config)
            for k in list(config["cont_weights"].keys()):
                r1, r2, r3 = st.columns([2,2,1])
                r1.write(f"**{k}**")
                config["cont_weights"][k] = r2.number_input("값", value=float(config["cont_weights"][k]), key=f"cv_{k}", label_visibility="collapsed")
                if r3.button("🗑️", key=f"ck_{k}"):
                    del config["cont_weights"][k]
                    save_to_gsheets(all_df, config_data=config)

    st.divider()
    s1, s2, s3 = st.columns([1, 1, 1])
    new_penalty = s1.number_input("📉 감점률", value=float(config["penalty_rate"]), step=0.01)
    new_color = s2.color_picker("🎨 메인 컬러", value=config["main_color"])
    if s3.button("💾 모든 설정 저장 및 데이터 재계산", type="primary", use_container_width=True):
        config["penalty_rate"] = new_penalty
        config["main_color"] = new_color
        save_to_gsheets(all_df, config_data=config)
