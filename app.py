import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection

# 1. 페이지 설정
st.set_page_config(page_title="MNK 성과관리 시스템", layout="wide")

# 2. 구글 시트 연결
conn = st.connection("gsheets", type=GSheetsConnection)

# 3. 데이터 로드/저장 함수 (보강됨)
def load_data():
    try:
        return conn.read(worksheet="Data", ttl=0).dropna(how="all")
    except Exception as e:
        return pd.DataFrame()

def load_config():
    try:
        df_cfg = conn.read(worksheet="Config", ttl=0)
        row = df_cfg.iloc[0].to_dict()
        return {
            "diff_weights": json.loads(row["diff_weights"]),
            "cont_weights": json.loads(row["cont_weights"]),
            "penalty_rate": float(row["penalty_rate"]),
            "main_color": row["main_color"]
        }
    except:
        return {"diff_weights": {"S": 2.0, "A": 1.5, "B": 1.0, "C": 0.7}, "cont_weights": {"상": 1.2, "중": 1.0, "하": 0.8}, "penalty_rate": 0.05, "main_color": "#00FFD1"}

def save_to_gsheets(df, config_data=None):
    try:
        # 시트에 쓰기 전에 데이터 형식을 강제로 맞춤
        df_to_save = df.copy()
        conn.update(worksheet="Data", data=df_to_save)
        
        if config_data:
            cfg_df = pd.DataFrame([{"diff_weights": json.dumps(config_data["diff_weights"]), "cont_weights": json.dumps(config_data["cont_weights"]), "penalty_rate": config_data["penalty_rate"], "main_color": config_data["main_color"]}])
            conn.update(worksheet="Config", data=cfg_df)
        
        st.cache_data.clear()
        st.success("데이터가 안전하게 저장되었습니다!")
        st.rerun()
    except Exception as e:
        st.error(f"저장 중 오류가 발생했습니다. 구글 시트의 컬럼명과 개수를 확인해주세요. 에러내용: {e}")

# =============================================================================
# [PART 2] 점수 계산 엔진
# =============================================================================

def run_score_engine(project_df, p_diff, p_total_edits, cfg):
    df = project_df.copy()
    if len(df) == 0: return df
    
    df['공통수정분'] = 0.0
    p_total_edits = float(p_total_edits)
    
    # 1. 파트별 기본 점수 배분
    mkt_mask = df['파트'] == "마케팅"
    mkt_sum = df.loc[mkt_mask, '점수입력'].sum()
    design_mask = df['파트'] == "디자인컷"
    rem_pool = max(0, 100.0 - mkt_sum)

    if design_mask.sum() > 0:
        df.loc[design_mask, '점수입력'] = round(rem_pool / design_mask.sum(), 2)
    else:
        prod_mask = ~df['파트'].isin(["마케팅", "디자인컷"])
        total_w = sum([cfg["cont_weights"].get(row['기여도'], 1.0) for _, row in df[prod_mask].iterrows()])
        if total_w > 0:
            unit = rem_pool / total_w
            for idx in df[prod_mask].index:
                df.at[idx, '점수입력'] = round(unit * cfg["cont_weights"].get(df.at[idx, '기여도'], 1.0), 2)

    # 2. 감점 로직 (공통수정 + 개별수정)
    total_n = len(df)
    diff_w = cfg["diff_weights"].get(p_diff, 1.0)
    for idx in df.index:
        raw = df.at[idx, '점수입력']
        total_resp = (p_total_edits / total_n) + float(df.at[idx, '수정횟수'])
        penalty = round(raw * (total_resp * cfg["penalty_rate"]), 2)
        df.at[idx, '기본점수'] = round(raw, 2)
        df.at[idx, '감점점수'] = penalty
        df.at[idx, '최종점수'] = round(max(0, raw - penalty) * diff_w, 2)
    return df

# 데이터 및 설정 불러오기
config = load_config()
all_df = load_data()

# CSS 디자인
st.markdown(f"""
    <style>
    .metric-card {{ background-color: #2D2D3A; padding: 15px; border-radius: 10px; border-left: 5px solid {config['main_color']}; margin-bottom: 10px; }}
    .metric-value {{ font-weight: 700; font-size: 20px; color: {config['main_color']}; }}
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# [PART 3] 메인 UI (TABS)
# =============================================================================

tabs = st.tabs(["📝 작업 등록", "🗂️ 프로젝트 관리", "📈 통계 대시보드", "⚙️ 설정"])

# [TAB 0] 작업 등록
with tabs[0]:
    st.subheader("1️⃣ 프로젝트 기본 정보")
    with st.container(border=True):
        c_y, c_m, c1, c2, c3, c4 = st.columns([1, 0.8, 1.5, 0.8, 1.2, 0.8])
        p_year = c_y.selectbox("연도", YEAR_OPTIONS, key="reg_y")
        p_month = c_m.selectbox("월", list(range(1, 13)), index=datetime.now().month-1, key="reg_m")
        p_name = c1.text_input("프로젝트 명")
        p_diff = c2.selectbox("난이도", list(config["diff_weights"].keys()))
        p_cat = c3.text_input("분류")
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
                    "이름": w_name, "파트": w_part, "기여도": w_cont, "수정횟수": w_indiv, "제외횟수": 0, "점수입력": 0.0
                })
                st.rerun()

    if st.session_state.temp_workers:
        st.table(pd.DataFrame(st.session_state.temp_workers)[["이름", "파트", "기여도", "수정횟수"]])
        if st.button("🚀 프로젝트 최종 저장", type="primary", use_container_width=True):
            t_df = pd.DataFrame(st.session_state.temp_workers)
            final_df = run_score_engine(t_df, p_diff, p_edits, config)
            gid = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{p_name}"
            final_df[['연도','월','프로젝트명','난이도','분류','프로젝트_수정횟수','group_id','등록일시']] = [p_year, p_month, p_name, p_diff, p_cat, p_edits, gid, datetime.now().strftime("%Y-%m-%d %H:%M")]
            all_df = pd.concat([all_df, final_df], ignore_index=True)
            st.session_state.temp_workers = []
            save_to_gsheets(all_df, gid=None)

# [TAB 1] 관리 (삭제 기능 위주)
with tabs[1]:
    if not all_df.empty:
        proj_list = all_df.drop_duplicates('group_id').sort_values('등록일시', ascending=False)
        for _, row in proj_list.iterrows():
            with st.expander(f"📌 {row['프로젝트명']} ({row['연도']}/{row['월']})"):
                st.write(all_df[all_df['group_id'] == row['group_id']])
                if st.button("🗑️ 삭제", key=f"del_{row['group_id']}"):
                    all_df = all_df[all_df['group_id'] != row['group_id']]
                    save_to_gsheets(all_df)

# [TAB 2] 통계
with tabs[2]:
    if not all_df.empty:
        chart_df = all_df.groupby('이름')[['기본점수', '최종점수']].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=chart_df['이름'], y=chart_df['최종점수'], marker_color=config['main_color'], text=chart_df['최종점수'], textposition='outside'))
        fig.update_layout(template="plotly_dark", title="작업자별 최종 합산 점수")
        st.plotly_chart(fig, use_container_width=True)

# [TAB 3] 설정
with tabs[3]:
    st.subheader("⚙️ 가중치 및 설정")
    new_penalty = st.number_input("감점률", value=config['penalty_rate'], step=0.01)
    new_color = st.color_picker("시스템 메인 컬러", value=config['main_color'])
    if st.button("💾 설정 저장"):
        config['penalty_rate'] = new_penalty
        config['main_color'] = new_color
        save_to_gsheets(all_df, config_data=config)
