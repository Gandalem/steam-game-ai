import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import requests
import os

# -----------------------------------------------------------------------------
# ⚙️ 설정 & 모델 로드 (핵심 변경!)
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="게임 성공 예측 AI")

# AI 모델을 캐싱하여 속도 최적화 (매번 다시 로딩하지 않음)
@st.cache_resource
def load_models():
    # 같은 폴더에 있는 pkl 파일 로드
    return joblib.load('all_game_models.pkl')

try:
    models = load_models()
    model_loaded = True
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다: {e}")
    model_loaded = False

# -----------------------------------------------------------------------------
# 🎨 CSS 디자인
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f1f5f9; font-family: 'Malgun Gothic', sans-serif; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }
    div[data-testid="column"] { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }
    h1 { font-size: 1.8rem !important; color: #0f172a; margin-bottom: 0; }
    .stButton > button { width: 100%; background-color: #2563eb; color: white; border-radius: 8px; border: none; padding: 0.6rem 1rem; margin-top: 15px; font-size: 1rem; font-weight: bold; }
    .stButton > button:hover { background-color: #1d4ed8; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 메인 화면
# -----------------------------------------------------------------------------
c1, c2 = st.columns([3, 1])
with c1:
    st.title("🎮 GameDev AI: 성공 예측 솔루션")
    st.caption("Machine Learning & Steam Data Analytics")
with c2:
    if model_loaded:
        st.markdown("<div style='text-align:right; color:green; font-weight:bold; margin-top:10px;'>🟢 AI 시스템 준비완료</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:right; color:red; font-weight:bold; margin-top:10px;'>🔴 모델 파일 없음</div>", unsafe_allow_html=True)

st.write("") 

col1, col2, col3 = st.columns([1, 1.2, 1.5], gap="medium")

# --- [왼쪽] 모델 선택 ---
with col1:
    st.subheader("🛠 모델 설정")
    model_choice = st.radio("사용할 알고리즘", ["XGBoost (Pro)", "Random Forest", "Logistic Regression"])
    st.markdown("---")
    st.info(f"**선택됨:** {model_choice}")
    st.caption("로컬 환경에서 즉시 연산됩니다.")

# --- [가운데] 입력 파라미터 ---
with col2:
    st.subheader("📝 파라미터 입력")
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        genre = st.selectbox("장르", ["전략", "RPG", "FPS", "시뮬레이션", "퍼즐"])
    with c_sub2:
        platform = st.selectbox("플랫폼", ["PC", "모바일", "콘솔", "웹"])
        
    budget = st.slider("예산 ($1,000)", 10, 5000, 350)
    team_size = st.number_input("팀 규모 (명)", 1, 200, 10)
    
    st.markdown("---")
    competitor_id = st.text_input("경쟁작 App ID", value="945360", help="Steam App ID 입력") 
    predict_btn = st.button("🚀 분석 시작", type="primary")

# --- [오른쪽] 결과 대시보드 ---
with col3:
    st.subheader("📊 분석 결과")

    final_score = 0
    steam_players = 0
    
    if not predict_btn:
        st.info("좌측 데이터를 입력하고 분석 버튼을 누르세요.")
    else:
        if not model_loaded:
            st.error("모델 파일(all_game_models.pkl)이 없습니다.")
        else:
            with st.spinner('AI가 데이터를 분석 중입니다...'):
                # 1. Steam API 직접 호출
                try:
                    url = f"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={competitor_id}"
                    resp = requests.get(url, timeout=3).json()
                    if resp['response']['result'] == 1:
                        steam_players = resp['response']['player_count']
                except:
                    steam_players = 0 # 실패 시 0명 처리

                # 2. 입력 데이터 가공
                genre_map = {"전략": 1, "RPG": 2, "FPS": 3, "시뮬레이션": 4, "퍼즐": 5}
                g_code = genre_map.get(genre, 6)
                
                # [장르, 예산, 팀규모, 경쟁작동접자]
                input_data = [[g_code, budget, team_size, steam_players]]

                # 3. 모델 예측 (Azure 없이 직접 수행)
                if "XGBoost" in model_choice:
                    final_score = models['xgb'].predict(input_data)[0]
                elif "Random Forest" in model_choice:
                    final_score = models['rf'].predict(input_data)[0]
                else:
                    final_score = models['lr'].predict_proba(input_data)[0][1] * 100
                
                final_score = float(np.clip(final_score, 0, 100))

            # 결과 표시
            m1, m2 = st.columns(2)
            m1.metric("성공 확률", f"{final_score:.1f}%")
            m2.metric("경쟁작 동접자", f"{steam_players:,}명")

            # 게이지 차트
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = final_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#2563eb"}}
            ))
            fig.update_layout(height=200, margin=dict(t=30,b=10,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

            if final_score >= 70:
                st.success("매우 긍정적인 프로젝트입니다!")
            else:
                st.warning("리스크가 감지되었습니다.")
