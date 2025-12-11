import streamlit as st
import plotly.graph_objects as go
import requests
import json

# -----------------------------------------------------------------------------
# 설정: Azure Function URL (로컬 테스트 시 보통 http://localhost:7071/api/함수이름)
# 배포 후에는 실제 Azure URL로 변경해야 합니다.
# -----------------------------------------------------------------------------
AZURE_FUNCTION_URL = "https://stu456-game-api.azurewebsites.net/api/httptrigger1" 
# 주의: Azure Function 함수 이름이 HttpTrigger1 이라고 가정함

st.set_page_config(layout="wide", page_title="GameDev AI: Success Predictor")

# 스타일 CSS (이전과 동일)
st.markdown("""
<style>
    .stApp { background-color: #0b1120; color: white; }
    div[data-testid="stVerticalBlock"] > div { background-color: #151e32; padding: 20px; border-radius: 15px; border: 1px solid #2a3b55; }
    h1, h2, h3 { color: #e0e0e0 !important; }
    .stButton > button { width: 100%; background-color: #1e293b; color: #00bfff; border: 1px solid #00bfff; height: 60px; font-weight: bold; }
    .stButton > button:hover { background-color: #00bfff; color: #0b1120; box-shadow: 0 0 10px #00bfff; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🎮 GameDev AI: Success Predictor (Azure + Steam)")
st.markdown("---")

col1, col2, col3 = st.columns([1, 1.2, 1.2])

# --- [왼쪽] Model Selection ---
with col1:
    st.markdown("### Model Selection")
    st.write("")
    model_choice = st.radio(
        "Choose a Model:",
        ["XGBoost (Pro)", "Random Forest", "Logistic Regression"],
        label_visibility="collapsed"
    )
    if model_choice == "XGBoost (Pro)":
        st.info("✅ **XGBoost Selected**\n\nAzure Cloud에서 고속 연산 처리됩니다.")

# --- [가운데] Input Parameters ---
with col2:
    st.markdown("### Input Parameters")
    genre = st.selectbox("Genre", ["RPG", "FPS", "Simulation", "Strategy"])
    budget = st.slider("Budget ($1,000s)", 10, 5000, 350)
    
    # [NEW] Steam API 연동을 위한 입력값
    st.markdown("#### Competitor Analysis (Steam Data)")
    st.caption("비슷한 장르의 경쟁 게임 Steam App ID를 입력하여 시장 데이터를 반영합니다.")
    # 기본값은 'Among Us'의 App ID (945360)
    competitor_id = st.text_input("Competitor App ID", value="945360") 
    
    predict_btn = st.button("🚀 Analyze with Azure Cloud", type="primary")

# --- [오른쪽] Prediction Results ---
with col3:
    st.markdown("### Prediction Results")

    final_score = 0
    steam_players = 0
    
    if predict_btn:
        with st.spinner('Connecting to Azure Cloud & Fetching Steam Data...'):
            try:
                # 1. Azure Function으로 데이터 전송
                payload = {
                    "model": model_choice,
                    "budget": budget,
                    "genre": genre,
                    "competitor_id": competitor_id
                }
                
                # 실제 API 호출 (타임아웃 설정)
                # 배포 전 테스트할 땐 Azure Function을 로컬에서 켜두어야 합니다.
                response = requests.post(AZURE_FUNCTION_URL, json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    final_score = result.get("success_prob", 0)
                    steam_players = result.get("competitor_players", 0)
                    st.toast(f"Steam API Success: Found {steam_players:,} active players!", icon="🎮")
                else:
                    st.error("Azure Function Error")
                    
            except Exception as e:
                st.error(f"Connection Failed: {e}")
                st.caption("Azure Function이 실행 중인지 확인하세요.")

    # 차트 그리기 (결과가 있을 때만 업데이트, 초기값 0)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = final_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': "%", 'font': {'color': "white", 'size': 40}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00ff9d"},
            'bgcolor': "#1e293b",
            'borderwidth': 2,
            'bordercolor': "#333"
        }
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=250, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    if final_score > 0:
        st.info(f"Analysis based on competitor's **{steam_players:,}** concurrent players.")
