import streamlit as st
import plotly.graph_objects as go
import requests
import json

# -----------------------------------------------------------------------------
# ⚙️ 설정: Azure Function URL
# (아까 배포 성공한 주소를 여기에 그대로 두세요)
# -----------------------------------------------------------------------------
AZURE_FUNCTION_URL = "https://stu456-game-api.azurewebsites.net/api/HttpTrigger1?code=euPgWVAwL_-v3RWH8iDu804DVzCAb-ptsOfeowcWTiHFAzFuQzSXOA==" 

st.set_page_config(layout="wide", page_title="GameDev AI: Success Predictor")

# -----------------------------------------------------------------------------
# 🎨 UI 디자인 (Light Mode - 화이트 & 블루 테마)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* 전체 배경: 아주 연한 회색 */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* 컨테이너(카드) 스타일: 흰색 배경 + 그림자 효과 */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
    }

    /* 헤더 텍스트 색상 */
    h1, h2, h3 {
        color: #1a202c !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* 입력 필드 라벨 색상 */
    label, .stMarkdown p {
        color: #4a5568 !important;
        font-weight: 500;
    }

    /* 버튼 스타일 */
    .stButton > button {
        width: 100%;
        background-color: #3b82f6; /* 밝은 블루 */
        color: white;
        border: none;
        border-radius: 8px;
        height: 55px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.2s;
    }
    
    /* 버튼 호버 효과 */
    .stButton > button:hover {
        background-color: #2563eb;
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* 성공 확률 텍스트 박스 */
    .result-box {
        background-color: #eff6ff;
        color: #1e40af;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-top: 10px;
        border: 1px solid #bfdbfe;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 헤더 영역
# -----------------------------------------------------------------------------
st.title("🎮 GameDev AI: Success Predictor")
st.markdown("Azure Cloud & Steam Data 기반 게임 성공 예측 솔루션")
st.markdown("---")

# -----------------------------------------------------------------------------
# 메인 레이아웃 (3단 컬럼)
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 1.2, 1.2])

# --- [왼쪽] Model Selection ---
with col1:
    st.subheader("Model Selection")
    st.write("")
    
    # 모델 선택
    model_choice = st.radio(
        "Choose a Model:",
        ["XGBoost (Pro)", "Random Forest", "Logistic Regression"]
    )
    
    st.write("---")
    
    # 선택된 모델 설명 (밝은 색상 박스)
    if model_choice == "XGBoost (Pro)":
        st.info("✅ **XGBoost Selected**\n\n속도와 성능이 가장 우수한 부스팅 모델입니다. Azure 서버에서 고속 연산됩니다.")
    elif model_choice == "Random Forest":
        st.success("✅ **Random Forest Selected**\n\n안정적인 예측력을 가진 앙상블 모델입니다.")
    else:
        st.warning("✅ **Logistic Regression Selected**\n\n데이터의 선형적인 관계를 분석합니다.")

# --- [가운데] Input Parameters ---
with col2:
    st.subheader("Input Parameters")
    
    genre = st.selectbox("Genre", ["Strategy", "RPG", "FPS", "Simulation", "Puzzle", "Casual"])
    budget = st.slider("Budget ($1,000s)", 10, 5000, 350)
    team_size = st.number_input("Team Size", min_value=1, max_value=200, value=10)
    platform = st.selectbox("Platform", ["PC (Steam)", "Mobile", "Console", "Web"])
    
    st.write("")
    st.markdown("#### Competitor Analysis (Steam Data)")
    st.caption("경쟁 게임의 Steam App ID를 입력하면 실시간 동시 접속자 데이터를 반영합니다.")
    
    competitor_id = st.text_input("Competitor App ID", value="945360") 
    
    st.write("")
    predict_btn = st.button("🚀 Analyze with Azure Cloud")

# --- [오른쪽] Prediction Results ---
with col3:
    st.subheader("Prediction Results")

    final_score = 0
    steam_players = 0
    
    # 버튼 클릭 시 실행
    if predict_btn:
        with st.spinner('Connecting to Azure Cloud...'):
            try:
                # Azure Function 호출
                payload = {
                    "model": model_choice,
                    "budget": budget,
                    "genre": genre,
                    "competitor_id": competitor_id
                }
                
                # 타임아웃 10초 설정
                response = requests.post(AZURE_FUNCTION_URL, json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    final_score = result.get("success_prob", 0)
                    steam_players = result.get("competitor_players", 0)
                    st.toast("Analysis Complete!", icon="✅")
                else:
                    st.error(f"Azure Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Connection Failed. URL을 확인하세요.")
                st.caption(f"{e}")

    # 1. 게이지 차트 (Light Mode용 색상 적용)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = final_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': "%", 'font': {'color': "#1a202c", 'size': 45}}, # 검정 텍스트
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#a0aec0"},
            'bar': {'color': "#3b82f6"},  # 밝은 파랑
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#cbd5e0",
            'steps': [
                {'range': [0, 100], 'color': '#f1f5f9'} # 아주 연한 회색 배경
            ],
        }
    ))
    
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#2d3748"},
        height=250,
        margin=dict(t=30,b=10,l=20,r=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 결과 텍스트 표시
    if final_score > 0:
        if final_score >= 80:
            msg = "High Likelihood of Success"
            color_box = "#dcfce7" # 연한 초록
            text_color = "#166534"
        elif final_score >= 50:
            msg = "Moderate Likelihood"
            color_box = "#fef9c3" # 연한 노랑
            text_color = "#854d0e"
        else:
            msg = "Low Likelihood"
            color_box = "#fee2e2" # 연한 빨강
            text_color = "#991b1b"
            
        st.markdown(f"""
        <div style="background-color:{color_box}; color:{text_color}; padding:15px; border-radius:10px; text-align:center; font-weight:bold; margin-bottom:15px;">
            {msg}
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"Analysis includes real-time data from **{steam_players:,}** active players on Steam.")

    # 2. 바 차트 (Light Mode용)
    if final_score > 0:
        models = ['XGBoost', 'Random Forest', 'Logistic Reg']
        # 예시용 비교 데이터 (실제로는 다르게 계산 가능)
        scores = [final_score, final_score-5, final_score-12]
        colors = ['#3b82f6', '#60a5fa', '#93c5fd'] # 블루 계열 그라데이션

        fig_bar = go.Figure(data=[go.Bar(
            x=models,
            y=scores,
            marker_color=colors
        )])

        fig_bar.update_layout(
            title="Model Comparison",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#4a5568'),
            yaxis=dict(range=[0, 100], showgrid=True, gridcolor='#e2e8f0'),
            height=200,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
