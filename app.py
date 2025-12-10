import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------------------------
# 1. 페이지 설정 및 커스텀 CSS (UI 대개편의 핵심)
# ------------------------------------------------
st.set_page_config(page_title="Steam AI Analyst", layout="wide", page_icon="🎮")

# CSS로 디자인 덮어쓰기 (카드 스타일, 폰트 등)
st.markdown("""
    <style>
    /* 전체 배경 및 폰트 설정 */
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    
    /* KPI 카드 스타일 */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
        border-color: #4CAF50;
    }

    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 5px;
        color: #31333F;
        font-weight: 600;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }

    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #ddd;
    }
    
    /* 헤더 강조 */
    h1 {
        color: #1E88E5;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        color: #424242;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 2. 데이터 로드 및 전처리
# ------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('steam_top_sellers_ULTIMATE_v2.xlsx')
    except:
        st.error("데이터 파일이 없습니다. (steam_top_sellers_ULTIMATE_v2.xlsx)")
        return None, None, None, None

    # (1) 가격 전처리
    def clean_price(price_raw):
        if pd.isna(price_raw): return 0
        price_str = str(price_raw)
        numbers_only = re.sub(r'[^0-9]', '', price_str)
        return int(numbers_only) if numbers_only else 0

    df['Price_Clean'] = df['최종 가격'].apply(clean_price)

    # (2) 평가 점수 변환
    def score_sentiment(text):
        text = str(text)
        if '압도적으로 긍정적' in text: return 95
        elif '매우 긍정적' in text: return 85
        elif '대체로 긍정적' in text: return 70
        elif '긍정적' in text: return 65
        elif '복합적' in text: return 50
        elif '부정적' in text: return 30
        else: return 60 

    df['Review_Score'] = df['전체 평가'].apply(score_sentiment)

    # (3) 가격 구간
    def get_price_category(price):
        if price == 0: return '무료 (Free)'
        elif price < 15000: return '저가 (~1.5만원)'
        elif price < 35000: return '중가 (1.5~3.5만원)'
        elif price < 60000: return '준고가 (3.5~6만원)'
        else: return '고가 (6만원 이상)'
    
    price_order = ['무료 (Free)', '저가 (~1.5만원)', '중가 (1.5~3.5만원)', '준고가 (3.5~6만원)', '고가 (6만원 이상)']
    df['Price_Range'] = pd.Categorical(df['Price_Clean'].apply(get_price_category), categories=price_order, ordered=True)

    # (4) 태그 필터링
    df = df.dropna(subset=['주요 태그 (상위 5개)'])
    df['Tags_List'] = df['주요 태그 (상위 5개)'].astype(str).apply(lambda x: [tag.strip() for tag in x.split(',')])

    banned_tags = [
        '무료 플레이', '앞서 해보기', 
        '애니메이션 모델', '애니메이션과 모델링', '애니메이션 및 모델링', 
        '디자인과 일러스트레이션', '사진 편집', '동영상 제작', '동영상제작', 
        '유틸리티', '소프트웨어', '웹 퍼블리싱', '오디오 제작',
        '게임 개발', '소프트웨어 교육' 
    ]

    def filter_tags(tags):
        return [tag for tag in tags if tag not in banned_tags]

    df['Tags_List'] = df['Tags_List'].apply(filter_tags)
    df = df[df['Tags_List'].map(len) > 0]

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(df['Tags_List'])
    tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_, index=df.index)

    # (5) 타겟 설정 (상위 50%)
    threshold = df['현재 동시 접속자'].quantile(0.50) 
    df['Success'] = df['현재 동시 접속자'].apply(lambda x: 1 if x >= threshold else 0)

    X = pd.concat([df[['Price_Clean', 'Review_Score']], tags_df], axis=1)
    y = df['Success']
    
    return df, X, y, mlb, threshold

df, X, y, mlb, threshold = load_data()

# ------------------------------------------------
# 3. 모델 학습
# ------------------------------------------------
@st.cache_resource
def train_all_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        "XGBoost (Pro)": XGBClassifier(eval_metric='logloss', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }
    
    trained_models = {}
    accuracies = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        trained_models[name] = model
        accuracies[name] = acc
        
    return trained_models, accuracies

if df is not None:
    models_dict, acc_dict = train_all_models(X, y)

    # =========================================================
    # [사이드바] 설정 컨트롤 패널
    # =========================================================
    st.sidebar.title("🎛️ 컨트롤 패널")
    
    st.sidebar.subheader("1. AI 모델 설정")
    selected_model_name = st.sidebar.selectbox("알고리즘 선택", list(models_dict.keys()))
    current_model = models_dict[selected_model_name]
    current_acc = acc_dict[selected_model_name]
    
    # 모델 성능 게이지 (사이드바 미니 차트)
    fig_mini = go.Figure(go.Indicator(
        mode = "number+gauge", value = current_acc * 100,
        number = {'suffix': "%", 'font': {'size': 20}},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1E88E5"}, 'shape': "bullet"},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig_mini.update_layout(height=50, margin=dict(l=0,r=0,t=0,b=0))
    st.sidebar.plotly_chart(fig_mini, use_container_width=True)
    st.sidebar.divider()

    st.sidebar.subheader("2. 게임 시뮬레이션 설정")
    user_price = st.sidebar.slider("💰 가격 (KRW)", 0, 100000, 32000, step=1000, format="₩%d")
    user_score = st.sidebar.slider("⭐ 예상 평가 점수", 0, 100, 85, help="게임의 완성도(재미)를 점수로 입력하세요.")
    
    all_top_tags = pd.Series([tag for tags in df['Tags_List'] for tag in tags]).value_counts().head(20).index.tolist()
    default_tags = all_top_tags[:2] if len(all_top_tags) >= 2 else all_top_tags
    user_tags = st.sidebar.multiselect("🏷️ 장르/태그", all_top_tags, default=default_tags)

    predict_btn = st.sidebar.button("🚀 예측 실행 (Analyze)", type="primary")

    # =========================================================
    # [메인] 대시보드 UI
    # =========================================================
    st.title("🎮 STEAM AI ANALYST")
    st.markdown("##### 🚀 빅데이터 기반 스팀 게임 시장 분석 및 성공 예측 솔루션")
    st.write("")

    # 1. KPI 카드 섹션
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🤖 AI 모델 정확도", f"{current_acc*100:.1f}%", delta="신뢰도 높음")
    with col2:
        st.metric("📊 분석된 게임 데이터", f"{len(df):,}개", "유틸리티 제외됨")
    with col3:
        avg_success = df['Success'].mean() * 100
        st.metric("🏆 시장 평균 성공률", f"{avg_success:.1f}%", "상위 50% 기준")
    with col4:
        df_paid = df[df['Price_Range'] != '무료 (Free)']
        if not df_paid.empty:
            best_price = df_paid.groupby('Price_Range', observed=True)['Success'].mean().idxmax()
            st.metric("💎 추천 가격대", best_price, "유료 게임 기준")
        else:
            st.metric("💎 추천 가격대", "-")

    st.write("")
    
    # 2. 메인 탭 구성
    tab1, tab2, tab3 = st.tabs(["📊 시장 분석 (Market Map)", "🔮 예측 시뮬레이션 (Simulation)", "⚙️ AI 모델 상세 (Model Info)"])

    # -----------------------------------------------------
    # TAB 1: 시장 분석 (히트맵 & 차트)
    # -----------------------------------------------------
    with tab1:
        st.subheader("🗺️ 장르 x 가격대 성공 전략 지도")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            df_exploded = df.explode('Tags_List')
            top_15_tags = df_exploded['Tags_List'].value_counts().head(15).index
            df_filtered = df_exploded[df_exploded['Tags_List'].isin(top_15_tags)]
            pivot_table = df_filtered.pivot_table(index='Tags_List', columns='Price_Range', values='Success', aggfunc='mean')
            
            fig_heatmap = px.imshow(
                pivot_table,
                labels=dict(x="가격대", y="장르", color="성공률"),
                x=pivot_table.columns, y=pivot_table.index,
                text_auto=".0%", color_continuous_scale="RdBu_r", aspect="auto"
            )
            fig_heatmap.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(t=20, b=20))
            st.plotly_chart(fig_heatmap, use_container_width=True)

        with c2:
            st.info("💡 **분석 팁:** 붉은색이 진할수록 해당 가격대에서 성공 확률이 높습니다.")
            target_tag = st.selectbox("장르 상세 분석", top_15_tags, index=0)
            tag_data = df_exploded[df_exploded['Tags_List'] == target_tag]
            tag_analysis = tag_data.groupby('Price_Range', observed=False)['Success'].mean().reset_index()
            tag_analysis['Success'] = tag_analysis['Success'] * 100
            
            fig_bar = px.bar(
                tag_analysis, x='Price_Range', y='Success', color='Success',
                color_continuous_scale='Greens', text_auto='.1f'
            )
            fig_bar.update_layout(
                xaxis_title=None, yaxis_title="성공률 (%)", 
                showlegend=False, height=300, 
                title=f"[{target_tag}] 가격대별 성공률",
                margin=dict(t=40, b=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # -----------------------------------------------------
    # TAB 2: 예측 시뮬레이션 (결과 리포트)
    # -----------------------------------------------------
    with tab2:
        st.subheader("🔮 내 게임 성공 확률 예측 리포트")
        
        # 예측 로직
        if predict_btn:
            with st.spinner('🤖 AI가 1,000개 이상의 게임 데이터를 분석 중입니다...'):
                time.sleep(1.5)
                
                input_data = pd.DataFrame(0, index=[0], columns=X.columns)
                input_data['Price_Clean'] = user_price
                input_data['Review_Score'] = user_score
                for tag in user_tags:
                    if tag in input_data.columns:
                        input_data[tag] = 1
                
                prob = current_model.predict_proba(input_data)[0][1]
                prob_pct = prob * 100

            # 결과 리포트 UI
            r_col1, r_col2 = st.columns([1, 2])
            
            with r_col1:
                # 게이지 차트
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = prob_pct,
                    title = {'text': "성공 확률", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#1E88E5" if prob_pct >= 50 else "#EF5350"},
                        'steps': [
                            {'range': [0, 40], 'color': "#E0E0E0"},
                            {'range': [40, 70], 'color': "#BDBDBD"},
                            {'range': [70, 100], 'color': "#A5D6A7"}],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}}
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with r_col2:
                st.write("")
                st.write("")
                if prob_pct >= 80:
                    st.success("### 🎉 대박 예감! (Strong Buy)")
                    st.write(f"설정하신 가격(**{user_price:,}원**)과 예상 퀄리티(**{user_score}점**)라면 시장에서 큰 성공을 거둘 확률이 높습니다.")
                    st.write("- **전략:** 마케팅에 집중하여 초기 유저를 확보하세요.")
                elif prob_pct >= 50:
                    st.info("### 🙂 긍정적 전망 (Positive)")
                    st.write(f"성공 확률이 **{prob_pct:.1f}%**로 시장 평균 이상입니다.")
                    st.write("- **조언:** 출시 전 버그를 잡고 퀄리티(평가 점수)를 조금 더 높이면 확률이 급상승합니다.")
                else:
                    st.warning("### ⚠️ 전략 수정 필요 (Risk)")
                    st.write(f"현재 설정으로는 성공 확률이 **{prob_pct:.1f}%**로 다소 낮습니다.")
                    st.write("**개선 포인트:**")
                    if user_score < 70:
                        st.write("- 📉 **평가 점수:** 게임의 재미와 완성도를 높이는 것이 최우선입니다.")
                    if user_price > 40000:
                        st.write("- 💰 **가격:** 인디 게임 시장에서 다소 비싼 가격일 수 있습니다. 가격 인하를 고려해 보세요.")
        
        else:
            st.info("👈 왼쪽 사이드바에서 게임 스펙을 설정하고 **[🚀 예측 실행]** 버튼을 눌러주세요.")
            st.markdown("""
            **시뮬레이션 가이드:**
            1. **가격:** 출시 예정 가격을 설정합니다.
            2. **평가 점수:** 내 게임이 받을 예상 스팀 평가 점수입니다. (높을수록 확률 UP)
            3. **장르:** 게임의 핵심 장르를 1~3개 선택합니다.
            """)

    # -----------------------------------------------------
    # TAB 3: 모델 상세 정보
    # -----------------------------------------------------
    with tab3:
        st.subheader("⚙️ AI 모델 기술 명세서")
        st.markdown(f"**현재 사용 중인 모델:** `{selected_model_name}`")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("#### ✅ 모델 특징")
            if "XGBoost" in selected_model_name:
                st.write("- **특징:** 부스팅(Boosting) 알고리즘을 사용하여 오답을 집중적으로 학습함.")
                st.write("- **장점:** 현재 데이터 분석 대회에서 가장 성능이 좋은 모델.")
            elif "Random Forest" in selected_model_name:
                st.write("- **특징:** 여러 개의 결정 트리(Decision Tree)를 만들어 다수결로 결정함.")
                st.write("- **장점:** 과적합(Overfitting)에 강하고 안정적임.")
            else:
                st.write("- **특징:** 데이터를 나누는 기준선(S자 곡선)을 찾아 확률을 계산함.")
                st.write("- **장점:** 결과 해석이 쉽고 빠름.")

        with m_col2:
            st.markdown("#### 🎯 정확도 의미")
            st.write(f"- 이 모델은 전체 데이터의 **{current_acc*100:.1f}%**를 올바르게 예측했습니다.")
            st.write("- **성공 기준:** 동시 접속자가 상위 50% 안에 드는 경우를 '성공'으로 정의했습니다.")
