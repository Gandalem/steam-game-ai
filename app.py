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
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------------------------
# 1. 페이지 설정 및 커스텀 CSS
# ------------------------------------------------
st.set_page_config(page_title="Steam Game Success Predictor", layout="wide", page_icon="🎮")

st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    /* 카드 스타일 적용 */
    .css-card {
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    /* KPI 카드 스타일 */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
    }
    /* 헤더 스타일 */
    h1 { color: #1E88E5; }
    h2 { color: #333; font-size: 1.5rem; margin-bottom: 1rem; }
    h3 { color: #555; font-size: 1.2rem; margin-top: 0; }
    /* 버튼 스타일 */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    /* 성공/실패 박스 스타일 */
    .success-box { padding: 15px; border-radius: 8px; background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .warning-box { padding: 15px; border-radius: 8px; background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .info-box { padding: 15px; border-radius: 8px; background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
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

    def clean_price(price_raw):
        if pd.isna(price_raw): return 0
        price_str = str(price_raw)
        numbers_only = re.sub(r'[^0-9]', '', price_str)
        return int(numbers_only) if numbers_only else 0
    df['Price_Clean'] = df['최종 가격'].apply(clean_price)

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

    def get_price_category(price):
        if price == 0: return '무료 (Free)'
        elif price < 15000: return '저가 (~1.5만원)'
        elif price < 35000: return '중가 (1.5~3.5만원)'
        elif price < 60000: return '준고가 (3.5~6만원)'
        else: return '고가 (6만원 이상)'
    price_order = ['무료 (Free)', '저가 (~1.5만원)', '중가 (1.5~3.5만원)', '준고가 (3.5~6만원)', '고가 (6만원 이상)']
    df['Price_Range'] = pd.Categorical(df['Price_Clean'].apply(get_price_category), categories=price_order, ordered=True)

    df = df.dropna(subset=['주요 태그 (상위 5개)'])
    df['Tags_List'] = df['주요 태그 (상위 5개)'].astype(str).apply(lambda x: [tag.strip() for tag in x.split(',')])
    banned_tags = ['무료 플레이', '앞서 해보기', '애니메이션 모델', '애니메이션과 모델링', '애니메이션 및 모델링', '디자인과 일러스트레이션', '사진 편집', '동영상 제작', '동영상제작', '유틸리티', '소프트웨어', '웹 퍼블리싱', '오디오 제작', '게임 개발', '소프트웨어 교육']
    df['Tags_List'] = df['Tags_List'].apply(lambda tags: [tag for tag in tags if tag not in banned_tags])
    df = df[df['Tags_List'].map(len) > 0]

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(df['Tags_List'])
    tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_, index=df.index)

    threshold = df['현재 동시 접속자'].quantile(0.50) 
    df['Success'] = df['현재 동시 접속자'].apply(lambda x: 1 if x >= threshold else 0)
    X = pd.concat([df[['Price_Clean', 'Review_Score']], tags_df], axis=1)
    y = df['Success']
    return df, X, y, mlb, threshold

df, X, y, mlb, threshold = load_data()

# ------------------------------------------------
# 3. 모델 학습 (F1-Score 추가)
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
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        trained_models[name] = model
    return trained_models, scores

if df is not None:
    models_dict, scores_dict = train_all_models(X, y)

    # =========================================================
    # 메인 레이아웃 시작
    # =========================================================
    st.title("🎮 스팀 게임 성공 예측 (Steam Success Predictor)")
    st.markdown("##### 빅데이터 기반 AI가 당신의 게임 아이디어를 분석합니다.")
    st.divider()

    # KPI 카드 섹션
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    best_model_name = max(scores_dict, key=lambda k: scores_dict[k]['accuracy'])
    with kpi1: st.metric("최고 모델 정확도", f"{scores_dict[best_model_name]['accuracy']*100:.1f}%", best_model_name)
    with kpi2: st.metric("분석된 게임 수", f"{len(df):,}개", "순수 게임 기준")
    with kpi3: st.metric("평균 성공률", f"{df['Success'].mean()*100:.1f}%", "상위 50% 기준")
    with kpi4: st.metric("성공 기준 (동접자)", f"{int(threshold):,}명 이상")
    st.divider()

    # 3단 컬럼 레이아웃
    col_left, col_center, col_right = st.columns([1, 1.2, 1], gap="medium")

    # -----------------------------------------------------
    # [왼쪽 컬럼] 입력 및 제어
    # -----------------------------------------------------
    with col_left:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.header("📝 게임 스펙 입력")
        with st.form("prediction_form"):
            user_price = st.number_input("💰 출시 가격 (KRW)", 0, 100000, 32000, step=1000)
            user_score = st.slider("⭐ 예상 평가 점수", 0, 100, 85)
            all_tags = mlb.classes_.tolist()
            user_tags = st.multiselect("🏷️ 장르/태그 (최대 5개)", all_tags, default=all_tags[:2] if len(all_tags)>2 else all_tags)
            submitted = st.form_submit_button("예측하기 🚀")

        st.divider()
        st.header("⚙️ AI 모델 선택")
        selected_model_name = st.radio("사용할 알고리즘", list(models_dict.keys()))
        current_model = models_dict[selected_model_name]
        current_scores = scores_dict[selected_model_name]

        st.markdown(f"""
        <div class="info-box">
            <strong>모델 성능 정보:</strong><br>
            • 정확도 (Accuracy): <strong>{current_scores['accuracy']*100:.1f}%</strong><br>
            • F1-Score: <strong>{current_scores['f1']:.3f}</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # [중앙 & 오른쪽 컬럼] 예측 결과 처리
    # -----------------------------------------------------
    if submitted:
        with st.spinner('AI가 데이터를 분석 중입니다...'):
            time.sleep(1)
            input_data = pd.DataFrame(0, index=[0], columns=X.columns)
            input_data['Price_Clean'] = user_price
            input_data['Review_Score'] = user_score
            for tag in user_tags:
                if tag in input_data.columns: input_data[tag] = 1
            prob = current_model.predict_proba(input_data)[0][1]
            prob_pct = prob * 100

        # [중앙 컬럼] 결과 리포트
        with col_center:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.header("📊 분석 결과 리포트")
            st.markdown(f"<h1 style='text-align: center; font-size: 4rem; color: #1E88E5;'>{prob_pct:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>예상 성공 확률</h3>", unsafe_allow_html=True)
            st.write("")
            
            if prob_pct >= 80:
                st.markdown("""
                <div class="success-box">
                    <h4>🎉 대박 예감! (Strong Buy)</h4>
                    설정하신 스펙은 시장에서 큰 성공을 거둘 가능성이 매우 높습니다. 적극적인 마케팅을 준비하세요!
                </div>
                """, unsafe_allow_html=True)
            elif prob_pct >= 50:
                st.markdown("""
                <div class="info-box">
                    <h4>🙂 긍정적 전망 (Positive)</h4>
                    평균 이상의 성과가 기대됩니다. 출시 전 게임의 완성도를 조금 더 높인다면 대박도 노려볼 수 있습니다.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <h4>⚠️ 전략 수정 필요 (Risk)</h4>
                    현재 스펙으로는 성공 확률이 낮습니다. 가격을 낮추거나, 게임의 재미(평가 점수)를 높이는 전략 수정이 필요합니다.
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            st.markdown("#### 💡 AI의 조언")
            if user_score < 75: st.write("- **평가 점수:** 게임의 퀄리티를 높여 긍정적인 초기 평가를 받는 것이 중요합니다.")
            if user_price > 45000: st.write("- **가격:** 인디 게임치고는 가격이 다소 높습니다. 진입 장벽을 낮추는 것을 고려해 보세요.")
            if not user_tags: st.write("- **태그:** 게임의 특징을 잘 나타내는 태그를 추가하면 타겟 유저에게 노출될 확률이 높아집니다.")
            st.markdown('</div>', unsafe_allow_html=True)

        # [오른쪽 컬럼] 시각화
        with col_right:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.header("📈 시각화 분석")
            
            # 게이지 차트
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = prob_pct,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1E88E5" if prob_pct >= 50 else "#EF5350"},
                    'steps': [{'range': [0, 100], 'color': "#e9ecef"}]}
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.divider()
            st.subheader("🤖 모델 성능 비교")
            # 모델 비교 막대 차트
            acc_df = pd.DataFrame({
                'Model': scores_dict.keys(),
                'Accuracy': [score['accuracy'] for score in scores_dict.values()]
            }).sort_values(by='Accuracy', ascending=True)
            
            fig_bar = px.bar(acc_df, x='Accuracy', y='Model', orientation='h', text_auto='.1%',
                             color='Accuracy', color_continuous_scale='Blues')
            fig_bar.update_layout(xaxis_title="정확도", yaxis_title=None, showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
            fig_bar.update_xaxes(range=[0.5, 1.0])
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    else: # 아직 예측 버튼을 안 눌렀을 때
        with col_center:
            st.info("👈 왼쪽에서 게임 정보를 입력하고 **'예측하기 🚀'** 버튼을 눌러주세요.")
        with col_right:
             # 장르별 성공률 히트맵 (기본 표시)
            df_exploded = df.explode('Tags_List')
            top_tags = df_exploded['Tags_List'].value_counts().head(10).index
            df_filtered = df_exploded[df_exploded['Tags_List'].isin(top_tags)]
            pivot = df_filtered.pivot_table(index='Tags_List', columns='Price_Range', values='Success', aggfunc='mean')
            fig_heatmap = px.imshow(pivot, labels=dict(x="가격", y="장르", color="성공률"), color_continuous_scale="RdBu_r", aspect="auto")
            fig_heatmap.update_layout(height=400, title="장르 x 가격대 성공 히트맵", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_heatmap, use_container_width=True)
