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

# ------------------------------------------------
# 1. 페이지 설정 및 스타일링
# ------------------------------------------------
st.set_page_config(page_title="Steam Market Compass", layout="wide", page_icon="🧭")

st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🧭 스팀 게임 시장 나침반 (Market Compass)")
st.markdown("빅데이터 분석을 통해 **게임의 적정 가격**과 **성공 확률**을 시각적으로 탐색합니다.")

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

    # 가격 전처리
    def clean_price(price_raw):
        if pd.isna(price_raw): return 0
        price_str = str(price_raw)
        numbers_only = re.sub(r'[^0-9]', '', price_str)
        return int(numbers_only) if numbers_only else 0

    df['Price_Clean'] = df['최종 가격'].apply(clean_price)

    # 가격 구간(Category) 생성
    def get_price_category(price):
        if price == 0: return '무료 (Free)'
        elif price < 15000: return '저가 (~1.5만원)'
        elif price < 35000: return '중가 (1.5~3.5만원)'
        elif price < 60000: return '준고가 (3.5~6만원)'
        else: return '고가 (6만원 이상)'
    
    price_order = ['무료 (Free)', '저가 (~1.5만원)', '중가 (1.5~3.5만원)', '준고가 (3.5~6만원)', '고가 (6만원 이상)']
    df['Price_Range'] = pd.Categorical(df['Price_Clean'].apply(get_price_category), categories=price_order, ordered=True)

    # -----------------------------------------------------------
    # [수정됨] 태그 전처리 및 불필요한 태그 제거 로직
    # -----------------------------------------------------------
    df = df.dropna(subset=['주요 태그 (상위 5개)'])
    
    # 1. 먼저 리스트로 변환
    df['Tags_List'] = df['주요 태그 (상위 5개)'].astype(str).apply(lambda x: [tag.strip() for tag in x.split(',')])

    # 2. 제외할 태그 목록 정의 (사용자 요청 반영)
    banned_tags = ['무료 플레이', '앞서 해보기', '애니메이션 모델', '디자인과 일러스트레이션']

    # 3. 제외 태그 필터링 함수
    def filter_tags(tags):
        return [tag for tag in tags if tag not in banned_tags]

    df['Tags_List'] = df['Tags_List'].apply(filter_tags)

    # 4. 태그가 다 지워져서 빈 리스트가 된 행은 삭제 (데이터 품질 유지)
    df = df[df['Tags_List'].map(len) > 0]

    # -----------------------------------------------------------

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(df['Tags_List'])
    tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_, index=df.index)

    # 타겟 설정 (상위 20% 동접자 = 성공)
    threshold = df['현재 동시 접속자'].quantile(0.80)
    df['Success'] = df['현재 동시 접속자'].apply(lambda x: 1 if x >= threshold else 0)

    X = pd.concat([df[['Price_Clean']], tags_df], axis=1)
    y = df['Success']
    
    return df, X, y, mlb, threshold

df, X, y, mlb, threshold = load_data()

# ------------------------------------------------
# 3. 모델 학습 (백그라운드)
# ------------------------------------------------
if df is not None:
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X, y)

    # ------------------------------------------------
    # 4. KPI 대시보드
    # ------------------------------------------------
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎮 분석된 게임 수", f"{len(df):,}개")
    with col2:
        avg_success = df['Success'].mean() * 100
        st.metric("🏆 시장 평균 성공률", f"{avg_success:.1f}%")
    with col3:
        # 가장 성공률 높은 가격대 (데이터가 충분한 경우만)
        if not df.empty:
            best_price_range = df.groupby('Price_Range')['Success'].mean().idxmax()
            st.metric("💎 황금 가격대", best_price_range)
    with col4:
        st.metric("🔥 대박 기준 (동접자)", f"{int(threshold):,}명 ↑")
    
    st.divider()

    # ------------------------------------------------
    # 5. 인터랙티브 히트맵
    # ------------------------------------------------
    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.subheader("🗺️ 장르 x 가격대 성공 지도")
        st.caption("불필요한 태그(무료 플레이 등)는 제외되었습니다.")
        
        # 데이터 가공
        df_exploded = df.explode('Tags_List')
        
        # 상위 15개 태그 추출
        top_15_tags = df_exploded['Tags_List'].value_counts().head(15).index
        df_filtered = df_exploded[df_exploded['Tags_List'].isin(top_15_tags)]
        
        pivot_table = df_filtered.pivot_table(index='Tags_List', columns='Price_Range', values='Success', aggfunc='mean')
        
        # Plotly 히트맵
        fig_heatmap = px.imshow(
            pivot_table,
            labels=dict(x="가격대", y="장르", color="성공률"),
            x=pivot_table.columns,
            y=pivot_table.index,
            text_auto=".0%",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig_heatmap.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col_side:
        st.subheader("🔍 장르별 상세 탐색")
        
        top_tags = top_15_tags.tolist()
        # 데이터가 있을 때만 위젯 표시
        if top_tags:
            selected_tag = st.selectbox("분석할 장르를 선택하세요", top_tags, index=0)
            
            tag_data = df_exploded[df_exploded['Tags_List'] == selected_tag]
            tag_analysis = tag_data.groupby('Price_Range')['Success'].mean().reset_index()
            tag_analysis['Success'] = tag_analysis['Success'] * 100
            
            fig_bar = px.bar(
                tag_analysis, 
                x='Price_Range', 
                y='Success', 
                color='Success',
                color_continuous_scale='Greens',
                title=f"[{selected_tag}] 가격대별 성공률",
                text_auto='.1f'
            )
            fig_bar.update_layout(
                xaxis_title=None, 
                yaxis_title="성공률 (%)", 
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ------------------------------------------------
    # 6. 사이드바 예측 시뮬레이터
    # ------------------------------------------------
    st.sidebar.header("🕹️ 내 게임 시뮬레이션")
    
    st.sidebar.write("💰 출시 가격 설정")
    user_price = st.sidebar.slider("", 0, 100000, 25000, step=1000, format="₩%d")
    
    all_top_tags = pd.Series([tag for tags in df['Tags_List'] for tag in tags]).value_counts().head(20).index.tolist()
    
    st.sidebar.write("🏷️ 장르 선택 (최대 3개)")
    # 태그 선택 시 기본값이 리스트에 없으면 에러나므로 안전장치 마련
    default_tags = all_top_tags[:2] if len(all_top_tags) >= 2 else all_top_tags
    user_tags = st.sidebar.multiselect("", all_top_tags, default=default_tags, label_visibility="collapsed")

    if st.sidebar.button("🚀 예측 실행 (Click)", type="primary"):
        with st.spinner('AI 엔진 가동 중...'):
            time.sleep(0.8)
            
            input_data = pd.DataFrame(0, index=[0], columns=X.columns)
            input_data['Price_Clean'] = user_price
            for tag in user_tags:
                if tag in input_data.columns:
                    input_data[tag] = 1
            
            prob = model.predict_proba(input_data)[0][1]
        
        st.sidebar.divider()
        st.sidebar.subheader("분석 결과")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "성공 확률"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "lightgreen"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80}}))
        
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.sidebar.plotly_chart(fig_gauge, use_container_width=True)

        if prob >= 0.5:
            st.sidebar.success("🎉 시장 진입 추천!")
        else:
            st.sidebar.warning("⚠️ 가격/장르 재검토 필요")
