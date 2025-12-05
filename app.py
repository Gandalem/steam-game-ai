import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier

# ------------------------------------------------
# 1. 페이지 설정 및 디자인
# ------------------------------------------------
st.set_page_config(page_title="Steam Market Insight", layout="wide", page_icon="💰")

st.title("💰 스팀 게임: 장르별 황금 가격대 분석기")
st.markdown("""
이 프로그램은 스팀 데이터를 분석하여 **"어떤 장르를 얼마에 팔아야 대박이 나는가?"**를 시각화해 줍니다.
왼쪽 사이드바에서 내 게임의 성공 확률도 예측해 보세요!
""")

# ------------------------------------------------
# 2. 데이터 로드 및 전처리
# ------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('steam_top_sellers_ULTIMATE_v2.xlsx')
    except:
        st.error("데이터 파일이 없습니다.")
        return None, None, None, None

    # 가격 전처리
    def clean_price(price_raw):
        if pd.isna(price_raw): return 0
        price_str = str(price_raw)
        numbers_only = re.sub(r'[^0-9]', '', price_str)
        return int(numbers_only) if numbers_only else 0

    df['Price_Clean'] = df['최종 가격'].apply(clean_price)

    # 가격 구간(Category) 생성 - 분석용
    def get_price_category(price):
        if price == 0: return '0. 무료 (Free)'
        elif price < 10000: return '1. 1만원 미만'
        elif price < 30000: return '2. 1~3만원'
        elif price < 60000: return '3. 3~6만원'
        else: return '4. 6만원 이상'
    
    df['Price_Range'] = df['Price_Clean'].apply(get_price_category)

    # 태그 전처리
    df = df.dropna(subset=['주요 태그 (상위 5개)'])
    df['Tags_List'] = df['주요 태그 (상위 5개)'].astype(str).apply(lambda x: [tag.strip() for tag in x.split(',')])

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(df['Tags_List'])
    tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_, index=df.index)

    # 타겟 설정 (상위 20% 동접자 = 성공)
    threshold = df['현재 동시 접속자'].quantile(0.80)
    df['Success'] = df['현재 동시 접속자'].apply(lambda x: 1 if x >= threshold else 0)

    X = pd.concat([df[['Price_Clean']], tags_df], axis=1)
    y = df['Success']
    
    return df, X, y, mlb

df, X, y, mlb = load_data()

# ------------------------------------------------
# 3. 모델 학습 (예측 기능용 - 백그라운드 실행)
# ------------------------------------------------
if df is not None:
    # 사용자 예측을 위해 모델은 뒤에서 조용히 학습시킵니다.
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X, y)

    # ------------------------------------------------
    # 4. (사이드바) 내 게임 예측하기
    # ------------------------------------------------
    st.sidebar.header("🛠️ 내 게임 성공 예측")
    st.sidebar.info("개발 중인 게임의 스펙을 입력하세요.")
    
    user_price = st.sidebar.number_input("출시 예정 가격 (원)", min_value=0, value=25000, step=1000)
    
    top_tags = pd.Series([tag for tags in df['Tags_List'] for tag in tags]).value_counts().head(20).index.tolist()
    user_tags = st.sidebar.multiselect("장르 선택 (최대 3개)", top_tags, default=top_tags[:2])

    if st.sidebar.button("🚀 성공 확률 예측"):
        with st.spinner('시장 데이터 분석 중...'):
            time.sleep(1)
            # 입력 데이터 변환
            input_data = pd.DataFrame(0, index=[0], columns=X.columns)
            input_data['Price_Clean'] = user_price
            for tag in user_tags:
                if tag in input_data.columns:
                    input_data[tag] = 1
            
            # 예측
            pred_prob = model.predict_proba(input_data)[0][1]
        
        st.sidebar.divider()
        if pred_prob >= 0.5:
            st.sidebar.success(f"예측 결과: 대박 가능성 높음! ({pred_prob*100:.1f}%)")
            st.sidebar.balloons()
        else:
            st.sidebar.warning(f"예측 결과: 시장 진입 주의 ({pred_prob*100:.1f}%)")
            st.sidebar.caption("가격이나 장르를 변경해 보세요.")

    # ------------------------------------------------
    # 5. (메인) 장르별 가격대 분석 (인사이트 시각화)
    # ------------------------------------------------
    st.subheader("📊 장르(Tag) x 가격대별 성공률 분석 히트맵")
    st.markdown("색이 **진할수록(붉을수록)** 해당 가격대에서 성공 확률이 높다는 뜻입니다.")

    # 데이터 가공 (태그별로 쪼개기)
    df_exploded = df.explode('Tags_List')
    
    # 상위 15개 태그만 추출 (너무 많으면 그래프가 지저분함)
    top_15_tags = df_exploded['Tags_List'].value_counts().head(15).index
    df_filtered = df_exploded[df_exploded['Tags_List'].isin(top_15_tags)]

    # 피벗 테이블 생성 (인덱스:태그, 컬럼:가격대, 값:성공률)
    pivot_table = df_filtered.pivot_table(index='Tags_List', columns='Price_Range', values='Success', aggfunc='mean')
    
    # 히트맵 그리기
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".0%", cmap="YlOrRd", linewidths=.5, ax=ax)
    plt.title("Top 15 장르별 가격대 성공률 (Success Rate)", fontsize=15)
    plt.xlabel("가격 구간", fontsize=12)
    plt.ylabel("장르 (Tag)", fontsize=12)
    st.pyplot(fig)

    st.divider()

    # ------------------------------------------------
    # 6. 개별 태그 심층 분석 (Drill Down)
    # ------------------------------------------------
    st.subheader("🔍 특정 장르 상세 분석")
    selected_tag = st.selectbox("분석하고 싶은 장르를 선택하세요:", top_tags)

    # 선택한 태그 데이터만 필터링
    tag_data = df_exploded[df_exploded['Tags_List'] == selected_tag]
    
    # 가격대별 성공률 계산
    analysis = tag_data.groupby('Price_Range')['Success'].mean().reset_index()
    analysis['Success'] = analysis['Success'] * 100 # 백분율로 변환

    # 막대 그래프 그리기
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write(f"**[{selected_tag}]** 장르 요약")
        best_price = analysis.loc[analysis['Success'].idxmax()]
        st.success(f"🏆 추천 가격대: **{best_price['Price_Range']}**")
        st.metric("최고 성공률", f"{best_price['Success']:.1f}%")
        st.caption(f"총 {len(tag_data)}개의 게임 데이터 분석됨")

    with col2:
        st.bar_chart(analysis.set_index('Price_Range'))
