import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix

# ------------------------------------------------
# 1. 페이지 설정 및 디자인
# ------------------------------------------------
st.set_page_config(page_title="Steam Success Predictor", layout="wide")

st.title("🎮 스팀 게임 흥행 예측기 (AI)")
st.markdown("""
이 프로그램은 **머신러닝(XGBoost)**을 활용해 게임 스펙만 보고 대박 여부를 예측합니다.
교수님 제출용: **ROC Curve & 이중 검증(Cross Validation)** 결과가 포함되어 있습니다.
""")

# ------------------------------------------------
# 2. 데이터 로드 및 전처리 (캐싱으로 속도 최적화)
# ------------------------------------------------
@st.cache_data
def load_and_process_data():
    # 데이터 로드 (같은 폴더에 엑셀 파일이 있어야 함)
    try:
        df = pd.read_excel('steam_top_sellers_ULTIMATE_v2.xlsx')
    except:
        # 엑셀 없으면 예시 데이터 생성 (에러 방지용)
        st.error("데이터 파일을 찾을 수 없습니다. 같은 폴더에 엑셀 파일을 넣어주세요.")
        return None, None, None, None

    # 가격 전처리
    def clean_price(price_raw):
        if pd.isna(price_raw): return 0
        price_str = str(price_raw)
        numbers_only = re.sub(r'[^0-9]', '', price_str)
        return int(numbers_only) if numbers_only else 0

    df['Price_Clean'] = df['최종 가격'].apply(clean_price)

    # 태그 전처리
    df = df.dropna(subset=['주요 태그 (상위 5개)'])
    df['Tags_List'] = df['주요 태그 (상위 5개)'].astype(str).apply(lambda x: [tag.strip() for tag in x.split(',')])

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(df['Tags_List'])
    tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_, index=df.index)

    # 타겟 설정 (상위 20% 동접자 기준)
    threshold = df['현재 동시 접속자'].quantile(0.80)
    df['Success'] = df['현재 동시 접속자'].apply(lambda x: 1 if x >= threshold else 0)

    # 학습 데이터 준비
    X = pd.concat([df[['Price_Clean']], tags_df], axis=1)
    y = df['Success']
    
    return X, y, mlb, threshold

X, y, mlb, threshold = load_and_process_data()

if X is not None:
    # ------------------------------------------------
    # 3. 모델 학습 (자동 수행)
    # ------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # 성능 지표 계산
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    # ------------------------------------------------
    # 4. (왼쪽) 사이드바: 사용자 입력
    # ------------------------------------------------
    st.sidebar.header("🛠️ 게임 스펙 설정")
    
    user_price = st.sidebar.number_input("게임 가격 (원)", min_value=0, value=30000, step=1000)
    
    # 상위 20개 인기 태그만 추출해서 선택지로 제공
    top_tags = pd.Series([tag for tags in mlb.inverse_transform(X.iloc[:, 1:].values) for tag in tags]).value_counts().head(20).index.tolist()
    user_tags = st.sidebar.multiselect("게임 장르/태그 선택", top_tags, default=['Action', 'Indie'])

    if st.sidebar.button("🚀 흥행 예측하기"):
        # 입력 데이터 변환
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        input_data['Price_Clean'] = user_price
        for tag in user_tags:
            if tag in input_data.columns:
                input_data[tag] = 1
        
        # 예측
        pred_prob = model.predict_proba(input_data)[0][1] # 성공 확률
        pred_result = "대박 (Hit)" if pred_prob >= 0.5 else "일반 (Normal)"
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("결과 분석")
        if pred_prob >= 0.5:
            st.sidebar.success(f"예측 결과: {pred_result}")
        else:
            st.sidebar.error(f"예측 결과: {pred_result}")
        st.sidebar.write(f"성공 확률: **{pred_prob*100:.1f}%**")

    # ------------------------------------------------
    # 5. (메인) 모델 성능 리포트 (교수님용)
    # ------------------------------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 모델 성능 지표")
        st.metric(label="정확도 (Accuracy)", value=f"{acc*100:.1f}%")
        st.metric(label="이중 검증(CV) 평균 점수", value=f"{cv_scores.mean()*100:.1f}%")
        st.info(f"성공 기준: 동시 접속자 {int(threshold)}명 이상")

    with col2:
        st.subheader("📈 ROC Curve Analysis")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔍 변수 중요도 (Feature Importance)")
    st.write("어떤 요소가 게임 성공에 가장 큰 영향을 미치는가?")
    
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

    st.bar_chart(importances)
