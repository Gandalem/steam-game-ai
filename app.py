import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time # 로딩 시간을 벌기 위해 추가
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix

# ------------------------------------------------
# 1. 페이지 설정 및 디자인
# ------------------------------------------------
st.set_page_config(page_title="Steam Success AI", layout="wide", page_icon="🎮")

st.title("🎮 스팀 게임 흥행 예측기 (AI Ver 2.0)")
st.markdown("""
이 프로그램은 **3가지 머신러닝 모델**을 비교 분석하여 게임의 성공 가능성을 예측합니다.
왼쪽 사이드바에서 게임 스펙을 설정하고 예측 버튼을 눌러보세요!
""")

# ------------------------------------------------
# 2. 데이터 로드 및 전처리
# ------------------------------------------------
@st.cache_data
def load_data():
    # 엑셀 파일 로드
    try:
        df = pd.read_excel('steam_top_sellers_ULTIMATE_v2.xlsx')
    except:
        st.error("데이터 파일(steam_top_sellers_ULTIMATE_v2.xlsx)을 찾을 수 없습니다.")
        return None, None, None, None

    # 가격 전처리 (문자열 -> 숫자)
    def clean_price(price_raw):
        if pd.isna(price_raw): return 0
        price_str = str(price_raw)
        numbers_only = re.sub(r'[^0-9]', '', price_str)
        return int(numbers_only) if numbers_only else 0

    df['Price_Clean'] = df['최종 가격'].apply(clean_price)

    # 태그 전처리 (원-핫 인코딩)
    df = df.dropna(subset=['주요 태그 (상위 5개)'])
    df['Tags_List'] = df['주요 태그 (상위 5개)'].astype(str).apply(lambda x: [tag.strip() for tag in x.split(',')])

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(df['Tags_List'])
    tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_, index=df.index)

    # 타겟 설정 (상위 20% 동접자 기준 성공=1, 실패=0)
    threshold = df['현재 동시 접속자'].quantile(0.80)
    df['Success'] = df['현재 동시 접속자'].apply(lambda x: 1 if x >= threshold else 0)

    # 학습용 데이터 합치기
    X = pd.concat([df[['Price_Clean']], tags_df], axis=1)
    y = df['Success']
    
    return X, y, mlb, threshold

X, y, mlb, threshold = load_data()

# ------------------------------------------------
# 3. 모델 학습 (3가지 모델 비교)
# ------------------------------------------------
if X is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 모델 정의
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }

    # 모델 학습 및 결과 저장 (캐싱하여 속도 향상)
    @st.cache_resource
    def train_models(_models, _X_train, _y_train, _X_test, _y_test):
        results = {}
        for name, model in _models.items():
            model.fit(_X_train, _y_train)
            y_pred = model.predict(_X_test)
            y_prob = model.predict_proba(_X_test)[:, 1]
            acc = accuracy_score(_y_test, y_pred)
            
            fpr, tpr, _ = roc_curve(_y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            results[name] = {
                'model': model,
                'accuracy': acc,
                'auc': roc_auc,
                'fpr': fpr,
                'tpr': tpr
            }
        return results

    model_results = train_models(models, X_train, y_train, X_test, y_test)
    
    # 가장 성능 좋은 모델 선정 (Prediction용)
    best_model_name = max(model_results, key=lambda k: model_results[k]['auc'])
    best_model = model_results[best_model_name]['model']

    # ------------------------------------------------
    # 4. (사이드바) 사용자 입력 & 예측
    # ------------------------------------------------
    st.sidebar.header("🛠️ 게임 스펙 설정")
    
    user_price = st.sidebar.number_input("게임 가격 (KRW)", min_value=0, value=30000, step=1000)
    
    # [수정 완료] 태그 선택 시 에러 방지를 위해 상위 2개를 기본값으로 자동 설정
    top_tags = pd.Series([tag for tags in mlb.inverse_transform(X.iloc[:, 1:].values) for tag in tags]).value_counts().head(20).index.tolist()
    user_tags = st.sidebar.multiselect("주요 태그 선택", top_tags, default=top_tags[:2])

    st.sidebar.markdown("---")
    
    # [기능 추가] 예측 버튼 클릭 시 로딩 애니메이션
    if st.sidebar.button("🚀 흥행 예측 시작"):
        # 1. 로딩바 보여주기 (Spinner)
        with st.spinner('AI가 스팀 데이터베이스를 분석 중입니다...'):
            time.sleep(1.5) # 사용자가 로딩을 느낄 수 있도록 1.5초 대기
            
            # 2. 입력 데이터 변환
            input_data = pd.DataFrame(0, index=[0], columns=X.columns)
            input_data['Price_Clean'] = user_price
            for tag in user_tags:
                if tag in input_data.columns:
                    input_data[tag] = 1
            
            # 3. 예측 수행 (최고 성능 모델 사용)
            pred_prob = best_model.predict_proba(input_data)[0][1]
            
        # 4. 결과 출력 (로딩 끝난 후)
        st.sidebar.subheader("🎯 분석 결과")
        if pred_prob >= 0.5:
            st.sidebar.success("예측: 대박 (Hit!)")
            st.sidebar.balloons() # 대박이면 풍선 효과!
        else:
            st.sidebar.error("예측: 일반 (Normal)")
            
        st.sidebar.write(f"성공 확률: **{pred_prob*100:.1f}%**")
        st.sidebar.caption(f"Used Model: {best_model_name}")

    # ------------------------------------------------
    # 5. (메인) 모델 비교 분석 (교수님용)
    # ------------------------------------------------
    st.subheader("📊 AI 모델 성능 비교 리포트")
    
    tab1, tab2 = st.tabs(["🏆 정확도 비교", "📈 ROC 커브 분석"])

    with tab1:
        st.write("3가지 알고리즘 중 어떤 모델이 가장 똑똑할까요?")
        # 정확도 비교 차트
        acc_df = pd.DataFrame({
            'Model': model_results.keys(),
            'Accuracy': [res['accuracy'] for res in model_results.values()]
        })
        st.bar_chart(acc_df.set_index('Model'), color="#4CAF50")
        
        # 1등 모델 강조
        st.success(f"가장 성능이 우수한 모델은 **[{best_model_name}]** 입니다. (정확도: {model_results[best_model_name]['accuracy']*100:.1f}%)")

    with tab2:
        st.write("모델의 변별력(AUC)을 나타내는 ROC 곡선입니다.")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for name, res in model_results.items():
            ax.plot(res['fpr'], res['tpr'], lw=2, label=f'{name} (AUC = {res["auc"]:.2f})')
            
        ax.plot([0, 1], [0, 1], 'k--', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend(loc="lower right")
        
        st.pyplot(fig)

    st.markdown("---")
    st.info(f"💡 **참고:** 성공 기준은 동시 접속자 상위 20% ({int(threshold)}명) 이상인 게임입니다.")
