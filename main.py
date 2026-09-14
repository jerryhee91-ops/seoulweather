import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.markdown("1908년 이후의 서울 기온 데이터를 바탕으로 선형 회귀 모델을 학습하고, 지정한 연도의 기온을 예측합니다.")

# Data Loading & Processing
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_preprocess_data():
    # 데이터 로드 (UTF-8 인코딩)
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # 날짜 데이터 처리 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_summary = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 조건 적용: 2025년 이하 & 관측일수 300일 이상
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & 
        (yearly_summary["관측일수"] >= 300)
    ].copy()
    
    # 회귀분석용 독립변수 (1908년부터 지난 연수)
    filtered_df["지난연수"] = filtered_df["연도"] - 1908
    
    return filtered_df

try:
    df_clean = load_and_preprocess_data()

    # 데이터 요약 정보 추출
    num_years = len(df_clean)
    start_year = int(df_clean["연도"].min())
    end_year = int(df_clean["연도"].max())

    # 회귀 모델 학습 (X: 지난연수, y: 평균기온)
    X = df_clean[["지난연수"]].values
    y = df_clean["평균기온"].values

    model = LinearRegression()
    model.fit(X, y)

    # 상관계수 및 회귀선 값 계산
    corr = np.corrcoef(df_clean["연도"], df_clean["평균기온"])[0, 1]
    df_clean["회귀선"] = model.predict(X)

    # 사이드바 / 예측 인터페이스
    st.sidebar.header("🔮 기온 예측")
    target_year = st.sidebar.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2030, step=1)

    # 선택된 연도 기온 예측
    predicted_temp = model.predict([[target_year - 1908]])[0]

    # 메인 화면 지표 표시
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📌 예측 결과")
        st.metric(
            label=f"**{target_year}년 예상 평균기온**", 
            value=f"{predicted_temp:.2f} °C"
        )
        
        st.markdown("---")
        st.subheader("📊 학습 데이터 정보")
        st.write(f"- **분석에 사용된 해의 개수:** `{num_years}`개")
        st.write(f"- **시작 연도:** `{start_year}`년")
        st.write(f"- **끝 연도:** `{end_year}`년")
        st.write(f"- **연도-기온 상관계수:** `{corr:.4f}`")

    with col2:
        st.subheader("📈 서울 연평균 기온 추이 및 회귀선")
        
        # Plotly 그래프 생성
        fig = go.Figure()

        # 관측 데이터 산점도
        fig.add_trace(
            go.Scatter(
                x=df_clean["연도"], 
                y=df_clean["평균기온"], 
                mode="markers", 
                name="연평균기온 (관측값)",
                marker=dict(color="#1f77b4", size=7, opacity=0.8)
            )
        )

        # 추세 직선 (1900년 ~ 2100년 확장 표시)
        line_years = np.arange(1900, 2101)
        line_elapsed = line_years - 1908
        line_pred = model.predict(line_elapsed.reshape(-1, 1))

        fig.add_trace(
            go.Scatter(
                x=line_years, 
                y=line_pred, 
                mode="lines", 
                name="회귀 직선",
                line=dict(color="#ff7f0e", width=2, dash="dash")
            )
        )

        # 사용자가 선택한 예측 포인트 표시
        fig.add_trace(
            go.Scatter(
                x=[target_year],
                y=[predicted_temp],
                mode="markers",
                name=f"선택 연도({target_year}년)",
                marker=dict(color="#d62728", size=13, symbol="star")
            )
        )

        # 그래프 레이아웃 설정 (가로축 연도 설정)
        fig.update_layout(
            xaxis_title="연도",
            yaxis_title="평균기온 (°C)",
            xaxis=dict(tickformat="d", range=[1895, 2105]),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")
