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
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    yearly_summary = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        평균기온=("평균기온", "mean")
    ).reset_index()
    
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & 
        (yearly_summary["관측일수"] >= 300)
    ].copy()
    
    filtered_df["지난연수"] = filtered_df["연도"] - 1908
    
    return filtered_df

try:
    df_clean = load_and_preprocess_data()

    num_years = len(df_clean)
    start_year = int(df_clean["연도"].min())
    end_year = int(df_clean["연도"].max())

    # 1. 전체 기간 회귀 모델 학습
    X_full = df_clean[["지난연수"]].values
    y_full = df_clean["평균기온"].values

    model_full = LinearRegression()
    model_full.fit(X_full, y_full)

    # 전체 기간 기울기 (100년당 상승온도)
    slope_100y_full = model_full.coef_[0] * 100

    # 2. 최근 20년 회귀 모델 학습
    recent_20_start_year = end_year - 19
    df_recent = df_clean[df_clean["연도"] >= recent_20_start_year].copy()
    
    X_recent = df_recent[["지난연수"]].values
    y_recent = df_recent["평균기온"].values

    model_recent = LinearRegression()
    model_recent.fit(X_recent, y_recent)

    # 최근 20년 기울기 (100년당 상승온도)
    slope_100y_recent = model_recent.coef_[0] * 100

    # 기타 계산값
    corr = np.corrcoef(df_clean["연도"], df_clean["평균기온"])[0, 1]

    # 사이드바 설정
    st.sidebar.header("🔮 기온 예측")
    target_year = st.sidebar.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2030, step=1)

    predicted_temp = model_full.predict([[target_year - 1908]])[0]

    # 🔥 100년당 기온 상승 폭 비교 화면 (상단 배치)
    st.markdown("### 🔥 기온 상승 속도 비교 (100년당 상승 온도)")
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.metric(
            label=f"🌐 **전체 기간 ({start_year}~{end_year}년)**",
            value=f"+{slope_100y_full:.2f} °C / 100년",
            help="전체 데이터를 기준으로 계산한 100년당 평균 기온 상승 폭입니다."
        )
    with m_col2:
        diff_slope = slope_100y_recent - slope_100y_full
        st.metric(
            label=f"🚀 **최근 20년 ({recent_20_start_year}~{end_year}년)**",
            value=f"+{slope_100y_recent:.2f} °C / 100년",
            delta=f"전체 대비 {diff_slope:+.2f} °C/100년",
            delta_color="normal",
            help="최근 20년 데이터를 기준으로 계산한 100년당 평균 기온 상승 폭입니다."
        )

    st.markdown("---")

    # 메인 컨텐츠 (예측 결과 & 그래프)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📌 선택 연도 예측 결과")
        st.metric(
            label=f"**{target_year}년 예상 평균기온 (전체 기간 모델 기준)**", 
            value=f"{predicted_temp:.2f} °C"
        )
        
        st.markdown("---")
        st.subheader("📊 학습 데이터 정보")
        st.write(f"- **분석에 사용된 해의 개수:** `{num_years}`개")
        st.write(f"- **시작 연도:** `{start_year}`년")
        st.write(f"- **끝 연도:** `{end_year}`년")
        st.write(f"- **전체 기간 상관계수:** `{corr:.4f}`")

    with col2:
        st.subheader("📈 서울 연평균 기온 추이 및 회귀선 비교")
        
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

        # 회귀선 x축 범위 (1900 ~ 2100)
        line_years = np.arange(1900, 2101)
        line_elapsed = line_years - 1908

        # 1. 전체 기간 회귀선
        line_pred_full = model_full.predict(line_elapsed.reshape(-1, 1))
        fig.add_trace(
            go.Scatter(
                x=line_years, 
                y=line_pred_full, 
                mode="lines", 
                name="전체 기간 회귀선",
                line=dict(color="#ff7f0e", width=2, dash="dash")
            )
        )

        # 2. 최근 20년 회귀선
        line_pred_recent = model_recent.predict(line_elapsed.reshape(-1, 1))
        fig.add_trace(
            go.Scatter(
                x=line_years, 
                y=line_pred_recent, 
                mode="lines", 
                name="최근 20년 회귀선",
                line=dict(color="#2ca02c", width=2, dash="dot")
            )
        )

        # 예측 포인트
        fig.add_trace(
            go.Scatter(
                x=[target_year],
                y=[predicted_temp],
                mode="markers",
                name=f"선택 연도({target_year}년)",
                marker=dict(color="#d62728", size=13, symbol="star")
            )
        )

        # Layout 설정
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
