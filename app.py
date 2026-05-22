import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# =========================
# 页面基础设置
# =========================

st.set_page_config(
    page_title="金融资产收益与风险分析平台",
    page_icon="📊",
    layout="wide"
)

# =========================
# CSS 页面美化
# =========================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 48%, #111827 100%);
        color: #e5e7eb;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }

    .sub-title {
        font-size: 16px;
        color: #94a3b8;
        margin-bottom: 28px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 28px;
        margin-bottom: 14px;
    }

    .info-box {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 18px;
        padding: 18px 22px;
        color: #cbd5e1;
        line-height: 1.8;
        box-shadow: 0 12px 35px rgba(0,0,0,0.28);
        margin-bottom: 16px;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.90);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
        margin-bottom: 12px;
        min-height: 118px;
    }

    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #f8fafc;
    }

    .positive {
        color: #22c55e;
    }

    .negative {
        color: #ef4444;
    }

    .neutral {
        color: #eab308;
    }

    .risk-low {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        padding: 8px 14px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

    .risk-mid {
        background: rgba(234, 179, 8, 0.15);
        color: #eab308;
        padding: 8px 14px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

    .risk-high {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        padding: 8px 14px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

   .watermark {
    margin-top: 70px;
    padding-top: 35px;
    padding-bottom: 30px;
    border-top: 1px solid rgba(148, 163, 184, 0.25);
    text-align: center;
    color: rgba(226, 232, 240, 0.65);
    font-size: 24px;
    letter-spacing: 2px;
}

.watermark span {
    color: rgba(248, 250, 252, 0.92);
    font-weight: 900;
    font-size: 32px;

    }

    div[data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid rgba(148, 163, 184, 0.22);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# 函数区
# =========================

def metric_card(label, value, css_class=""):
    st.metric(label=label, value=value)


def calculate_metrics(price_series, risk_free_rate_percent):
    """
    根据价格序列计算金融风险收益指标。
    """
    price_series = price_series.dropna()

    if len(price_series) < 2:
        return None

    returns = price_series.pct_change().dropna()

    start_price = price_series.iloc[0]
    end_price = price_series.iloc[-1]

    total_return = (end_price / start_price - 1) * 100

    periods = len(price_series) - 1
    annualized_return = ((end_price / start_price) ** (12 / periods) - 1) * 100

    annualized_volatility = returns.std() * np.sqrt(12) * 100

    cumulative_max = price_series.cummax()
    drawdown = (price_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100

    risk_free_monthly = risk_free_rate_percent / 100 / 12
    excess_return = returns - risk_free_monthly

    if returns.std() != 0 and len(returns) > 1:
        sharpe_ratio = excess_return.mean() / returns.std() * np.sqrt(12)
    else:
        sharpe_ratio = 0

    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
    best_period = returns.max() * 100 if len(returns) > 0 else 0
    worst_period = returns.min() * 100 if len(returns) > 0 else 0

    return {
        "start_price": start_price,
        "end_price": end_price,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "best_period": best_period,
        "worst_period": worst_period,
        "returns": returns,
        "drawdown": drawdown
    }


def get_risk_level(metrics):
    """
    根据波动率、最大回撤和夏普比率生成风险等级。
    """
    score = 0

    if metrics["annualized_volatility"] > 40:
        score += 3
    elif metrics["annualized_volatility"] > 25:
        score += 2
    else:
        score += 1

    if metrics["max_drawdown"] < -30:
        score += 3
    elif metrics["max_drawdown"] < -15:
        score += 2
    else:
        score += 1

    if metrics["sharpe_ratio"] < 0.5:
        score += 3
    elif metrics["sharpe_ratio"] < 1:
        score += 2
    else:
        score += 1

    if score <= 4:
        return "低风险", score, "risk-low"
    elif score <= 7:
        return "中等风险", score, "risk-mid"
    else:
        return "高风险", score, "risk-high"


def build_demo_data():
    """
    构造演示数据。
    """
    data = {
        "日期": pd.date_range(start="2025-01-01", periods=24, freq="MS"),
        "AAPL": [
            180, 185, 178, 190, 195, 205, 210, 208, 215, 225, 230, 238,
            240, 235, 245, 250, 258, 260, 268, 275, 280, 285, 292, 300
        ],
        "NVDA": [
            480, 520, 500, 560, 610, 700, 760, 740, 810, 880, 920, 980,
            1020, 990, 1080, 1160, 1210, 1300, 1380, 1350, 1440, 1520, 1600, 1680
        ],
        "JPM": [
            160, 162, 158, 165, 170, 172, 168, 175, 180, 182, 185, 190,
            188, 192, 195, 198, 202, 205, 207, 210, 213, 215, 218, 220
        ],
        "MSFT": [
            370, 380, 375, 390, 400, 410, 420, 415, 430, 445, 455, 465,
            470, 460, 475, 490, 505, 515, 525, 535, 545, 558, 570, 585
        ],
        "TSLA": [
            240, 260, 230, 280, 300, 290, 330, 310, 350, 370, 340, 390,
            410, 380, 430, 450, 420, 470, 500, 460, 520, 540, 510, 560
        ]
    }

    demo_df = pd.DataFrame(data)
    demo_df["日期"] = pd.to_datetime(demo_df["日期"])
    demo_df = demo_df.set_index("日期")
    return demo_df


def read_uploaded_excel(uploaded_file):
    """
    读取用户上传的 Excel。
    要求：第一列是日期，后面的列是资产价格。
    """
    raw_df = pd.read_excel(uploaded_file)

    if raw_df.shape[1] < 2:
        st.error("Excel 至少需要两列：第一列日期 + 至少一个资产价格列。")
        st.stop()

    date_col = raw_df.columns[0]
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors="coerce")
    raw_df = raw_df.dropna(subset=[date_col])
    raw_df = raw_df.set_index(date_col)

    for col in raw_df.columns:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    raw_df = raw_df.dropna(how="all")
    raw_df = raw_df.sort_index()

    return raw_df


# =========================
# 页面标题
# =========================

st.markdown('<div class="main-title">金融资产收益与风险分析平台</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">支持 Excel 上传，自动计算收益率、波动率、最大回撤、夏普比率、相关性与风险评级</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
<b>使用说明：</b>上传一个 Excel 文件，第一列必须是日期，后面的列为不同资产价格。
如果暂时没有 Excel，可以在左侧控制面板勾选“使用演示数据”。
</div>
""", unsafe_allow_html=True)


# =========================
# 侧边栏控制面板
# =========================

st.sidebar.markdown("## 控制面板")

use_demo = st.sidebar.checkbox("使用演示数据", value=True)

uploaded_file = st.sidebar.file_uploader(
    "上传 Excel 文件",
    type=["xlsx"]
)

risk_free_rate = st.sidebar.slider(
    "无风险收益率（年化）",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.1
)

show_raw_data = st.sidebar.checkbox("显示原始数据", value=False)


# =========================
# 数据读取
# =========================

if use_demo:
    df = build_demo_data()
else:
    if uploaded_file is None:
        st.warning("请在左侧上传 Excel 文件，或者勾选“使用演示数据”。")
        st.stop()

    try:
        df = read_uploaded_excel(uploaded_file)
    except Exception as e:
        st.error("Excel 读取失败，请检查文件格式。")
        st.write(e)
        st.stop()

asset_list = list(df.columns)

if len(asset_list) == 0:
    st.error("没有识别到资产价格列，请检查 Excel 文件。")
    st.stop()


selected_asset = st.sidebar.selectbox(
    "选择单只资产",
    asset_list
)

compare_assets = st.sidebar.multiselect(
    "选择对比资产",
    asset_list,
    default=asset_list[:min(4, len(asset_list))]
)

metrics = calculate_metrics(df[selected_asset], risk_free_rate)

if metrics is None:
    st.error("当前资产有效价格数据不足，至少需要两个价格点。")
    st.stop()

level, score, level_class = get_risk_level(metrics)


# =========================
# 顶部状态栏
# =========================

st.markdown(f"""
<div class="info-box">
当前分析资产：<b>{selected_asset}</b>　
风险等级：<span class="{level_class}">{level}</span>　
风险评分：<b>{score}</b>　
样本数量：<b>{len(df)}</b>　
无风险收益率假设：<b>{risk_free_rate:.1f}%</b>
</div>
""", unsafe_allow_html=True)


# =========================
# 核心指标卡片
# =========================

st.markdown('<div class="section-title">核心指标概览</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    css = "positive" if metrics["total_return"] >= 0 else "negative"
    metric_card("阶段收益率", f"{metrics['total_return']:.2f}%", css)

with col2:
    css = "positive" if metrics["annualized_return"] >= 0 else "negative"
    metric_card("年化收益率", f"{metrics['annualized_return']:.2f}%", css)

with col3:
    metric_card("年化波动率", f"{metrics['annualized_volatility']:.2f}%", "neutral")

with col4:
    metric_card("最大回撤", f"{metrics['max_drawdown']:.2f}%", "negative")

col5, col6, col7, col8 = st.columns(4)

with col5:
    css = "positive" if metrics["sharpe_ratio"] >= 1 else "neutral"
    metric_card("夏普比率", f"{metrics['sharpe_ratio']:.2f}", css)

with col6:
    metric_card("上涨胜率", f"{metrics['win_rate']:.2f}%", "positive")

with col7:
    metric_card("最佳单期收益", f"{metrics['best_period']:.2f}%", "positive")

with col8:
    metric_card("最差单期收益", f"{metrics['worst_period']:.2f}%", "negative")


# =========================
# 价格走势
# =========================

st.markdown('<div class="section-title">价格走势</div>', unsafe_allow_html=True)

fig_price = go.Figure()

fig_price.add_trace(go.Scatter(
    x=df.index,
    y=df[selected_asset],
    mode="lines+markers",
    name=selected_asset,
    line=dict(width=3),
    marker=dict(size=6)
))

fig_price.update_layout(
    template="plotly_dark",
    height=460,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.65)",
    title=f"{selected_asset} Price Trend",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig_price, use_container_width=True)


# =========================
# 收益率与回撤
# =========================

left, right = st.columns(2)

with left:
    st.markdown('<div class="section-title">收益率变化</div>', unsafe_allow_html=True)

    returns_percent = metrics["returns"] * 100

    fig_return = go.Figure()

    fig_return.add_trace(go.Bar(
        x=returns_percent.index,
        y=returns_percent,
        name="Return"
    ))

    fig_return.update_layout(
        template="plotly_dark",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.65)",
        title=f"{selected_asset} Period Return",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig_return, use_container_width=True)

with right:
    st.markdown('<div class="section-title">回撤曲线</div>', unsafe_allow_html=True)

    drawdown_percent = metrics["drawdown"] * 100

    fig_drawdown = go.Figure()

    fig_drawdown.add_trace(go.Scatter(
        x=drawdown_percent.index,
        y=drawdown_percent,
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(width=3)
    ))

    fig_drawdown.update_layout(
        template="plotly_dark",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.65)",
        title=f"{selected_asset} Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig_drawdown, use_container_width=True)


# =========================
# 多资产对比
# =========================

st.markdown('<div class="section-title">多资产对比分析</div>', unsafe_allow_html=True)

if len(compare_assets) == 0:
    st.warning("请至少选择一个资产进行对比。")
else:
    compare_rows = []

    for asset in compare_assets:
        m = calculate_metrics(df[asset], risk_free_rate)

        if m is not None:
            compare_rows.append({
                "资产": asset,
                "阶段收益率": m["total_return"],
                "年化收益率": m["annualized_return"],
                "年化波动率": m["annualized_volatility"],
                "最大回撤": m["max_drawdown"],
                "夏普比率": m["sharpe_ratio"],
                "上涨胜率": m["win_rate"]
            })

    compare_df = pd.DataFrame(compare_rows)

    st.dataframe(
        compare_df.style.format({
            "阶段收益率": "{:.2f}%",
            "年化收益率": "{:.2f}%",
            "年化波动率": "{:.2f}%",
            "最大回撤": "{:.2f}%",
            "夏普比率": "{:.2f}",
            "上涨胜率": "{:.2f}%"
        }),
        use_container_width=True
    )

    fig_compare = go.Figure()

    for asset in compare_assets:
        valid_series = df[asset].dropna()

        if len(valid_series) >= 2:
            normalized = valid_series / valid_series.iloc[0] * 100

            fig_compare.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized,
                mode="lines",
                name=asset,
                line=dict(width=3)
            ))

    fig_compare.update_layout(
        template="plotly_dark",
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.65)",
        title="Normalized Price Comparison",
        xaxis_title="Date",
        yaxis_title="Start = 100",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig_compare, use_container_width=True)


# =========================
# 风险收益分布
# =========================

st.markdown('<div class="section-title">风险收益分布</div>', unsafe_allow_html=True)

risk_return_rows = []

for asset in asset_list:
    m = calculate_metrics(df[asset], risk_free_rate)

    if m is not None:
        risk_return_rows.append({
            "资产": asset,
            "年化收益率": m["annualized_return"],
            "年化波动率": m["annualized_volatility"],
            "夏普比率": max(m["sharpe_ratio"], 0.1),
            "最大回撤": m["max_drawdown"]
        })

risk_return_df = pd.DataFrame(risk_return_rows)

fig_scatter = px.scatter(
    risk_return_df,
    x="年化波动率",
    y="年化收益率",
    size="夏普比率",
    hover_name="资产",
    text="资产",
    title="Risk-Return Map",
    template="plotly_dark"
)

fig_scatter.update_traces(textposition="top center")

fig_scatter.update_layout(
    height=500,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.65)",
    xaxis_title="Annualized Volatility (%)",
    yaxis_title="Annualized Return (%)",
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig_scatter, use_container_width=True)


# =========================
# 资产相关性矩阵
# =========================

st.markdown('<div class="section-title">资产相关性矩阵</div>', unsafe_allow_html=True)

returns_df = df.pct_change().dropna()

if returns_df.shape[1] >= 2 and len(returns_df) >= 2:
    corr = returns_df.corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    fig_corr.update_layout(
        template="plotly_dark",
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.65)",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("资产数量或有效数据不足，暂时无法生成相关性矩阵。")


# =========================
# 自动分析结论
# =========================

st.markdown('<div class="section-title">自动分析结论</div>', unsafe_allow_html=True)

conclusion = ""

if metrics["total_return"] > 50:
    conclusion += f"{selected_asset} 在样本期内收益表现很强，阶段收益率达到 {metrics['total_return']:.2f}%。"
elif metrics["total_return"] > 20:
    conclusion += f"{selected_asset} 在样本期内取得较好收益，阶段收益率为 {metrics['total_return']:.2f}%。"
elif metrics["total_return"] > 0:
    conclusion += f"{selected_asset} 在样本期内小幅上涨，阶段收益率为 {metrics['total_return']:.2f}%。"
else:
    conclusion += f"{selected_asset} 在样本期内出现负收益，阶段收益率为 {metrics['total_return']:.2f}%。"

if metrics["annualized_volatility"] > 40:
    conclusion += f" 该资产年化波动率达到 {metrics['annualized_volatility']:.2f}%，价格波动较大，风险水平偏高。"
elif metrics["annualized_volatility"] > 25:
    conclusion += f" 该资产年化波动率为 {metrics['annualized_volatility']:.2f}%，属于中等偏高水平。"
else:
    conclusion += f" 该资产年化波动率为 {metrics['annualized_volatility']:.2f}%，价格稳定性相对较好。"

if metrics["max_drawdown"] < -30:
    conclusion += f" 最大回撤为 {metrics['max_drawdown']:.2f}%，说明曾出现明显下跌。"
elif metrics["max_drawdown"] < -15:
    conclusion += f" 最大回撤为 {metrics['max_drawdown']:.2f}%，存在一定回撤风险。"
else:
    conclusion += f" 最大回撤为 {metrics['max_drawdown']:.2f}%，回撤控制相对较好。"

if metrics["sharpe_ratio"] > 1:
    conclusion += f" 夏普比率为 {metrics['sharpe_ratio']:.2f}，风险调整后收益较好。"
elif metrics["sharpe_ratio"] > 0.5:
    conclusion += f" 夏普比率为 {metrics['sharpe_ratio']:.2f}，风险调整后收益一般。"
else:
    conclusion += f" 夏普比率为 {metrics['sharpe_ratio']:.2f}，单位风险带来的收益补偿不足。"

st.markdown(f"""
<div class="info-box">
{conclusion}
<br><br>
注意：本平台用于金融数据分析练习，不构成投资建议。若使用真实数据，应进一步结合基本面、估值水平、宏观环境和行业周期进行判断。
</div>
""", unsafe_allow_html=True)


# =========================
# 原始数据展示
# =========================

if show_raw_data:
    st.markdown('<div class="section-title">原始数据表</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)


# =========================
# 底部水印
# =========================

st.divider()
st.markdown(
    "<h2 style='text-align:center; color:#94a3b8;'>小新SCX</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#64748b;'>Financial Analytics Dashboard</p >",
    unsafe_allow_html=True
)
