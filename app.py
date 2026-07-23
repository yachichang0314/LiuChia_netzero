
from __future__ import annotations

import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from optimizer import ModelInputs, solve_re100_lp


st.set_page_config(
    page_title="園區 RE100 太陽光電與儲能規劃工具",
    page_icon="☀️",
    layout="wide",
)

# =========================================================
# Style
# =========================================================
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f7faf8 0%, #eef5f1 100%);
    }
    .block-container {
        max-width: 1480px;
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }
    .hero {
        background: linear-gradient(135deg, #123c32 0%, #1d6c58 55%, #5fae8f 100%);
        color: white;
        border-radius: 24px;
        padding: 30px 34px;
        box-shadow: 0 12px 30px rgba(18, 60, 50, 0.20);
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        line-height: 1.7;
        opacity: 0.96;
    }
    .goal-card {
        background: white;
        border: 1px solid rgba(30, 70, 55, 0.09);
        border-radius: 18px;
        padding: 18px 20px;
        min-height: 185px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    }
    .goal-title {
        font-weight: 800;
        font-size: 1.08rem;
        color: #153b31;
        margin-bottom: 0.45rem;
    }
    .goal-note {
        color: #5b6d66;
        font-size: 0.91rem;
        line-height: 1.55;
    }
    .pill {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        background: #e6f3ed;
        color: #1d6c58;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 8px;
    }
    .kpi-card {
        background: white;
        border-radius: 18px;
        padding: 17px 19px;
        border: 1px solid rgba(20, 60, 45, 0.07);
        box-shadow: 0 5px 16px rgba(0,0,0,0.05);
        min-height: 132px;
    }
    .kpi-title {
        color: #60736b;
        font-size: 0.86rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        color: #153b31;
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .kpi-note {
        color: #7a8983;
        font-size: 0.80rem;
        line-height: 1.42;
    }
    .section-note {
        background: #f2f8f5;
        border-left: 4px solid #2f8b70;
        border-radius: 8px;
        padding: 12px 14px;
        color: #425a51;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Data / calculation helpers
# =========================================================
def uploaded_file_to_df(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV with common Traditional Chinese encodings."""
    raw = uploaded_file.getvalue()

    for encoding in ("utf-8-sig", "utf-8", "big5", "cp950"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("無法辨識 CSV 編碼，請另存為 UTF-8 CSV 後重新上傳。")


def prepare_model_input(
    load_df: pd.DataFrame,
    solar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and merge:
    1. load file: timestamp, load_kw
    2. solar file: timestamp, solar_profile
    """
    load_required = {"timestamp", "load_kw"}
    solar_required = {"timestamp", "solar_profile"}

    load_missing = load_required - set(load_df.columns)
    solar_missing = solar_required - set(solar_df.columns)

    if load_missing:
        raise ValueError(
            f"院區逐時用電檔缺少欄位：{', '.join(sorted(load_missing))}"
        )
    if solar_missing:
        raise ValueError(
            f"太陽光電 profile 檔缺少欄位：{', '.join(sorted(solar_missing))}"
        )

    load = load_df[["timestamp", "load_kw"]].copy()
    solar = solar_df[["timestamp", "solar_profile"]].copy()

    load["timestamp"] = pd.to_datetime(load["timestamp"], errors="coerce")
    solar["timestamp"] = pd.to_datetime(solar["timestamp"], errors="coerce")
    load["load_kw"] = pd.to_numeric(load["load_kw"], errors="coerce")
    solar["solar_profile"] = pd.to_numeric(
        solar["solar_profile"], errors="coerce"
    )

    load = load.dropna(subset=["timestamp", "load_kw"])
    solar = solar.dropna(subset=["timestamp", "solar_profile"])

    if load["timestamp"].duplicated().any():
        duplicate_count = int(load["timestamp"].duplicated().sum())
        raise ValueError(f"院區逐時用電檔有 {duplicate_count} 筆重複 timestamp。")

    if solar["timestamp"].duplicated().any():
        duplicate_count = int(solar["timestamp"].duplicated().sum())
        raise ValueError(
            f"太陽光電 profile 檔有 {duplicate_count} 筆重複 timestamp。"
        )

    if (load["load_kw"] < 0).any():
        raise ValueError("load_kw 不可小於 0。")

    if (solar["solar_profile"] < 0).any():
        raise ValueError("solar_profile 不可小於 0。")

    load = load.sort_values("timestamp")
    solar = solar.sort_values("timestamp")

    merged = pd.merge(
        load,
        solar,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp").reset_index(drop=True)

    if merged.empty:
        raise ValueError("兩個檔案沒有相同的 timestamp，無法合併。")

    load_only = len(load) - len(merged)
    solar_only = len(solar) - len(merged)

    merged.attrs["load_rows"] = len(load)
    merged.attrs["solar_rows"] = len(solar)
    merged.attrs["load_unmatched_rows"] = load_only
    merged.attrs["solar_unmatched_rows"] = solar_only

    return merged


def make_kpi_card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def currency(v: float) -> str:
    return f"{v:,.0f}"


# =========================================================
# Session state
# =========================================================
if "result" not in st.session_state:
    st.session_state["result"] = None


# =========================================================
# Header
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">☀️ 園區 RE100 太陽光電與儲能規劃工具</div>
        <div class="hero-subtitle">
            根據園區逐時用電與太陽光電發電曲線，評估三種能源目標，
            並尋找太陽光電容量、儲能功率與儲能容量的適合配置。
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

g1, g2, g3 = st.columns(3)
with g1:
    st.markdown(
        """
        <div class="goal-card">
            <span class="pill">目標一</span>
            <div class="goal-title">年度 RE100 會計達成</div>
            <div class="goal-note">
                允許實際向電網購電，但以自建光電的再生能源屬性及外購憑證，
                在年度總量上涵蓋全部用電。此模式通常不一定需要儲能。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with g2:
    st.markdown(
        """
        <div class="goal-card">
            <span class="pill">目標二</span>
            <div class="goal-title">年度 RE100＋提高現地自用</div>
            <div class="goal-note">
                年度帳面仍達成 RE100，但同時提高園區現地太陽光電與儲能供應比例，
                降低灰電購電與棄電。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with g3:
    st.markdown(
        """
        <div class="goal-card">
            <span class="pill">目標三</span>
            <div class="goal-title">逐時綠電匹配</div>
            <div class="goal-note">
                每小時以太陽光電直接供電或由太陽光電充入的儲能放電支應負載。
                可設定逐時 100% 達成的小時比例。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Sidebar controls
# =========================================================
with st.sidebar:
    st.title("規劃條件")

    load_file = st.file_uploader(
        "1. 上傳院區逐時用電 CSV",
        type=["csv"],
        key="load_file",
        help="必要欄位：timestamp、load_kw",
    )

    solar_file = st.file_uploader(
        "2. 上傳太陽光電逐時 profile CSV",
        type=["csv"],
        key="solar_file",
        help="必要欄位：timestamp、solar_profile",
    )

    goal_label = st.radio(
        "選擇規劃目標",
        [
            "目標一｜年度 RE100 會計達成",
            "目標二｜年度 RE100＋提高現地自用",
            "目標三｜逐時綠電匹配",
        ],
    )

    goal_map = {
        "目標一｜年度 RE100 會計達成": "annual_re100",
        "目標二｜年度 RE100＋提高現地自用": "annual_re100_self_consumption",
        "目標三｜逐時綠電匹配": "hourly_matching",
    }
    goal_mode = goal_map[goal_label]

    onsite_clean_target = 0.70
    hourly_matching_target = 1.00

    if goal_mode == "annual_re100_self_consumption":
        onsite_clean_target = st.slider(
            "現地綠電供應率最低目標",
            min_value=0.00,
            max_value=1.00,
            value=0.70,
            step=0.01,
        )

    if goal_mode == "hourly_matching":
        hourly_matching_target = st.slider(
            "每小時最低綠電供應比例",
            min_value=0.50,
            max_value=1.00,
            value=1.00,
            step=0.01,
        )

    st.markdown("---")
    st.subheader("容量上限")
    max_pv_kw = st.number_input("太陽光電最大可建置容量 (kW)", min_value=0.0, value=10000.0, step=100.0)
    max_battery_power_kw = st.number_input("儲能最大功率 (kW)", min_value=0.0, value=5000.0, step=100.0)
    max_battery_energy_kwh = st.number_input("儲能最大容量 (kWh)", min_value=0.0, value=30000.0, step=500.0)

    st.markdown("---")
    st.subheader("成本參數")
    pv_capex_per_kw = st.number_input("光電建置成本 (元/kW)", min_value=0.0, value=40000.0, step=1000.0)
    pv_om_per_kw_year = st.number_input("光電年維運成本 (元/kW-年)", min_value=0.0, value=500.0, step=50.0)
    battery_power_cost_per_kw = st.number_input("儲能功率成本 (元/kW)", min_value=0.0, value=7000.0, step=500.0)
    battery_energy_cost_per_kwh = st.number_input("儲能容量成本 (元/kWh)", min_value=0.0, value=12000.0, step=500.0)
    grid_price = st.number_input("電網購電價格 (元/kWh)", min_value=0.0, value=4.0, step=0.1)
    certificate_price = st.number_input("外購綠電憑證成本 (元/kWh)", min_value=0.0, value=1.5, step=0.1)

    with st.expander("進階技術參數"):
        round_trip_efficiency = st.slider("儲能往返效率", 0.50, 1.00, 0.90, 0.01)
        min_soc = st.slider("最小 SOC", 0.0, 0.5, 0.10, 0.05)
        max_soc = st.slider("最大 SOC", 0.5, 1.0, 0.90, 0.05)
        battery_om_ratio = st.number_input("儲能年維運費占建置成本比例", 0.0, 0.2, 0.02, 0.005)
        pv_life_years = st.number_input("光電使用年限", 1, 40, 20, 1)
        battery_life_years = st.number_input("儲能使用年限", 1, 30, 10, 1)
        discount_rate = st.number_input("折現率", 0.0, 0.2, 0.05, 0.005)

    run_btn = st.button("開始規劃", type="primary", use_container_width=True)
    clear_btn = st.button("清除結果", use_container_width=True)

    if clear_btn:
        st.session_state["result"] = None


# =========================================================
# Main tabs
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["使用說明", "資料預覽", "規劃結果", "指標定義"])

with tab1:
    st.markdown("### 資料格式")
    f1, f2 = st.columns(2)

    with f1:
        st.markdown("#### 院區逐時用電檔")
        st.code(
            "timestamp,load_kw\n"
            "2025-01-01 00:00:00,850\n"
            "2025-01-01 01:00:00,810\n"
            "2025-01-01 12:00:00,1060",
            language="csv",
        )

    with f2:
        st.markdown("#### 太陽光電逐時 profile 檔")
        st.code(
            "timestamp,solar_profile\n"
            "2025-01-01 00:00:00,0\n"
            "2025-01-01 01:00:00,0\n"
            "2025-01-01 12:00:00,0.78",
            language="csv",
        )
    st.markdown(
        """
        <div class="section-note">
        <b>solar_profile</b> 為每 1 kW 太陽光電在該小時可發出的電量係數。
        若為逐時資料，通常可用 0～1 的容量因子表示。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 三種目標的差異")
    st.dataframe(
        pd.DataFrame(
            {
                "項目": ["是否允許灰電", "是否允許外購憑證", "儲能必要性", "主要最佳化重點"],
                "年度 RE100 會計達成": ["允許", "允許", "通常非必要", "最低年化成本"],
                "年度 RE100＋提高現地自用": ["允許，但希望降低", "允許", "通常有助益", "成本與現地綠電利用"],
                "逐時綠電匹配": ["未達標小時才允許", "不計入物理匹配", "通常必要", "逐時達標與最低成本"],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    if load_file is None and solar_file is None:
        st.info("請從左側分別上傳院區逐時用電檔與太陽光電 profile 檔。")
    else:
        p1, p2 = st.columns(2)

        with p1:
            st.markdown("### 院區逐時用電")
            if load_file is None:
                st.info("尚未上傳院區逐時用電檔。")
            else:
                try:
                    load_preview = uploaded_file_to_df(load_file)
                    st.dataframe(
                        load_preview.head(100),
                        use_container_width=True,
                        hide_index=True,
                    )
                except Exception as exc:
                    st.error(str(exc))

        with p2:
            st.markdown("### 太陽光電逐時 profile")
            if solar_file is None:
                st.info("尚未上傳太陽光電 profile 檔。")
            else:
                try:
                    solar_preview = uploaded_file_to_df(solar_file)
                    st.dataframe(
                        solar_preview.head(100),
                        use_container_width=True,
                        hide_index=True,
                    )
                except Exception as exc:
                    st.error(str(exc))

        if load_file is not None and solar_file is not None:
            try:
                preview_df = prepare_model_input(
                    uploaded_file_to_df(load_file),
                    uploaded_file_to_df(solar_file),
                )

                st.markdown("### 合併後模型資料")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("共同資料筆數", f"{len(preview_df):,}")
                c2.metric(
                    "期間",
                    f"{preview_df['timestamp'].min():%Y-%m-%d} 至 "
                    f"{preview_df['timestamp'].max():%Y-%m-%d}",
                )
                c3.metric(
                    "總用電量",
                    f"{preview_df['load_kw'].sum():,.0f} kWh",
                )
                c4.metric(
                    "每 kW 光電年發電量",
                    f"{preview_df['solar_profile'].sum():,.1f} kWh/kW",
                )

                unmatched_load = preview_df.attrs.get(
                    "load_unmatched_rows", 0
                )
                unmatched_solar = preview_df.attrs.get(
                    "solar_unmatched_rows", 0
                )
                if unmatched_load or unmatched_solar:
                    st.warning(
                        "兩檔時間未完全對齊："
                        f"用電檔有 {unmatched_load} 筆未配對，"
                        f"光電檔有 {unmatched_solar} 筆未配對。"
                        "模型只使用 timestamp 相同的資料。"
                    )
                else:
                    st.success("兩個檔案的 timestamp 已完整配對。")

                st.dataframe(
                    preview_df.head(100),
                    use_container_width=True,
                    hide_index=True,
                )

                chart_df = preview_df.set_index("timestamp")[
                    ["load_kw", "solar_profile"]
                ]
                fig_preview = go.Figure()
                fig_preview.add_trace(
                    go.Scatter(
                        x=chart_df.index,
                        y=chart_df["load_kw"],
                        name="Load (kW)",
                    )
                )
                fig_preview.add_trace(
                    go.Scatter(
                        x=chart_df.index,
                        y=chart_df["solar_profile"],
                        name="Solar profile",
                        yaxis="y2",
                    )
                )
                fig_preview.update_layout(
                    height=430,
                    yaxis=dict(title="Load (kW)"),
                    yaxis2=dict(
                        title="Solar profile",
                        overlaying="y",
                        side="right",
                    ),
                    margin=dict(l=20, r=20, t=25, b=20),
                )
                st.plotly_chart(fig_preview, use_container_width=True)

            except Exception as exc:
                st.error(f"資料合併失敗：{exc}")

with tab3:
    if run_btn:
        if load_file is None or solar_file is None:
            st.error("請先上傳院區逐時用電檔與太陽光電 profile 檔。")
        else:
            try:
                df = prepare_model_input(
                    uploaded_file_to_df(load_file),
                    uploaded_file_to_df(solar_file),
                )

                if min_soc >= max_soc:
                    raise ValueError("最小 SOC 必須小於最大 SOC。")

                model_inputs = ModelInputs(
                    pv_capex_per_kw=float(pv_capex_per_kw),
                    pv_om_per_kw_year=float(pv_om_per_kw_year),
                    pv_life_years=int(pv_life_years),
                    battery_power_capex_per_kw=float(battery_power_cost_per_kw),
                    battery_energy_capex_per_kwh=float(battery_energy_cost_per_kwh),
                    battery_om_ratio=float(battery_om_ratio),
                    battery_life_years=int(battery_life_years),
                    grid_price_per_kwh=float(grid_price),
                    certificate_price_per_kwh=float(certificate_price),
                    round_trip_efficiency=float(round_trip_efficiency),
                    initial_soc_ratio=float(min_soc),
                    min_soc_ratio=float(min_soc),
                    max_soc_ratio=float(max_soc),
                    discount_rate=float(discount_rate),
                    max_pv_kw=float(max_pv_kw),
                    max_battery_power_kw=float(max_battery_power_kw),
                    max_battery_energy_kwh=float(max_battery_energy_kwh),
                )

                with st.spinner("正在執行線性規劃，計算最佳光電與儲能配置…"):
                    result = solve_re100_lp(
                        df=df,
                        inputs=model_inputs,
                        goal_mode=goal_mode,
                        onsite_clean_target=float(onsite_clean_target),
                        hourly_matching_target=float(hourly_matching_target),
                        solver_name="CBC",
                        time_limit_seconds=120,
                    )

                st.session_state["result"] = {
                    "goal_mode": goal_mode,
                    "goal_label": goal_label,
                    "onsite_clean_target": onsite_clean_target,
                    "hourly_matching_target": hourly_matching_target,
                    "result": result,
                }

            except Exception as exc:
                st.error(f"規劃失敗：{exc}")

    state = st.session_state.get("result")

    if state is None:
        st.info("設定條件後，按左側「開始規劃」。")
    elif state["result"] is None:
        st.error("在目前容量上限與目標條件下找不到可行配置。請提高光電或儲能上限，或降低逐時達標比例。")
    else:
        result = state["result"]
        ts = result["timeseries"].copy()

        st.markdown(f"### {state['goal_label']}｜建議配置")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            make_kpi_card("太陽光電容量", f"{result['pv_capacity_kw']:,.0f} kW", "模型搜尋得到的建議裝置容量")
        with k2:
            make_kpi_card("儲能功率", f"{result['battery_power_kw']:,.0f} kW", "最大充放電功率")
        with k3:
            make_kpi_card("儲能容量", f"{result['battery_energy_kwh']:,.0f} kWh", "可儲存的電量")
        with k4:
            make_kpi_card("儲能時數", f"{result['battery_duration_h']:.1f} 小時", "儲能容量 ÷ 儲能功率")

        k5, k6, k7, k8 = st.columns(4)
        with k5:
            make_kpi_card("年度 RE100 達成率", f"{result['annual_re_ratio']:.1%}", "自有再生能源屬性＋外購憑證")
        with k6:
            make_kpi_card("現地綠電供應率", f"{result['onsite_clean_ratio']:.1%}", "光電直接供應＋綠電儲能放電")
        with k7:
            make_kpi_card("逐時 100% 達成率", f"{result['hourly_100_ratio']:.1%}", "每小時完全由現地綠電支應的比例")
        with k8:
            make_kpi_card("年化總成本", f"{currency(result['total_annual_cost'])} 元", "光電＋儲能＋電網＋憑證")

        left, right = st.columns([1.0, 1.15])
        with left:
            st.markdown("### 電量與憑證概覽")
            overview = pd.DataFrame(
                {
                    "項目": [
                        "全年用電量",
                        "太陽光電發電量",
                        "現地綠電供應量",
                        "台電購電量",
                        "棄電量",
                        "外購憑證需求量",
                    ],
                    "數值 (kWh)": [
                        result["total_load_kwh"],
                        result["total_pv_generation_kwh"],
                        result["total_clean_supply_kwh"],
                        result["total_grid_purchase_kwh"],
                        result["total_curtailment_kwh"],
                        result["external_certificate_kwh"],
                    ],
                }
            )
            st.dataframe(overview, use_container_width=True, hide_index=True)

        with right:
            st.markdown("### 年化成本拆解")
            cost_df = pd.DataFrame(
                {
                    "成本項目": ["太陽光電", "儲能", "電網購電", "外購憑證"],
                    "年化成本": [
                        result["pv_annual_cost"],
                        result["battery_annual_cost"],
                        result["grid_annual_cost"],
                        result["certificate_annual_cost"],
                    ],
                }
            )
            fig_cost = px.bar(cost_df, x="成本項目", y="年化成本", text_auto=".3s")
            fig_cost.update_layout(height=390, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_cost, use_container_width=True)

        sub1, sub2, sub3, sub4 = st.tabs(["指定日期供需圖", "SOC 曲線", "逐時綠電熱圖", "時序明細"])

        with sub1:
            ts["date_only"] = pd.to_datetime(ts["timestamp"]).dt.date
            available_dates = sorted(ts["date_only"].unique())
            selected_date = st.selectbox("選擇日期", available_dates, key="dispatch_day")
            day = ts[ts["date_only"] == selected_date]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=day["timestamp"], y=day["pv_direct_use"], name="PV direct"))
            fig.add_trace(go.Bar(x=day["timestamp"], y=day["battery_discharge"], name="Battery discharge"))
            fig.add_trace(go.Bar(x=day["timestamp"], y=day["grid_purchase"], name="Grid purchase"))
            fig.add_trace(go.Scatter(x=day["timestamp"], y=day["load_kw"], name="Load", mode="lines"))
            fig.update_layout(
                barmode="stack",
                height=500,
                yaxis_title="kWh",
                margin=dict(l=20, r=20, t=25, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with sub2:
            fig_soc = go.Figure()
            fig_soc.add_trace(go.Scatter(x=ts["timestamp"], y=ts["soc_kwh"], name="SOC (kWh)"))
            fig_soc.update_layout(height=430, yaxis_title="kWh", margin=dict(l=20, r=20, t=25, b=20))
            st.plotly_chart(fig_soc, use_container_width=True)

        with sub3:
            heat = ts.copy()
            heat["datetime"] = pd.to_datetime(heat["timestamp"])
            heat["date"] = heat["datetime"].dt.date
            heat["hour"] = heat["datetime"].dt.hour
            pivot = heat.pivot(index="date", columns="hour", values="hourly_clean_ratio")
            fig_heat = px.imshow(pivot, aspect="auto", zmin=0, zmax=1)
            fig_heat.update_layout(height=720, xaxis_title="Hour", yaxis_title="Date")
            st.plotly_chart(fig_heat, use_container_width=True)

        with sub4:
            st.dataframe(ts, use_container_width=True, hide_index=True)

            output = ts.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下載時序結果 CSV",
                data=output,
                file_name="re100_planning_timeseries.csv",
                mime="text/csv",
            )

with tab4:
    st.markdown("### 指標定義")
    st.dataframe(
        pd.DataFrame(
            {
                "指標": [
                    "年度 RE100 達成率",
                    "現地綠電供應率",
                    "逐時 100% 達成率",
                    "外購憑證需求量",
                    "儲能時數",
                ],
                "定義": [
                    "自建光電所保留的再生能源屬性與外購憑證合計，除以全年用電量。",
                    "太陽光電直接供電加上由太陽光電充入之儲能放電，除以全年用電量。",
                    "全年各小時中，現地綠電供應完全涵蓋該小時負載的比例。",
                    "為使年度再生能源屬性覆蓋全年用電，仍需額外取得並註銷的憑證電量。",
                    "儲能容量 kWh 除以儲能功率 kW。",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "本版為網站與容量搜尋原型，採離散搜尋方式產生建議配置；"
        "後續若要作為正式投資決策工具，建議再接回線性規劃或混合整數規劃模型。"
    )
