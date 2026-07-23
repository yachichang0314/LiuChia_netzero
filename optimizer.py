"""
optimizer.py
============

園區 RE100 太陽光電、儲能與憑證最佳化模型。

支援三種目標：
1. annual_re100
2. annual_re100_self_consumption
3. hourly_matching

重要假設：
- 儲能只能由太陽光電充電，不允許台電充電。
- 外購憑證只用於年度 RE100 會計，不計入逐時物理匹配。
- 太陽光電容量、儲能功率與儲能容量均為連續決策變數。
- 採線性規劃 LP。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pulp


@dataclass
class ModelInputs:
    pv_capex_per_kw: float = 40000.0
    pv_om_per_kw_year: float = 500.0
    pv_life_years: int = 20

    battery_power_capex_per_kw: float = 7000.0
    battery_energy_capex_per_kwh: float = 12000.0
    battery_om_ratio: float = 0.02
    battery_life_years: int = 10

    grid_price_per_kwh: float = 4.0
    certificate_price_per_kwh: float = 1.5

    round_trip_efficiency: float = 0.90
    initial_soc_ratio: float = 0.10
    min_soc_ratio: float = 0.10
    max_soc_ratio: float = 0.90

    discount_rate: float = 0.05

    max_pv_kw: float = 10000.0
    max_battery_power_kw: float = 5000.0
    max_battery_energy_kwh: float = 30000.0


def annuity_factor(rate: float, years: int) -> float:
    if years <= 0:
        raise ValueError("使用年限必須大於 0。")
    if abs(rate) < 1e-12:
        return 1.0 / years
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def validate_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "load_kw", "solar_profile"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要欄位：{', '.join(sorted(missing))}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["load_kw"] = pd.to_numeric(out["load_kw"], errors="coerce")
    out["solar_profile"] = pd.to_numeric(out["solar_profile"], errors="coerce")
    out = out.dropna(subset=["timestamp", "load_kw", "solar_profile"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    if out.empty:
        raise ValueError("資料清理後沒有可用資料。")
    if (out["load_kw"] < 0).any():
        raise ValueError("load_kw 不可小於 0。")
    if (out["solar_profile"] < 0).any():
        raise ValueError("solar_profile 不可小於 0。")
    if out["load_kw"].sum() <= 0:
        raise ValueError("全年總用電量必須大於 0。")
    if out["solar_profile"].sum() <= 0:
        raise ValueError("全年 solar_profile 總和必須大於 0。")
    return out


def _build_solver(solver_name: str, time_limit_seconds: Optional[int]) -> pulp.LpSolver:
    solver_name = solver_name.upper().strip()
    if solver_name == "CBC":
        return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)
    if solver_name == "HIGHS":
        return pulp.HiGHS(msg=False, timeLimit=time_limit_seconds)
    raise ValueError("solver_name 只支援 'CBC' 或 'HiGHS'。")


def solve_re100_lp(
    df: pd.DataFrame,
    inputs: ModelInputs,
    goal_mode: str,
    onsite_clean_target: float = 0.70,
    hourly_matching_target: float = 1.00,
    solver_name: str = "CBC",
    time_limit_seconds: Optional[int] = 120,
) -> Dict:
    df = validate_timeseries(df)

    goal_mode = goal_mode.strip().lower()
    valid_modes = {
        "annual_re100",
        "annual_re100_self_consumption",
        "hourly_matching",
    }
    if goal_mode not in valid_modes:
        raise ValueError(f"goal_mode 必須為：{', '.join(sorted(valid_modes))}")
    if not 0 <= onsite_clean_target <= 1:
        raise ValueError("onsite_clean_target 必須介於 0 與 1。")
    if not 0 <= hourly_matching_target <= 1:
        raise ValueError("hourly_matching_target 必須介於 0 與 1。")
    if not 0 < inputs.round_trip_efficiency <= 1:
        raise ValueError("round_trip_efficiency 必須介於 0 與 1。")
    if not 0 <= inputs.min_soc_ratio < inputs.max_soc_ratio <= 1:
        raise ValueError("SOC 比例設定不合理。")
    if not inputs.min_soc_ratio <= inputs.initial_soc_ratio <= inputs.max_soc_ratio:
        raise ValueError("initial_soc_ratio 必須位於 min_soc_ratio 與 max_soc_ratio 之間。")

    n = len(df)
    hours = list(range(n))
    load = {t: float(df.loc[t, "load_kw"]) for t in hours}
    solar_profile = {t: float(df.loc[t, "solar_profile"]) for t in hours}
    total_load = float(df["load_kw"].sum())

    eta_c = float(np.sqrt(inputs.round_trip_efficiency))
    eta_d = float(np.sqrt(inputs.round_trip_efficiency))
    pv_crf = annuity_factor(inputs.discount_rate, inputs.pv_life_years)
    batt_crf = annuity_factor(inputs.discount_rate, inputs.battery_life_years)

    prob = pulp.LpProblem("RE100_PV_Battery_Optimization", pulp.LpMinimize)

    pv_capacity_kw = pulp.LpVariable("pv_capacity_kw", 0, float(inputs.max_pv_kw))
    battery_power_kw = pulp.LpVariable("battery_power_kw", 0, float(inputs.max_battery_power_kw))
    battery_energy_kwh = pulp.LpVariable("battery_energy_kwh", 0, float(inputs.max_battery_energy_kwh))

    pv_to_load = {t: pulp.LpVariable(f"pv_to_load_{t}", lowBound=0) for t in hours}
    pv_to_battery = {t: pulp.LpVariable(f"pv_to_battery_{t}", lowBound=0) for t in hours}
    battery_discharge = {t: pulp.LpVariable(f"battery_discharge_{t}", lowBound=0) for t in hours}
    grid_purchase = {t: pulp.LpVariable(f"grid_purchase_{t}", lowBound=0) for t in hours}
    curtailment = {t: pulp.LpVariable(f"curtailment_{t}", lowBound=0) for t in hours}
    soc = {t: pulp.LpVariable(f"soc_{t}", lowBound=0) for t in hours}

    certificate_purchase_kwh = pulp.LpVariable(
        "certificate_purchase_kwh", lowBound=0, upBound=total_load
    )

    for t in hours:
        pv_generation_t = solar_profile[t] * pv_capacity_kw
        prob += pv_generation_t == pv_to_load[t] + pv_to_battery[t] + curtailment[t]
        prob += load[t] == pv_to_load[t] + battery_discharge[t] + grid_purchase[t]
        prob += pv_to_battery[t] <= battery_power_kw
        prob += battery_discharge[t] <= battery_power_kw
        prob += soc[t] >= inputs.min_soc_ratio * battery_energy_kwh
        prob += soc[t] <= inputs.max_soc_ratio * battery_energy_kwh

        if t == 0:
            prob += (
                soc[t]
                == inputs.initial_soc_ratio * battery_energy_kwh
                + eta_c * pv_to_battery[t]
                - battery_discharge[t] / eta_d
            )
        else:
            prob += (
                soc[t]
                == soc[t - 1]
                + eta_c * pv_to_battery[t]
                - battery_discharge[t] / eta_d
            )

        if goal_mode == "hourly_matching":
            prob += (
                pv_to_load[t] + battery_discharge[t]
                >= hourly_matching_target * load[t]
            )

    prob += soc[hours[-1]] == inputs.initial_soc_ratio * battery_energy_kwh

    total_pv_generation = pulp.lpSum(solar_profile[t] * pv_capacity_kw for t in hours)
    total_onsite_clean_supply = pulp.lpSum(
        pv_to_load[t] + battery_discharge[t] for t in hours
    )

    if goal_mode in {"annual_re100", "annual_re100_self_consumption"}:
        prob += total_pv_generation + certificate_purchase_kwh >= total_load
    else:
        prob += certificate_purchase_kwh == 0

    if goal_mode == "annual_re100_self_consumption":
        prob += total_onsite_clean_supply >= onsite_clean_target * total_load

    pv_annual_cost = (
        pv_capacity_kw * inputs.pv_capex_per_kw * pv_crf
        + pv_capacity_kw * inputs.pv_om_per_kw_year
    )
    battery_capex = (
        battery_power_kw * inputs.battery_power_capex_per_kw
        + battery_energy_kwh * inputs.battery_energy_capex_per_kwh
    )
    battery_annual_cost = battery_capex * batt_crf + battery_capex * inputs.battery_om_ratio
    grid_annual_cost = pulp.lpSum(grid_purchase[t] for t in hours) * inputs.grid_price_per_kwh
    certificate_annual_cost = certificate_purchase_kwh * inputs.certificate_price_per_kwh

    prob += pv_annual_cost + battery_annual_cost + grid_annual_cost + certificate_annual_cost

    solver = _build_solver(solver_name, time_limit_seconds)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(
            f"模型未取得最佳解，求解狀態：{status}。"
            "可提高求解時間、放寬目標或提高容量上限。"
        )

    pv_kw_value = float(pv_capacity_kw.value() or 0.0)
    batt_power_value = float(battery_power_kw.value() or 0.0)
    batt_energy_value = float(battery_energy_kwh.value() or 0.0)
    cert_value = float(certificate_purchase_kwh.value() or 0.0)

    rows = []
    for t in hours:
        pv_gen = solar_profile[t] * pv_kw_value
        row = {
            "timestamp": df.loc[t, "timestamp"],
            "load_kw": load[t],
            "solar_profile": solar_profile[t],
            "pv_generation": pv_gen,
            "pv_to_load": float(pv_to_load[t].value() or 0.0),
            "battery_charge": float(pv_to_battery[t].value() or 0.0),
            "battery_discharge": float(battery_discharge[t].value() or 0.0),
            "grid_purchase": float(grid_purchase[t].value() or 0.0),
            "curtailment": float(curtailment[t].value() or 0.0),
            "soc_kwh": float(soc[t].value() or 0.0),
        }
        row["clean_supply"] = row["pv_to_load"] + row["battery_discharge"]
        row["hourly_clean_ratio"] = (
            min(row["clean_supply"] / row["load_kw"], 1.0)
            if row["load_kw"] > 0
            else 1.0
        )
        rows.append(row)

    timeseries = pd.DataFrame(rows)
    total_pv_value = float(timeseries["pv_generation"].sum())
    total_clean_value = float(timeseries["clean_supply"].sum())
    total_grid_value = float(timeseries["grid_purchase"].sum())
    total_curtail_value = float(timeseries["curtailment"].sum())

    annual_re_ratio = min((total_pv_value + cert_value) / total_load, 1.0)
    onsite_clean_ratio = total_clean_value / total_load
    hourly_100_ratio = float((timeseries["hourly_clean_ratio"] >= 0.999999).mean())
    self_consumption_ratio = (
        (total_pv_value - total_curtail_value) / total_pv_value
        if total_pv_value > 0 else 0.0
    )

    pv_annual_cost_value = (
        pv_kw_value * inputs.pv_capex_per_kw * pv_crf
        + pv_kw_value * inputs.pv_om_per_kw_year
    )
    battery_capex_value = (
        batt_power_value * inputs.battery_power_capex_per_kw
        + batt_energy_value * inputs.battery_energy_capex_per_kwh
    )
    battery_annual_cost_value = (
        battery_capex_value * batt_crf
        + battery_capex_value * inputs.battery_om_ratio
    )
    grid_cost_value = total_grid_value * inputs.grid_price_per_kwh
    cert_cost_value = cert_value * inputs.certificate_price_per_kwh
    total_cost_value = (
        pv_annual_cost_value + battery_annual_cost_value
        + grid_cost_value + cert_cost_value
    )

    return {
        "goal_mode": goal_mode,
        "status": status,
        "objective_value": float(pulp.value(prob.objective) or 0.0),
        "pv_capacity_kw": pv_kw_value,
        "battery_power_kw": batt_power_value,
        "battery_energy_kwh": batt_energy_value,
        "battery_duration_h": batt_energy_value / batt_power_value if batt_power_value > 0 else 0.0,
        "annual_re_ratio": annual_re_ratio,
        "onsite_clean_ratio": onsite_clean_ratio,
        "hourly_100_ratio": hourly_100_ratio,
        "self_consumption_ratio": self_consumption_ratio,
        "total_load_kwh": total_load,
        "total_pv_generation_kwh": total_pv_value,
        "total_clean_supply_kwh": total_clean_value,
        "total_grid_purchase_kwh": total_grid_value,
        "total_curtailment_kwh": total_curtail_value,
        "external_certificate_kwh": cert_value,
        "pv_annual_cost": pv_annual_cost_value,
        "battery_annual_cost": battery_annual_cost_value,
        "grid_annual_cost": grid_cost_value,
        "certificate_annual_cost": cert_cost_value,
        "total_annual_cost": total_cost_value,
        "avg_cost_per_kwh": total_cost_value / total_load,
        "timeseries": timeseries,
    }
