#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6 溯源分析统计层（零重跑）：算子 → 失效模式 → 故障类型映射。

聚合已有零重跑数据，生成 traceability_map.json + traceability_summary.md：
  - 算子语义（OPERATOR_META，来自 03-operators-spec + mutation/operators 代码）
  - Phase 4 训练表现（PHASE4_TRAINING，来自 plan-06 §5 Phase 4 表）
  - 评估失效模式（failure_mode_coverage.csv，块1）
  - Fisher 显著性 / kill（kill_matrix_multi.csv）
  - TFR（tfr_candidate_fail_summary.csv）

核心归因区分（perturbation_phase）：
  - eval-time：State/Action/PV 算子评估期实时扰动观测/动作/策略输出，直接致失效
  - train-time：Reward 算子评估 deterministic、reward 不改轨迹，失效纯来自训练期奖励污染

故障类型（fault_type/fault_mechanism）是基于"算子语义 + 主导失效模式数据"的归因，
每条都可由 dominant_failure + perturbation 追溯，不武断。

状态层机理证据（失败 case 状态轨迹）由方案B-块2 补充，待 CARLA 恢复。
"""
import argparse
import csv
import json
import os
from collections import Counter

SEVERE = ["collision", "lane_deviation", "low_speed_timeout", "over_speed", "step_failure", "carla_crash"]
SUITES = ("rainy", "foggy")

# 算子语义 + 故障归因元数据（12 算子）
OPERATOR_META = {
    "StRepP": dict(category="State", hook="state_out", operation="Repeat", timing="Periodical",
                   perturbation="周期性用上一步状态替换当前状态（感知滞后）",
                   perturbation_phase="eval-time",
                   fault_type="感知滞后型",
                   fault_mechanism="状态周期性重复→感知信息滞后→响应迟缓，多表现为完不成(max_steps)、偶发碰撞"),
    "StDistP": dict(category="State", hook="state_out", operation="Disturbance", timing="Periodical",
                    perturbation="周期性向状态(nav+image)注入高斯噪声",
                    perturbation_phase="eval-time",
                    fault_type="感知噪声型（全面退化）",
                    fault_mechanism="状态噪声破坏障碍/边界/速度判断→最广泛失效(碰撞+压线+超速)"),
    "StDisoP": dict(category="State", hook="state_out", operation="Disorder", timing="Periodical",
                    perturbation="周期性从历史缓存随机抽一个状态替换当前状态",
                    perturbation_phase="eval-time",
                    fault_type="状态时序错乱型",
                    fault_mechanism="用历史错误状态替换当前→定位/决策错乱→碰撞+压线，并偶发低速卡死(唯一触发 low_speed_timeout)"),
    "StFuzS": dict(category="State", hook="state_out", operation="Fuzz", timing="Sustained",
                   perturbation="持续量化降低状态精度（nav 1 位小数、image 降 bit）",
                   perturbation_phase="eval-time",
                   fault_type="感知精度损失型",
                   fault_mechanism="持续量化丢失细节→边界/障碍判断失准→以碰撞为主"),
    "ReDistP": dict(category="Reward", hook="reward_out", operation="Disturbance", timing="Periodical",
                    perturbation="周期性向 reward 注入高斯噪声（训练期）",
                    perturbation_phase="train-time",
                    fault_type="奖励噪声型（训练期污染）",
                    fault_mechanism="评估 deterministic、reward 不改轨迹；失效来自训练期奖励噪声干扰速度相关奖励→策略学出超速倾向"),
    "ReDisoP": dict(category="Reward", hook="reward_out", operation="Disorder", timing="Periodical",
                    perturbation="周期性随机翻转/缩放 reward（训练期）",
                    perturbation_phase="train-time",
                    fault_type="奖励错乱型（训练崩坏）",
                    fault_mechanism="奖励随机翻转使策略学歪，训练未达阈值(best 1284, over_speed)；判 suspicious，不入主分母"),
    "ReRepP": dict(category="Reward", hook="reward_out", operation="Repeat", timing="Periodical",
                   perturbation="周期性重复上一步 reward（训练期）",
                   perturbation_phase="train-time",
                   fault_type="奖励信号失真型（训练期污染）",
                   fault_mechanism="评估 reward 不改轨迹；失效来自训练期奖励重复→车道保持奖励信号失真→压线为主"),
    "AcDisoR": dict(category="Action", hook="action_in", operation="Disorder", timing="Random",
                    perturbation="随机以新随机动作替换决策动作",
                    perturbation_phase="eval-time",
                    fault_type="决策错误型（训练崩坏）",
                    fault_mechanism="随机决策错误过强使训练未达阈值(best 825, lane_deviation)；判 suspicious，不入主分母"),
    "AcFuzS": dict(category="Action", hook="action_in", operation="Fuzz", timing="Sustained",
                   perturbation="持续量化降低动作精度",
                   perturbation_phase="eval-time",
                   fault_type="控制精度损失型",
                   fault_mechanism="动作量化使转向/油门粗糙→控制犹豫→大量完不成(max_steps)、偶发碰撞"),
    "AcRepR": dict(category="Action", hook="action_in", operation="Repeat", timing="Random",
                   perturbation="随机重复上一步动作",
                   perturbation_phase="eval-time",
                   fault_type="控制滞后型",
                   fault_mechanism="动作重复→控制指令滞后→避障响应不及→碰撞"),
    "PVDistR": dict(category="PolicyValue", hook="pv_out", operation="Disturbance", timing="Random",
                    perturbation="随机向策略输出 action_mean 注入噪声",
                    perturbation_phase="eval-time",
                    fault_type="策略输出扰动型",
                    fault_mechanism="策略输出加噪使动作偏移→超速为主，伴随碰撞"),
    "ESRemS": dict(category="ExplorationStrategy", hook="es_sample", operation="Remove", timing="Sustained",
                   perturbation="持续用 mean 替代采样，移除探索",
                   perturbation_phase="train-time",
                   fault_type="探索缺失型（训练退化）",
                   fault_mechanism="移除探索使 PPO 退化，训不出(best 2.03, early_stop_500k)；判 trivial_fail，不入主分母"),
}

# Phase 4 训练表现（plan-06 §5）
PHASE4_TRAINING = {
    "StDisoP": dict(episodes=901, last_timestep=1002898, stop_reason="route_completed", best_rolling50=2084.33, reached_threshold=True),
    "StRepP": dict(episodes=738, last_timestep=1000516, stop_reason="route_completed", best_rolling50=1895.14, reached_threshold=True),
    "StDistP": dict(episodes=896, last_timestep=1002330, stop_reason="route_completed", best_rolling50=1908.36, reached_threshold=True),
    "StFuzS": dict(episodes=865, last_timestep=1000602, stop_reason="route_completed", best_rolling50=2004.75, reached_threshold=True),
    "AcRepR": dict(episodes=775, last_timestep=1000422, stop_reason="route_completed", best_rolling50=1878.54, reached_threshold=True),
    "AcFuzS": dict(episodes=1049, last_timestep=1001640, stop_reason="route_completed", best_rolling50=2029.07, reached_threshold=True),
    "AcDisoR": dict(episodes=1703, last_timestep=1000689, stop_reason="lane_deviation", best_rolling50=824.61, reached_threshold=False),
    "ReRepP": dict(episodes=864, last_timestep=1002300, stop_reason="route_completed", best_rolling50=1938.46, reached_threshold=True),
    "ReDistP": dict(episodes=746, last_timestep=1002628, stop_reason="route_completed", best_rolling50=1883.65, reached_threshold=True),
    "ReDisoP": dict(episodes=1031, last_timestep=1000053, stop_reason="over_speed", best_rolling50=1283.94, reached_threshold=False),
    "PVDistR": dict(episodes=910, last_timestep=1000194, stop_reason="route_completed", best_rolling50=1922.10, reached_threshold=True),
    "ESRemS": dict(episodes=7793, last_timestep=500796, stop_reason="low_speed_timeout", best_rolling50=2.03, reached_threshold=False),
}

ADMISSION = {
    **{op: "pass" for op in ["StDisoP", "StRepP", "StDistP", "StFuzS", "AcRepR", "AcFuzS", "ReRepP", "ReDistP", "PVDistR"]},
    "AcDisoR": "suspicious", "ReDisoP": "suspicious", "ESRemS": "trivial_fail",
}


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage-csv", required=True, help="failure_mode_coverage.csv (块1)")
    ap.add_argument("--kill-csv", required=True, help="kill_matrix_multi.csv")
    ap.add_argument("--tfr-csv", required=True, help="tfr_candidate_fail_summary.csv")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 评估失效模式（聚合 rainy+foggy）
    cov = read_csv(args.coverage_csv)
    eval_fail = {}  # op -> {suite -> {reason:count}}, and aggregated
    for r in cov:
        cand = r["candidate"]
        op = cand.replace("mutant_", "") if cand.startswith("mutant_") else cand
        eval_fail.setdefault(op, {})
        eval_fail[op][r["suite"]] = {reason: int(r.get(reason, 0) or 0) for reason in
                                     ["route_completed", "max_steps_reached"] + SEVERE if r.get(reason)}

    # Fisher / kill
    kill = read_csv(args.kill_csv)
    kill_by = {}  # op -> {suite -> {fisher_p, kill_fisher, mutant_TFR}}
    for r in kill:
        op = r["mutant"].replace("mutant_", "")
        kill_by.setdefault(op, {})[r["suite"]] = dict(
            fisher_p=float(r["fisher_p"]), kill_fisher=int(r["kill_fisher"]),
            mutant_TFR=float(r["mutant_TFR"]), baseline_TFR=float(r["baseline_TFR"]))

    # TFR summary（拿 done_reason 分解备用，已在 coverage 里有）
    tfr = read_csv(args.tfr_csv)
    tfr_by = {}
    for r in tfr:
        cand = r["candidate"]
        op = cand.replace("mutant_", "") if cand.startswith("mutant_") else cand
        tfr_by.setdefault(op, {})[r["suite"]] = dict(TFR=float(r["TFR"]),
                                                      mean_progress=float(r["mean_progress_ratio"]),
                                                      mean_reward=float(r["mean_total_reward"]))

    def dominant_failure(op):
        agg = Counter()
        for suite in SUITES:
            for reason, c in eval_fail.get(op, {}).get(suite, {}).items():
                if reason in SEVERE:
                    agg[reason] += c
        return agg.most_common(1)[0][0] if agg else None, dict(agg)

    operators = {}
    failure_to_ops = {reason: [] for reason in SEVERE}
    for op, meta in OPERATOR_META.items():
        dom, severe_counts = dominant_failure(op)
        entry = dict(meta)
        entry["admission_status"] = ADMISSION[op]
        entry["training"] = PHASE4_TRAINING[op]
        if op in eval_fail:
            entry["evaluation"] = {
                "in_main_eval": True,
                "severe_failure_counts_rainy_foggy": severe_counts,
                "dominant_failure_mode": dom,
                "per_suite": {s: dict(
                    TFR=tfr_by.get(op, {}).get(s, {}).get("TFR"),
                    fisher_p=kill_by.get(op, {}).get(s, {}).get("fisher_p"),
                    killed_fisher=kill_by.get(op, {}).get(s, {}).get("kill_fisher"),
                    mean_progress=tfr_by.get(op, {}).get(s, {}).get("mean_progress"),
                    mean_reward=tfr_by.get(op, {}).get(s, {}).get("mean_reward"),
                ) for s in SUITES},
            }
            for reason, c in severe_counts.items():
                if reason in failure_to_ops and c > 0:
                    failure_to_ops[reason].append(dict(operator=op, count=c))
        else:
            entry["evaluation"] = {"in_main_eval": False,
                                   "note": "未进入主评估（admission=%s），溯源仅基于训练表现" % ADMISSION[op]}
        operators[op] = entry

    for reason in failure_to_ops:
        failure_to_ops[reason].sort(key=lambda x: -x["count"])

    out = dict(
        meta=dict(
            description="Phase 6 溯源映射统计层（零重跑）：算子→失效模式→故障类型",
            main_denominator="9 个 Phase 4.5 pass mutant",
            excluded=["AcDisoR(suspicious)", "ReDisoP(suspicious)", "ESRemS(trivial_fail)"],
            failure_modes_observed=sorted({r for op in operators
                                           for r in operators[op].get("evaluation", {}).get("severe_failure_counts_rainy_foggy", {})}),
            state_layer_evidence="失败 case 状态轨迹（方案B-块2）待 CARLA 恢复后补充",
        ),
        operators=operators,
        failure_mode_to_operators=failure_to_ops,
        fault_type_taxonomy={
            "eval-time perturbation": "State/Action/PV：评估期实时扰动观测/动作/策略输出，直接致失效",
            "train-time poisoning": "Reward/ES：评估 deterministic 不受扰动，失效来自训练期信号污染导致的策略退化",
        },
    )

    json_path = os.path.join(args.output_dir, "traceability_map.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 人类可读汇总
    lines = ["# Phase 6 溯源映射汇总（统计层，零重跑）", "",
             "主分母 9 个 pass mutant；AcDisoR/ReDisoP(suspicious)、ESRemS(trivial_fail) 单独列、不入主分母。", "",
             "## 算子 → 失效模式 → 故障类型", "",
             "| 算子 | 类别 | 扰动相位 | 主导失效 | rainy/foggy TFR | Fisher kill | 故障类型 |",
             "|---|---|---|---|---|---|---|"]
    for op, e in operators.items():
        ev = e.get("evaluation", {})
        if ev.get("in_main_eval"):
            tfr_r = ev["per_suite"]["rainy"]["TFR"]
            tfr_f = ev["per_suite"]["foggy"]["TFR"]
            kr = ev["per_suite"]["rainy"]["killed_fisher"]
            kf = ev["per_suite"]["foggy"]["killed_fisher"]
            killstr = "%s/%s" % ("✓" if kr else "✗", "✓" if kf else "✗")
            lines.append("| %s | %s | %s | %s | %.2f/%.2f | %s | %s |" %
                         (op, e["category"], e["perturbation_phase"], ev["dominant_failure_mode"],
                          tfr_r, tfr_f, killstr, e["fault_type"]))
        else:
            lines.append("| %s | %s | %s | (未入主评估) | - | - | %s |" %
                         (op, e["category"], e["perturbation_phase"], e["fault_type"]))
    lines += ["", "## 失效模式 → 算子（反向索引，rainy+foggy 合计）", ""]
    for reason in SEVERE:
        ops = failure_to_ops.get(reason, [])
        if ops:
            lines.append("- **%s**: %s" % (reason, ", ".join("%s(%d)" % (o["operator"], o["count"]) for o in ops)))
    lines += ["", "## 故障机理（逐算子）", ""]
    for op, e in operators.items():
        lines.append("- **%s**（%s, %s）：%s" % (op, e["category"], e["fault_type"], e["fault_mechanism"]))
    lines += ["", "## 待补：状态层机理证据",
              "Reward 类(ReRepP/ReDistP)失效来自训练期奖励污染，评估期不改轨迹——此机理已可由 deterministic 评估论证；",
              "State/Action/PV 类的逐步状态异常轨迹由方案B-块2（失败 case 重放）补充，待 CARLA 恢复。"]

    md_path = os.path.join(args.output_dir, "traceability_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\njson=" + json_path)
    print("md=" + md_path)


if __name__ == "__main__":
    main()
