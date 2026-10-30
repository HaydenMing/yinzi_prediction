"""将候选、选中模型、持价基线及逐日价格整理到固定表格。"""
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import save_json, read_json, write_csv, metric_values
from experiments import LABELS
from training import folder_for


def key(r):
    return r["material"], r["model"], r["recipe_id"], r["seed"]


def baseline_for(data):
    a = data["arrays"]
    split = data["report"]["splits"]["test"]
    s = slice(split["start"], split["end"])
    return metric_values(a["actual"][s], a["baseline"][s], a["baseline"][s])


def columns(material, model, recipe):
    early = recipe["order"] == "early"
    return {
        "物料": material, "框架": LABELS[model], "框架代码": model, "方案": recipe["id"],
        "融合顺序": "先融合再提取" if early else "先提取再融合",
        "价格编码器": "线性基础表示" if early else recipe["price"],
        "新闻原文编码器": "仅基础Embedding" if early else recipe["news"],
        "因子原文编码器": "仅基础Embedding" if early else recipe["factor"],
        "共享特征编码器": recipe["joint"] if early else None,
        "新闻时序编码器": None if early else recipe.get("news_temporal", "gru"),
        "因子时序编码器": None if early else recipe.get("factor_temporal", "gru"),
        "共享时序编码器": recipe["price"] if early else None,
        "配对组": recipe["pair_id"],
    }


def aggregate_metrics(runs, field):
    metric_keys = list(dict.fromkeys(k for r in runs for k in r[field]))
    values = {}
    for k in metric_keys:
        a = [r[field][k] for r in runs if r[field][k] is not None]
        values[k] = float(np.mean(a)) if a else None
    return values


def add_metrics(row, prefix, metrics):
    labels = {"mae": "MAE", "rmse": "RMSE", "mape_percent": "MAPE(%)", "wape_percent": "WAPE(%)",
              "direction_accuracy": "方向准确率(0至1)", "flat_count": "持平日数", "flat_mae": "持平日MAE",
              "change_count": "变动日数", "change_mae": "变动日MAE",
              "up_count": "上涨日数", "up_mae": "上涨日MAE", "down_count": "下跌日数", "down_mae": "下跌日MAE"}
    for k, label in labels.items():
        row[prefix + label] = metrics.get(k) if metrics else None


def write_reports(output, datasets, records, groups, winners, tests):
    output = Path(output)
    baselines = {m: baseline_for(d) for m, d in datasets.items()}
    test_map = {key(r): r for r in tests}
    purposes = defaultdict(list)
    for winner in winners:
        gkey = (winner["material"], winner["model"], winner["recipe_id"])
        purposes[gkey].append(winner["selection_scope"])
    grouped_runs = defaultdict(list)
    for r in records:
        grouped_runs[(r["material"], r["model"], r["recipe_id"])].append(r)
    rows, summaries = [], []
    for material, baseline in baselines.items():
        row = {"行类型": "持价基线", "物料": material, "框架": "持价基线",
               "测试状态": "已完成", "说明": "预测下一报价等于上一报价；非训练模型"}
        add_metrics(row, "验证", None)
        add_metrics(row, "测试", baseline)
        rows.append(row)
    for g in groups:
        gkey = (g["material"], g["model"], g["recipe_id"])
        runs = grouped_runs[gkey]
        truns = [test_map[key(r)] for r in runs if key(r) in test_map]
        row = dict(columns(g["material"], g["model"], g["recipe"]), 行类型="三种子汇总" if len(runs) == 3 else "种子汇总",
                   种子=None, 种子数=len(runs), 参数量=g["parameters"],
                   验证选择用途=";".join(purposes[gkey]), 测试状态="已完成" if len(truns) == len(runs) else "未测试或尚未完成")
        add_metrics(row, "验证", aggregate_metrics(runs, "validation"))
        add_metrics(row, "测试", aggregate_metrics(truns, "test") if len(truns) == len(runs) else None)
        row["验证MAE种子标准差"] = g["validation_mae_std"]
        for metric in ("mae", "rmse"):
            row["测试" + metric.upper() + "种子标准差"] = float(np.std([r["test"][metric] for r in truns])) if len(truns) == len(runs) else None
        baseline = baselines[g["material"]]["mae"]
        row["测试持价基线MAE"] = baseline
        row["测试MAE相对基线改善(%)"] = (baseline - row["测试MAE"]) / baseline * 100 if row["测试MAE"] is not None else None
        row["说明"] = "种子标准差使用ddof=0；测试不参与选择；early三视图已混合，不能解释为独立模态"
        summaries.append(row)
        rows.append(row)
    for r in records:
        tested = test_map.get(key(r))
        row = dict(columns(r["material"], r["model"], r["recipe"]), 行类型="逐种子明细",
                   种子=r["seed"], 种子数=1, 参数量=r["parameters"], 最佳轮次=r["best_epoch"],
                   实际训练轮数=r["epochs_run"], 训练秒数=r["train_seconds"],
                   验证选择用途=";".join(purposes[(r["material"], r["model"], r["recipe_id"])]),
                   测试状态="已完成" if tested else "未选中或尚未测试")
        add_metrics(row, "验证", r["validation"])
        add_metrics(row, "测试", tested["test"] if tested else None)
        baseline = baselines[r["material"]]["mae"]
        row["测试持价基线MAE"] = baseline
        row["测试MAE相对基线改善(%)"] = (baseline - tested["test"]["mae"]) / baseline * 100 if tested else None
        row["结果目录"] = str(folder_for(output, r["material"], r["model"], r["recipe_id"], r["seed"]))
        rows.append(row)
    write_csv(output / "所有实验结果.csv", rows)
    write_csv(output / "方案均值对比.csv", summaries)
    pairs = defaultdict(dict)
    for g in groups:
        if g["recipe"]["paired"]:
            pairs[(g["material"], g["model"], g["recipe"]["pair_id"])][g["recipe"]["order"]] = g
    comparisons = []
    for (material, model, pair_id), pair in sorted(pairs.items()):
        if set(pair) != {"late", "early"}:
            continue
        late, early = pair["late"], pair["early"]
        comparisons.append({
            "物料": material, "框架": LABELS[model], "配对配置": pair_id,
            "先提取后融合_验证MAE": late["validation_mae_mean"],
            "先融合后提取_验证MAE": early["validation_mae_mean"],
            "early减late_验证MAE": early["validation_mae_mean"] - late["validation_mae_mean"],
            "late参数量": late["parameters"], "early参数量": early["parameters"],
            "说明": "固定文字候选和价格/共享时序类型；结构与参数量仍不同，不能只归因于顺序",
        })
    write_csv(output / "先后融合配对比较.csv", comparisons)
    all_predictions, average_predictions = [], []
    tested_groups = defaultdict(list)
    for tested in tests:
        folder = folder_for(output, tested["material"], tested["model"], tested["recipe_id"], tested["seed"])
        predictions = read_json(folder / "selected_test_predictions.json")
        identity = {k: tested[k] for k in ("material", "model", "recipe_id", "seed")}
        all_predictions.extend(dict(identity, **p) for p in predictions)
        tested_groups[(tested["material"], tested["model"], tested["recipe_id"])].append(predictions)
    for (material, model, recipe_id), lists in tested_groups.items():
        dates = [r["date"] for r in lists[0]]
        if any([r["date"] for r in seq] != dates for seq in lists):
            raise ValueError("种子预测日期不一致")
        for i, day in enumerate(dates):
            average_predictions.append({
                "物料": material, "框架": LABELS[model], "方案": recipe_id, "日期": day,
                "真实价格": lists[0][i]["actual_price"], "上一报价": lists[0][i]["previous_price"],
                "种子预测价格均值": float(np.mean([seq[i]["predicted_price"] for seq in lists])),
                "种子预测价格标准差": float(np.std([seq[i]["predicted_price"] for seq in lists])),
                "种子数": len(lists), "说明": "这里是种子预测的平均；其误差不等同于各种子MAE的平均",
            })
    write_csv(output / "全部选中方案逐日预测.csv", all_predictions)
    write_csv(output / "选中方案种子平均预测.csv", average_predictions)
    save_json(output / "selected_test_summary.json", {"groups": summaries, "baselines": baselines,
              "selection_rule": "只按种子平均验证MAE选择", "winners": winners})
    lines = [
        "# 本轮预测结果", "", "目标：下一报价日价格；模型学习标准化价差，再加回上一报价。因子是文字输入，不是涨跌幅标签。",
        "逐日价格、价差、百分比和派生方向见各选中方案的测试逐日报价预测.csv。",
        "所有实验结果.csv同时含均值、种子明细和持价基线，使用行类型筛选，不能把汇总行和明细行再次混合求平均。", "",
        "配对顺序比较只用各自同样数量的配对候选；native_late使用额外单模态替换候选，独立报告。",
        "本轮沿用已查看过的历史测试区间，是诊断比较。参数量不同、因子与新闻同源、语义清理不完全，均需在汇报说明。", "",
        "|物料|框架|选择用途|验证选中方案|候选数|验证MAE|测试MAE|持价MAE|",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    summary_map = {(r["物料"], r["框架代码"], r["方案"]): r for r in summaries}
    for winner in winners:
        s = summary_map[(winner["material"], winner["model"], winner["recipe_id"])]
        test_text = "未测试" if s["测试MAE"] is None else f"{s['测试MAE']:.3f}"
        lines.append(f"|{winner['material']}|{LABELS[winner['model']]}|{winner['selection_scope']}|"
                     f"{winner['recipe_id']}|{winner['candidates']}|{winner['validation_mae_mean']:.3f}|"
                     f"{test_text}|{baselines[winner['material']]['mae']:.3f}|")
    lines += ["", "paired_late/paired_early：各6个配对候选中的验证胜出者（完整默认矩阵）。",
              "native_late：全部10个独立编码候选中的验证胜出者。缩小矩阵试运行时，以表中实际候选数为准。",
              "筛选后不成对的方案列为unpaired_early/unpaired_late，不能用于同预算顺序比较。",
              "方向是预测价差的符号；没有分类概率，也不能把因子解释成模型承诺的价格涨跌程度。",
              "原价格表计量单位未确认，全部指标沿用原尺度。"]
    (output / "导师汇报用结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

