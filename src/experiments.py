"""实验配置、验证选择范围。CLI读取旁边的JSON。"""
from collections import defaultdict
import numpy as np

MODELS = ("ucmib_pns", "graph_gated", "hierarchical", "parallel", "lcmf")
LABELS = {"ucmib_pns": "① UCMIB-PNS", "graph_gated": "② GAT门控",
          "hierarchical": "③ 层级融合", "parallel": "④ 并行注意力", "lcmf": "⑤ LCMF"}


def validate_recipes(recipes):
    ids = set()
    pairs = defaultdict(dict)
    if not recipes:
        raise ValueError("实验配置不能为空")
    for r in recipes:
        if r["id"] in ids or not r["id"].replace("_", "").isascii() or not r["id"].replace("_", "").isalnum():
            raise ValueError("方案ID重复或不是ASCII字母数字下划线")
        ids.add(r["id"])
        if r["order"] not in {"late", "early"} or r["price"] not in {"gru", "tcn"}:
            raise ValueError("未知融合顺序或时序编码器")
        if r["order"] == "late":
            if r["news"] not in {"mean", "textcnn", "bigru"} or r["factor"] not in {"mean", "textcnn", "bigru"}:
                raise ValueError("未知原文编码器")
            if any(r.get(k, "gru") not in {"gru", "tcn"} for k in ("news_temporal", "factor_temporal")):
                raise ValueError("未知新闻/因子时序编码器")
        elif r["joint"] not in {"mean", "textcnn", "bigru"}:
            raise ValueError("未知共享编码器")
        if r["paired"]:
            pair_id = r.get("pair_id")
            if not pair_id or r["order"] in pairs[pair_id]:
                raise ValueError("配对组标识为空或同一侧有重复候选")
            pairs[pair_id][r["order"]] = r
    for pair in pairs.values():
        if set(pair) == {"early", "late"}:
            a, b = pair["early"], pair["late"]
            if a["price"] != b["price"] or not a["joint"] == b["news"] == b["factor"]:
                raise ValueError("配对组必须保持文字候选和价格/共享时序类型一致")
    return {key: sorted(value) for key, value in pairs.items()}


def selection_scopes(recipes):
    sides = defaultdict(set)
    for r in recipes:
        if r["paired"]:
            sides[r["pair_id"]].add(r["order"])
    matched = {p for p, orders in sides.items() if orders == {"early", "late"}}
    result = {}
    for r in recipes:
        scopes = []
        if r["paired"]:
            prefix = "paired_" if r["pair_id"] in matched else "unpaired_"
            scopes.append(prefix + r["order"])
        elif r["order"] == "early":
            scopes.append("unpaired_early")
        if r["order"] == "late":
            scopes.append("native_late")
        result[r["id"]] = scopes
    return result


def group_validation(records, recipes, seeds):
    mapping = {r["id"]: r for r in recipes}
    groups = []
    by_key = defaultdict(list)
    for r in records:
        by_key[(r["material"], r["model"], r["recipe_id"])].append(r)
    for (material, model, recipe_id), runs in sorted(by_key.items()):
        if sorted(r["seed"] for r in runs) != sorted(seeds):
            continue
        values = [r["validation"]["mae"] for r in runs]
        groups.append({
            "material": material, "model": model, "recipe_id": recipe_id,
            "recipe": mapping[recipe_id], "seed_count": len(runs),
            "validation_mae_mean": float(np.mean(values)), "validation_mae_std": float(np.std(values)),
            "parameters": runs[0]["parameters"],
        })
    return groups


def choose_winners(groups):
    scopes = defaultdict(list)
    by_experiment = defaultdict(list)
    for g in groups:
        by_experiment[(g["material"], g["model"])].append(g)
    for (material, model), candidates in by_experiment.items():
        membership = selection_scopes([g["recipe"] for g in candidates])
        for g in candidates:
            for scope in membership[g["recipe_id"]]:
                scopes[(material, model, scope)].append(g)
    winners = []
    for (material, model, scope), candidates in sorted(scopes.items()):
        chosen = min(candidates, key=lambda g: (g["validation_mae_mean"], g["recipe_id"]))
        winners.append(dict(chosen, selection_scope=scope, candidates=len(candidates)))
    unique = {}
    for r in winners:
        key = (r["material"], r["model"], r["recipe_id"])
        if key not in unique:
            unique[key] = dict(r, selection_scopes=[])
        unique[key]["selection_scopes"].append(r["selection_scope"])
    return winners, list(unique.values())
