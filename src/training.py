"""训练及选中模型测试。测试集不参与早停、编码器或框架选择。"""
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from common import save_json, read_json, write_csv, save_npz, file_hash, metric_values, prediction_rows
from model import MultimodalModel


def folder_for(output, material, model, recipe_id, seed):
    return Path(output) / material / recipe_id / f"{model}_seed{seed}"


def make_model(data, name, recipe, args):
    return MultimodalModel(name, recipe, data["snapshots"], len(data["vocab"]) + 3,
                           args.hidden, args.embedding, args.chunk_size, args.chunk_batch).to(args.device)


def write_price_csv(path, rows):
    labels = {
        "date": "目标报价日", "previous_price": "上一报价", "actual_price": "真实价格",
        "predicted_price": "预测价格", "actual_change": "实际价差", "predicted_change": "预测价差",
        "actual_change_percent": "实际变化百分比", "predicted_change_percent": "预测变化百分比",
        "actual_direction": "实际方向", "predicted_direction": "预测方向",
        "normalized_change_prediction": "模型原始输出_标准化价差", "absolute_error": "价格绝对误差",
    }
    write_csv(path, [{labels[k]: v for k, v in row.items()} for row in rows])


@torch.no_grad()
def predict(model, data, split_name, batch_size, collect_features=False):
    model.eval()
    arrays = data["arrays"]
    split = data["report"]["splits"][split_name]
    outputs, features = [], {}
    for a in range(split["start"], split["end"], batch_size):
        b = min(a + batch_size, split["end"])
        price = torch.from_numpy(arrays["p"][a:b].astype(np.float32, copy=True))
        sids = torch.from_numpy(arrays["snapshot_ids"][a:b].copy())
        device = next(model.parameters()).device
        out = model(price.to(device), sids.to(device), return_features=collect_features)
        outputs.append(out["prediction"].cpu().numpy())
        if collect_features:
            traced = dict(out["encoder_features"], fusion_features=out["fusion_features"])
            for key, value in traced.items():
                features.setdefault(key, []).append(value.detach().cpu().numpy())
    normalized = np.concatenate(outputs)
    if not np.isfinite(normalized).all():
        raise ValueError("预测输出存在非有限数")
    return normalized, {k: np.concatenate(v) for k, v in features.items()}


def rows_and_metrics(data, split_name, normalized):
    arrays, report = data["arrays"], data["report"]
    split = report["splits"][split_name]
    s = slice(split["start"], split["end"])
    rows = prediction_rows(arrays["dates"][s], arrays["actual"][s], arrays["baseline"][s],
                           normalized, report["price_difference_scale"])
    values = np.array([r["predicted_price"] for r in rows])
    metrics = metric_values(arrays["actual"][s], values, arrays["baseline"][s])
    return rows, metrics


def verify_predictions(data, split, rows, metrics):
    bounds = data["report"]["splits"][split]
    s = slice(bounds["start"], bounds["end"])
    if [r["date"] for r in rows] != data["arrays"]["dates"][s].tolist():
        raise ValueError("保存预测与固定日期不一致")
    normalized = np.array([r["normalized_change_prediction"] for r in rows])
    expected, recomputed = rows_and_metrics(data, split, normalized)
    for old, new in zip(rows, expected):
        for key in new:
            if isinstance(new[key], (float, int)):
                if not np.isclose(old[key], new[key], atol=1e-6, rtol=1e-8):
                    raise ValueError(f"保存价格字段不一致：{key}")
            elif old[key] != new[key]:
                raise ValueError(f"保存方向或日期字段不一致：{key}")
    for key, value in recomputed.items():
        if value is None:
            if metrics.get(key) is not None:
                raise ValueError("无对应涨跌日的指标应为空")
        elif not np.isclose(metrics[key], value, atol=1e-6, rtol=1e-8):
            raise ValueError("保存指标与逐日预测不一致")


def train_one(data, material, name, recipe, seed, output, args):
    folder = folder_for(output, material, name, recipe["id"], seed)
    marker = folder / "validation_metrics.json"
    identity = {"material": material, "model": name, "recipe_id": recipe["id"], "seed": seed}
    if marker.is_file():
        record = read_json(marker)
        if any(record[k] != v for k, v in identity.items()) or record["recipe"] != recipe:
            raise ValueError("已保存训练记录身份不一致")
        if file_hash(folder / "best.pt") != record["checkpoint_sha256"]:
            raise ValueError("模型权重已变动")
        verify_predictions(data, "validation", read_json(folder / "validation_predictions.json"), record["validation"])
        print(f"复用 {material} {name} {recipe['id']} seed={seed}", flush=True)
        return record
    folder.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(args.deterministic)
    model = make_model(data, name, recipe, args)
    architecture = {
        "hidden": args.hidden, "embedding": args.embedding, "chunk_size": args.chunk_size,
        "chunk_batch": args.chunk_batch, "news_temporal": recipe.get("news_temporal", "gru") if recipe["order"] == "late" else None,
        "factor_temporal": recipe.get("factor_temporal", "gru") if recipe["order"] == "late" else None,
        "price_encoder": recipe["price"] if recipe["order"] == "late" else "linear_to_embedding",
        "shared_temporal": recipe["price"] if recipe["order"] == "early" else None,
    }
    save_json(folder / "model_config.json", dict(identity, recipe=recipe, architecture=architecture,
              target="下一报价日标准化价差；最终还原为价格",
              feature_meaning="独立模态" if recipe["order"] == "late" else "混合视图，不能解释为独立模态"))
    arrays, report = data["arrays"], data["report"]
    train = report["splits"]["train"]
    s = slice(train["start"], train["end"])
    dataset = TensorDataset(
        torch.from_numpy(arrays["p"][s].astype(np.float32, copy=True)),
        torch.from_numpy(arrays["snapshot_ids"][s].copy()),
        torch.from_numpy(arrays["y"][s].astype(np.float32, copy=True)),
        torch.arange(train["start"], train["end"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, stale, best_epoch = float("inf"), 0, 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for price, sids, y, ids in loader:
            price, sids, y, ids = [x.to(args.device) for x in (price, sids, y, ids)]
            optimizer.zero_grad(set_to_none=True)
            out = model(price, sids)
            loss = model.loss(out, y, ids)
            if not torch.isfinite(loss):
                raise ValueError("训练loss不是有限数")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            model.enqueue(out, ids)
            losses.append(float(loss.detach()))
        validation, _ = predict(model, data, "validation", args.batch_size)
        _, metrics = rows_and_metrics(data, "validation", validation)
        score = metrics["mae"]
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation_mae": score})
        if score < best:
            best, stale, best_epoch = score, 0, epoch
            temp = folder / "best.pt.tmp"
            with temp.open("wb") as stream:
                torch.save({"state_dict": model.state_dict(), "model": name, "recipe": recipe,
                            "architecture": architecture, "epoch": epoch, "seed": seed}, stream)
            temp.replace(folder / "best.pt")
        else:
            stale += 1
        save_json(folder / "training.json", history)
        if epoch == 1 or epoch % 5 == 0:
            print(f"{material} {name} {recipe['id']} seed={seed} epoch={epoch} val_MAE={score:.3f}", flush=True)
        if stale >= args.patience:
            break
    with (folder / "best.pt").open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validation, _ = predict(model, data, "validation", args.batch_size)
    rows, metrics = rows_and_metrics(data, "validation", validation)
    save_json(folder / "validation_predictions.json", rows)
    write_price_csv(folder / "验证逐日报价预测.csv", rows)
    record = dict(identity, recipe=recipe, validation=metrics, best_epoch=best_epoch,
                  epochs_run=len(history), parameters=sum(p.numel() for p in model.parameters()),
                  train_seconds=time.monotonic() - started,
                  checkpoint_sha256=file_hash(folder / "best.pt"), test_used_for_training=False)
    # 最后保存完成标记，中断时不存在标记的单次训练重头开始。
    save_json(marker, record)
    return record


def evaluate_one(data, chosen, seed, output, args):
    material, name, recipe = chosen["material"], chosen["model"], chosen["recipe"]
    folder = folder_for(output, material, name, recipe["id"], seed)
    validation = read_json(folder / "validation_metrics.json")
    checkpoint_hash = file_hash(folder / "best.pt")
    if checkpoint_hash != validation["checkpoint_sha256"]:
        raise ValueError("验证选优后模型权重发生变化")
    marker = folder / "selected_test_metrics.json"
    if marker.is_file():
        record = read_json(marker)
        if record["checkpoint_sha256"] != checkpoint_hash or record["selection_scopes"] != chosen["selection_scopes"]:
            raise ValueError("测试缓存的权重或选择用途不一致")
        if any(record[k] != validation[k] for k in ("material", "model", "recipe_id", "seed")):
            raise ValueError("测试记录身份不一致")
        verify_predictions(data, "test", read_json(folder / "selected_test_predictions.json"), record["test"])
        if file_hash(folder / "features.npz") != record["feature_sha256"]:
            raise ValueError("保存特征已变动")
        return record
    model = make_model(data, name, recipe, args)
    with (folder / "best.pt").open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    normalized, features = predict(model, data, "test", args.batch_size, collect_features=True)
    rows, metrics = rows_and_metrics(data, "test", normalized)
    save_json(folder / "selected_test_predictions.json", rows)
    write_price_csv(folder / "测试逐日报价预测.csv", rows)
    split = data["report"]["splits"]["test"]
    dates = data["arrays"]["dates"][split["start"]:split["end"]]
    save_npz(folder / "features.npz", dates=dates, normalized_prediction=normalized, **features)
    record = {k: validation[k] for k in ("material", "model", "recipe_id", "seed", "parameters")}
    record.update(test=metrics, recipe=recipe, selection_scopes=chosen["selection_scopes"],
                  checkpoint_sha256=checkpoint_hash, feature_sha256=file_hash(folder / "features.npz"))
    save_json(marker, record)
    print(f"测试 {material} {name} {recipe['id']} seed={seed} MAE={metrics['mae']:.3f}", flush=True)
    return record






