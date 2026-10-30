"""文件保存、复用一致性、回归指标及可直接阅读的价格输出。"""
import csv
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


@contextmanager
def output_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def direction(change):
    if abs(float(change)) <= 1e-8:
        return "持平"
    return "上涨" if change > 0 else "下跌"


def prediction_rows(dates, actual, previous, normalized, scale):
    actual, previous, normalized = [np.asarray(x, dtype=np.float64) for x in (actual, previous, normalized)]
    if not (len(dates) == len(actual) == len(previous) == len(normalized)):
        raise ValueError("价格输出长度不一致")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("目标尺度非法")
    if any(not np.isfinite(x).all() for x in (actual, previous, normalized)) or (previous <= 0).any():
        raise ValueError("输出存在非法价格或非有限数")
    delta = normalized * scale
    return [{
        "date": str(day), "previous_price": float(p),
        "actual_price": float(a), "predicted_price": float(p + d),
        "actual_change": float(a - p), "predicted_change": float(d),
        "actual_change_percent": float((a - p) / p * 100),
        "predicted_change_percent": float(d / p * 100),
        "actual_direction": direction(a - p), "predicted_direction": direction(d),
        "normalized_change_prediction": float(z), "absolute_error": float(abs(p + d - a)),
    } for day, a, p, d, z in zip(dates, actual, previous, delta, normalized)]


def metric_values(actual, predicted, previous):
    actual, predicted, previous = [np.asarray(x, dtype=np.float64) for x in (actual, predicted, previous)]
    if any(x.ndim != 1 for x in (actual, predicted, previous)) or not (len(actual) == len(predicted) == len(previous)) or not len(actual):
        raise ValueError("指标输入必须为等长非空一维数组")
    if any(not np.isfinite(x).all() for x in (actual, predicted, previous)) or (actual <= 0).any():
        raise ValueError("指标输入非法")
    error = np.abs(predicted - actual)
    change = actual - previous
    signs = np.array([direction(x) for x in change])
    predicted_signs = np.array([direction(x) for x in predicted - previous])
    result = {
        "mae": float(error.mean()), "rmse": float(np.sqrt(np.square(error).mean())),
        "mape_percent": float((error / np.abs(actual)).mean() * 100),
        "wape_percent": float(error.sum() / np.abs(actual).sum() * 100),
        "direction_accuracy": float((signs == predicted_signs).mean()),
    }
    for name, mask in (
        ("flat", np.abs(change) <= 1e-8), ("change", np.abs(change) > 1e-8),
        ("up", change > 1e-8), ("down", change < -1e-8),
    ):
        result[name + "_count"] = int(mask.sum())
        result[name + "_mae"] = float(error[mask].mean()) if mask.any() else None
    return result
