"""固定原样本；重读完整原文和文本因子；训练期建字表。无Torch/LLM调用。"""
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import numpy as np
from common import read_json, save_json, save_npz, write_csv, file_hash, fingerprint

REFERENCES = {
    "磷酸铁": "预测结果/五框架实验/formal_20261004",
    "六氟磷酸锂": "预测结果/五框架实验_多物料/六氟磷酸锂_20261026_190645",
}
TZ = timezone(timedelta(hours=8))


def chunk_ids(ids, size, overlap=0):
    if size < 5 or not 0 <= overlap < size:
        raise ValueError("块长度至少5，重叠必须小于块长度")
    ids = list(ids)
    chunks = []
    start = 0
    while start < len(ids):
        chunks.append(ids[start:start + size])
        if start + size >= len(ids):
            break
        start += size - overlap
    return chunks


def build_vocabulary(snapshots, training_ids, max_vocab):
    if max_vocab < 4:
        raise ValueError("词表容量不足")
    counts = Counter()
    for i in sorted(set(int(x) for x in training_ids)):
        counts.update(snapshots[i]["news"])
        counts.update(snapshots[i]["factor"])
    # 0=PAD，1=UNK，2=模态分隔；字表绝不读取验证/测试独有文字。
    ordered = sorted(counts, key=lambda x: (-counts[x], x))[:max_vocab - 3]
    return {char: i + 3 for i, char in enumerate(ordered)}


def publication(value):
    dt = datetime.fromisoformat(value)
    if len(value) == 10:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.replace(tzinfo=TZ) if dt.tzinfo is None else dt.astimezone(TZ)


def inspect_reference(root, material):
    portable_manifest = Path(root) / "data/input_manifest.json"
    if portable_manifest.is_file():
        spec = read_json(portable_manifest)["materials"][material]
        checked = {}
        base = Path(root).resolve()
        for relative, expected in spec["files"].items():
            p = (base / relative).resolve()
            if not p.is_relative_to(base) or not p.is_file() or file_hash(p) != expected:
                raise ValueError(f"可移植数据缺失、越界或哈希不同：{relative}")
            checked[str(p)] = expected
        ref = base / spec["reference"]
        return {"material": material, "reference": str(ref), "news": str(base / spec["news"]),
                "factor_results": str(base / spec["factor_results"]), "source_hashes": checked,
                "reference_hashes": {str(ref / "data" / name): file_hash(ref / "data" / name)
                for name in ("dataset.npz", "data_report.json", "sample_index.json", "cleaned_factors.json")}}
    ref = Path(root) / REFERENCES[material]
    manifest = read_json(ref / "manifest.json")
    source_hashes = {}
    for name, expected in manifest["hashes"].items():
        p = Path(name)
        if p.suffix.lower() == ".py":
            continue
        if not p.is_file() or file_hash(p) != expected:
            raise ValueError(f"原实验输入变更或缺失，不能混用：{p}")
        source_hashes[str(p)] = expected
    news_paths = [Path(p) for p in source_hashes if Path(p).name == "factor_inputs.jsonl"]
    factors_paths = [Path(p) for p in source_hashes if Path(p).name == "results.json"]
    if len(news_paths) != 1 or len(factors_paths) != 1:
        raise ValueError("不能唯一识别新闻和因子输入")
    artifacts = [ref / "data" / name for name in (
        "dataset.npz", "data_report.json", "sample_index.json", "cleaned_factors.json",
    )]
    return {"material": material, "reference": str(ref), "news": str(news_paths[0]),
            "factor_results": str(factors_paths[0]), "source_hashes": source_hashes,
            "reference_hashes": {str(p): file_hash(p) for p in artifacts}}


def load_prepared(folder):
    folder = Path(folder)
    ready = read_json(folder / "ready.json")
    for name, expected in ready["artifact_hashes"].items():
        if file_hash(folder / name) != expected:
            raise ValueError(f"准备数据已被改动：{folder / name}")
    with np.load(folder / "arrays.npz", allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    return {"arrays": arrays, "snapshots": read_json(folder / "snapshots.json"),
            "vocab": read_json(folder / "vocabulary.json"), "report": read_json(folder / "data_report.json")}


def prepare_material(root, material, folder, max_vocab):
    folder = Path(folder)
    info = inspect_reference(root, material)
    settings = {"reference": info, "max_vocab": max_vocab, "protocol": "raw_characters_v1"}
    if (folder / "ready.json").is_file():
        if read_json(folder / "ready.json")["settings"] != settings:
            raise ValueError("已准备数据与当前来源或词表设置不一致，请使用新输出目录")
        return load_prepared(folder)
    ref = Path(info["reference"])
    report = read_json(ref / "data/data_report.json")
    samples = read_json(ref / "data/sample_index.json")
    cleaned = read_json(ref / "data/cleaned_factors.json")
    raw_results = read_json(info["factor_results"])
    raw_by_id = {r["unit_id"]: r for r in raw_results}
    if len(raw_by_id) != len(raw_results):
        raise ValueError("因子结果ID重复")
    news = [json.loads(s) for s in Path(info["news"]).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    sources = {r["unit_id"]: r for r in news}
    if len(sources) != len(news):
        raise ValueError("新闻ID重复")
    if any(r.get("material", material) != material for r in news):
        raise ValueError("输入新闻物料不符")
    with np.load(ref / "data/dataset.npz", allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in ("p", "y", "actual", "baseline", "dates")}
        # 只复用5个状态，不将旧SVD文字向量送入新版。
        previous_flags = saved["t"][:, :, -5:].copy()
    n = len(samples)
    if n != report["total_samples"] or any(len(x) != n for x in arrays.values()):
        raise ValueError("固定样本数不一致")
    if [r["date"] for r in samples] != arrays["dates"].tolist():
        raise ValueError("目标日期不一致")
    if not np.allclose(arrays["baseline"] + arrays["y"] * report["price_difference_scale"], arrays["actual"], atol=1e-3, rtol=0):
        raise ValueError("原价差目标无法还原真实价格")
    splits = report["splits"]
    if not (splits["train"]["start"] == 0 and splits["train"]["end"] == splits["validation"]["start"]
            and splits["validation"]["end"] == splits["test"]["start"] and splits["test"]["end"] == n):
        raise ValueError("切分不连续")
    if any(a >= b for a, b in zip(arrays["dates"][:-1], arrays["dates"][1:])):
        raise ValueError("目标日期非严格递增")
    if not np.isfinite(arrays["p"]).all() or not np.isfinite(arrays["y"]).all():
        raise ValueError("固定输入存在非有限数")
    calendar = sorted({d for r in samples for d in r["price_window"]} | {r["date"] for r in samples})
    bounds = [publication(d + "T00:00:00+08:00") for d in calendar]
    index = {d: i for i, d in enumerate(calendar)}
    bins = [[] for _ in calendar]
    used_ids = {uid for r in samples for uid in r["news_ids"]}
    for uid in used_ids:
        if uid not in sources or uid not in cleaned:
            raise ValueError(f"新闻或清理记录缺失：{uid}")
    for r in news:
        if r["unit_id"] in used_ids:
            pos = bisect_right(bounds, publication(r["published_at"])) - 1
            if not 0 <= pos < len(calendar) - 1:
                raise ValueError("样本新闻不在可用时间范围")
            bins[pos].append(r["unit_id"])
    sample_snapshots = []
    flags_by_snapshot = {}
    price_by_snapshot = {}
    for i, row in enumerate(samples):
        ids = [index[d] for d in row["price_window"]]
        if len(ids) != report["window"] or row["price_window"][-1] >= row["date"]:
            raise ValueError("历史窗口长度或截止时间非法")
        actual_ids = [uid for sid in ids for uid in bins[sid]]
        if actual_ids != row["news_ids"]:
            raise ValueError(f"重建新闻与原样本不一致：{row['date']}")
        if any(publication(sources[uid]["published_at"]) >= publication(row["cutoff"]) for uid in actual_ids):
            raise ValueError("检测到未来新闻")
        for j, sid in enumerate(ids):
            numeric = arrays['p'][i, j]
            if sid in price_by_snapshot and not np.array_equal(price_by_snapshot[sid], numeric):
                raise ValueError('同一快照对应多个价格输入，不能按ID去重')
            price_by_snapshot[sid] = numeric
            flags = previous_flags[i, j].tolist()
            if sid in flags_by_snapshot and flags_by_snapshot[sid] != flags:
                raise ValueError("同一快照状态不一致")
            flags_by_snapshot[sid] = flags
        sample_snapshots.append(ids)
    snapshots, factor_catalog, news_catalog = [], [], []
    for sid, day in enumerate(calendar):
        news_texts, factor_texts = [], []
        for uid in bins[sid]:
            row, state = sources[uid], cleaned[uid]
            news_texts.append(row["title"] + "\n" + row["content"])
            news_catalog.append({"unit_id": uid, "published_at": row["published_at"],
                                 "snapshot_date": day, "title": row["title"],
                                 "source_url": row.get("source_url", ""), "content": row["content"],
                                 "factor_status": state["status"]})
            original_factors = raw_by_id.get(uid, {}).get("factors", [])
            if state["status"] != "extracted" and state["factors"]:
                raise ValueError("失败状态却存在可用因子")
            for j, f in enumerate(state["factors"]):
                if f not in original_factors:
                    raise ValueError("清理后因子不属于原提取结果")
                factor_texts.append(f["name"] + "：" + f["explanation"])
                factor_catalog.append({
                    "unit_id": uid, "factor_index": j, "published_at": row["published_at"],
                    "name": f["name"], "explanation": f["explanation"],
                    "evidence": json.dumps(f.get("evidence", []), ensure_ascii=False),
                    "source_url": row.get("source_url", ""),
                    "numerical_direction_or_strength": "未提供；因子是文字，不是涨跌标签或涨跌幅",
                })
        snapshots.append({"date": day, "news": "\n".join(news_texts),
                          "factor": "\n".join(factor_texts),
                          "flags": flags_by_snapshot.get(sid, [0., 0., 0., 0., 0.])})
    sample_snapshots = np.asarray(sample_snapshots, dtype=np.int64)
    tr = splits["train"]
    training_ids = np.unique(sample_snapshots[tr["start"]:tr["end"]])
    vocab = build_vocabulary(snapshots, training_ids, max_vocab)
    for row in snapshots:
        row["news_ids"] = [vocab.get(c, 1) for c in row["news"]]
        row["factor_ids"] = [vocab.get(c, 1) for c in row["factor"]]
    arrays["snapshot_ids"] = sample_snapshots
    report = dict(report, material=material, vocabulary_size=len(vocab) + 3,
                  news_units_used=len(used_ids), input_factor_count=len(factor_catalog),
                  preserved_news_characters=sum(len(r["news"]) for r in snapshots),
                  preserved_factor_characters=sum(len(r["factor"]) for r in snapshots),
                  representation="原文字符；不使用旧TF-IDF/SVD特征",
                  target="下一报价日标准化价差；输出恢复原尺度价格",
                  source_fingerprint=fingerprint(info))
    folder.mkdir(parents=True, exist_ok=True)
    save_npz(folder / "arrays.npz", **arrays)
    save_json(folder / "snapshots.json", snapshots)
    save_json(folder / "vocabulary.json", vocab)
    save_json(folder / "data_report.json", report)
    write_csv(folder / "新闻输入目录.csv", news_catalog)
    write_csv(folder / "因子输入目录.csv", factor_catalog)
    save_json(folder / "input_audit.json", info)
    files = ["arrays.npz", "snapshots.json", "vocabulary.json", "data_report.json",
             "新闻输入目录.csv", "input_audit.json"]
    if factor_catalog:
        files.append("因子输入目录.csv")
    save_json(folder / "ready.json", {"settings": settings,
              "artifact_hashes": {name: file_hash(folder / name) for name in files}})
    return {"arrays": arrays, "snapshots": snapshots, "vocab": vocab, "report": report}




