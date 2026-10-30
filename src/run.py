"""五框架×双物料×独立原文编码/联合编码。默认480候选训练；无API。"""
import argparse
from datetime import datetime
from pathlib import Path
import platform
import os
import sys
import numpy as np
from common import read_json, save_json, file_hash, output_lock
from data import REFERENCES, prepare_material, inspect_reference
from experiments import MODELS, validate_recipes, group_validation, choose_winners, selection_scopes


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("实验配置.json"))
    parser.add_argument("--stage", choices=["prepare", "train", "evaluate", "all"], default="all")
    parser.add_argument("--materials", nargs="+", choices=list(REFERENCES), default=list(REFERENCES))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--recipes", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--embedding", type=int, default=32)
    parser.add_argument("--max-vocab", type=int, default=8000)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action="store_true", help="强制确定性；个别CUDA算子不支持时会报错")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.embedding, args.chunk_batch, args.threads) < 1:
        parser.error("轮数、批大小、维度与线程数必须为正")
    if args.hidden < 2 or args.hidden % 2 or args.chunk_size < 5 or args.max_vocab < 4 or not 0 < args.lr < float("inf"):
        parser.error("hidden须为正偶数，分块至少5，词表至少4，学习率须为正有限数")
    for name in ("materials", "models", "seeds"):
        seq = getattr(args, name)
        if len(seq) != len(set(seq)):
            parser.error(f"{name}不能重复")
    if any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seed应在0至2^32-1之间")
    return args


def main():
    args = parse_args()
    recipes = read_json(args.config)["recipes"]
    validate_recipes(recipes)
    if args.recipes:
        if len(args.recipes) != len(set(args.recipes)) or set(args.recipes) - {r["id"] for r in recipes}:
            raise ValueError("指定方案重复或不存在")
        recipes = [r for r in recipes if r["id"] in args.recipes]
    pairs = validate_recipes(recipes)
    total = len(args.materials) * len(args.models) * len(recipes) * len(args.seeds)
    scopes = len({scope for scope_list in selection_scopes(recipes).values() for scope in scope_list})
    test_upper = len(args.materials) * len(args.models) * scopes * len(args.seeds)
    plan = {"materials": args.materials, "frameworks": args.models, "recipes": recipes,
            "seeds": args.seeds, "candidate_trainings": total, "selected_test_evaluations_at_most": test_upper,
            "paired_orders": pairs,
            "target": "下一报价日价格；因子是文字输入；方向和变化百分比由价格回归派生"}
    if args.plan_only:
        import json
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    requested = args.device
    args.device = 'cuda:0' if requested == 'cuda' or (requested == 'auto' and torch.cuda.is_available()) else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('要求CUDA，但PyTorch没有可用CUDA。请检查驱动、GPU可见性和CUDA版torch。')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    print('计算设备：' + args.device + (' / ' + torch.cuda.get_device_name(0) if args.device.startswith('cuda') else ''), flush=True)
    from training import train_one, evaluate_one, folder_for
    from reports import write_reports
    torch.set_num_threads(args.threads)
    args.root = args.root.resolve()
    args.output = (args.output or args.root / "预测结果/五框架分模态编码" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    args.config = args.config.resolve()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in {"stage", "plan_only"}}
    code = list(Path(__file__).parent.glob("*.py")) + [args.config]
    references = {material: inspect_reference(args.root, material) for material in args.materials}
    manifest = {"protocol": "modality_specific_raw_text_v1", "config": config, "recipes": recipes,
                "source_hashes": {str(p.resolve()): file_hash(p) for p in code},
                "inputs": references, "environment": {"python": platform.python_version(),
                "torch": str(torch.__version__), "numpy": np.__version__, "device": args.device,
                "cuda_runtime": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None},
                "target": "next_quotation_price_via_normalized_difference"}
    with output_lock(args.output):
        manifest_path = args.output / "manifest.json"
        if manifest_path.is_file() and read_json(manifest_path) != manifest:
            raise ValueError("此输出目录的代码、输入、环境或参数不一致。请使用新目录，避免混入结果。")
        save_json(manifest_path, manifest)
        save_json(args.output / "experiment_plan.json", plan)
        datasets = {}
        for material in args.materials:
            datasets[material] = prepare_material(args.root, material, args.output / "data" / material, args.max_vocab)
            report = datasets[material]["report"]
            print(f"准备 {material}：样本{report['total_samples']}；本轮窗口内新闻{report['news_units_used']}；"
                  f"输入因子{report['input_factor_count']}；词表{report['vocabulary_size']}", flush=True)
        prepared = {m: file_hash(args.output / "data" / m / "ready.json") for m in datasets}
        prepared_path = args.output / "prepared_inputs.json"
        if prepared_path.is_file() and read_json(prepared_path) != prepared:
            raise ValueError("准备数据发生变化")
        save_json(prepared_path, prepared)
        if args.stage == "prepare":
            print("只准备数据，未训练。输出：" + str(args.output))
            return
        jobs = [(m, name, r, seed) for m in args.materials for name in args.models for r in recipes for seed in args.seeds]
        if args.stage == "evaluate":
            missing = [folder_for(args.output, m, name, r["id"], seed) for m, name, r, seed in jobs
                       if not (folder_for(args.output, m, name, r["id"], seed) / "validation_metrics.json").is_file()]
            if missing:
                raise ValueError(f"尚缺{len(missing)}次候选训练；evaluate阶段不会补训练，先运行train阶段")
        records = []
        for m, name, recipe, seed in jobs:
            records.append(train_one(datasets[m], m, name, recipe, seed, args.output, args))
            save_json(args.output / "progress.json", {"completed": len(records), "planned": total,
                      "last": {"material": m, "model": name, "recipe_id": recipe["id"], "seed": seed}})
            print(f"候选完成 {len(records)}/{total}", flush=True)
        groups = group_validation(records, recipes, args.seeds)
        if len(groups) != len(args.materials) * len(args.models) * len(recipes):
            raise ValueError("候选不完整，不允许测试")
        winners, unique = choose_winners(groups)
        save_json(args.output / "验证选优.json", {"rule": "三种子平均验证MAE；测试不参与",
                  "groups": groups, "winners_by_scope": winners, "unique_selected": unique})
        tests = []
        planned_tests = len(unique) * len(args.seeds)
        for chosen in unique:
            for seed in args.seeds:
                marker = folder_for(args.output, chosen["material"], chosen["model"], chosen["recipe_id"], seed) / "selected_test_metrics.json"
                if args.stage in {"all", "evaluate"} or marker.is_file():
                    tests.append(evaluate_one(datasets[chosen["material"]], chosen, seed, args.output, args))
                    save_json(args.output / "test_progress.json", {"completed": len(tests), "planned": planned_tests})
        write_reports(args.output, datasets, records, groups, winners, tests)
        if args.stage == "train":
            print("候选训练与验证选择已完成；新测试尚未执行。用同参数及同输出目录运行 --stage evaluate。")
        else:
            print(f"完成：候选{len(records)}/{total}；选中测试{len(tests)}/{planned_tests}。")
        print("输出目录：" + str(args.output), flush=True)


if __name__ == "__main__":
    main()






