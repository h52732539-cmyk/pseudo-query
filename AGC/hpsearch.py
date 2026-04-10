"""
AGC 超参数网格搜索: 在关键超参数组合上运行训练，记录最优配置。

用法:
    python hpsearch.py [--config configs/default_agc.yaml] [--device cuda] [--dry-run]

搜索维度 (基于失效分析诊断):
  1. beta_bal   — 聚类均衡损失权重 (对抗 Gini=0.93 聚类崩塌)
  2. beta_div   — codebook 多样性损失权重 (对抗 cos_mean=0.44 退化)
  3. tau_end    — 温度退火下界 (控制软→硬分配程度)
  4. lr         — 学习率 (对抗严重过拟合)
  5. label_smoothing — 标签平滑 (对抗过拟合)
"""
import argparse
import itertools
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

AGC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGC_DIR))

# ─────────────── 搜索空间定义 ───────────────

SEARCH_SPACE = {
    "loss.beta_bal":        [0.05, 0.1, 0.2],
    "loss.beta_div":        [0.01, 0.05, 0.1],
    "agc.tau_end":          [0.2, 0.3, 0.5],
    "training.lr":          [3e-5, 5e-5, 1e-4],
    "loss.label_smoothing": [0.0, 0.1],
}

# 互斥 / 条件约束: 减少无效组合
def is_valid_combo(combo: dict) -> bool:
    """
    过滤不合理的组合:
    - beta_div=0.1 + beta_bal=0.05 时辅助损失可能不平衡 → 跳过
    - lr=1e-4 + label_smoothing=0 已是 baseline，已知失败 → 跳过
    """
    if combo["loss.beta_div"] >= 0.1 and combo["loss.beta_bal"] <= 0.05:
        return False
    if combo["training.lr"] >= 1e-4 and combo["loss.label_smoothing"] == 0.0:
        return False
    return True


def generate_combos(space: dict, max_combos: int = 50) -> list:
    """生成所有合法组合，限制最大数量。"""
    keys = sorted(space.keys())
    values = [space[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combo = dict(zip(keys, vals))
        if is_valid_combo(combo):
            combos.append(combo)
    if len(combos) > max_combos:
        # 优先级排序: 高 beta_bal、中 lr、中 tau_end 的组合更可能有效
        combos.sort(key=lambda c: (
            -c["loss.beta_bal"],
            abs(c["training.lr"] - 5e-5),
            abs(c["agc.tau_end"] - 0.3),
        ))
        combos = combos[:max_combos]
    return combos


def combo_to_overrides(combo: dict) -> str:
    """将组合转为 --overrides 格式字符串。"""
    parts = []
    for k, v in sorted(combo.items()):
        parts.append(f"{k}={v}")
    return ";".join(parts)


def combo_to_name(combo: dict) -> str:
    """生成简短运行名称。"""
    parts = []
    short_keys = {
        "loss.beta_bal": "bal",
        "loss.beta_div": "div",
        "agc.tau_end": "tau",
        "training.lr": "lr",
        "loss.label_smoothing": "ls",
    }
    for k in sorted(combo.keys()):
        sk = short_keys.get(k, k.split(".")[-1])
        v = combo[k]
        if isinstance(v, float) and v < 0.01:
            parts.append(f"{sk}{v:.0e}")
        else:
            parts.append(f"{sk}{v}")
    return "_".join(parts)


# ─────────────── 主程序 ───────────────

def main():
    parser = argparse.ArgumentParser(description="AGC Hyperparameter Search")
    parser.add_argument("--config", type=str, default="configs/default_agc.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_combos", type=int, default=50,
                        help="Maximum number of combos to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print combos, don't train")
    args = parser.parse_args()

    # 日志
    log_dir = AGC_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = AGC_DIR / f"hpsearch_results_{timestamp}.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"hpsearch_{timestamp}.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("hpsearch")

    # 生成组合
    combos = generate_combos(SEARCH_SPACE, max_combos=args.max_combos)
    logger.info(f"Total combos to search: {len(combos)}")
    logger.info(f"Results will be saved to: {results_file}")

    if args.dry_run:
        for i, combo in enumerate(combos):
            name = combo_to_name(combo)
            overrides = combo_to_overrides(combo)
            logger.info(f"  [{i+1:02d}] {name}: --overrides '{overrides}'")
        logger.info(f"Dry run complete. {len(combos)} combos listed.")
        return

    # 导入训练函数
    from train import main as train_main, apply_overrides

    results = []
    best_r1 = 0.0
    best_combo = None

    for i, combo in enumerate(combos):
        name = combo_to_name(combo)
        overrides = combo_to_overrides(combo)
        logger.info(f"{'='*60}")
        logger.info(f"[{i+1}/{len(combos)}] Running: {name}")
        logger.info(f"  Overrides: {overrides}")

        # 构造 sys.argv 来调用 train_main
        sys.argv = [
            "train.py",
            "--config", args.config,
            "--overrides", overrides,
            "--run_name", name,
        ]
        if args.device:
            sys.argv.extend(["--device", args.device])

        try:
            best_val_r1 = train_main()
            if best_val_r1 is None:
                best_val_r1 = 0.0
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            best_val_r1 = 0.0

        result = {
            "run_name": name,
            "overrides": overrides,
            "combo": combo,
            "best_val_r1": best_val_r1,
        }
        results.append(result)

        logger.info(f"  Result: R@1 = {best_val_r1:.2f}")

        if best_val_r1 > best_r1:
            best_r1 = best_val_r1
            best_combo = combo
            logger.info(f"  ★ New overall best R@1: {best_r1:.2f}")

        # 实时保存结果
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "search_space": {k: [str(v) for v in vs] for k, vs in SEARCH_SPACE.items()},
                "total_combos": len(combos),
                "completed": i + 1,
                "best_r1": best_r1,
                "best_combo": best_combo,
                "results": sorted(results, key=lambda r: -r["best_val_r1"]),
            }, f, indent=2, ensure_ascii=False)

    # 最终汇总
    logger.info(f"\n{'='*60}")
    logger.info(f"Hyperparameter search complete!")
    logger.info(f"Total runs: {len(results)}")
    logger.info(f"Best R@1: {best_r1:.2f}")
    logger.info(f"Best combo: {best_combo}")
    logger.info(f"Results saved to: {results_file}")

    # 打印 Top-5
    sorted_results = sorted(results, key=lambda r: -r["best_val_r1"])
    logger.info(f"\nTop-5 configurations:")
    for rank, r in enumerate(sorted_results[:5], 1):
        logger.info(f"  #{rank} R@1={r['best_val_r1']:.2f} | {r['run_name']}")


if __name__ == "__main__":
    main()
