# -*- coding: utf-8 -*-
"""按场景切分数据集，生成划分清单（不移动任何文件）。

规则
----
- 场景 IN_S0 / IN_S1 -> train
- 场景 IN_S2 -> test
- background -> 按 train_ratio 随机划分（随机种子固定在 config.json）

产物
----
- output/Q1/splits/dataset_split.xlsx：每行一个文件，含个体/型号/场景/标签/划分
- log/Q1/split_dataset.log：运行日志

注意：本脚本只读取 train_extracted 下的 .mat 文件名，**不修改、不移动任何文件**。

用法
----
    python script/Q1/stage_1/split_dataset.py
"""

import json
import logging
import os
import random
import sys
from collections import Counter

from openpyxl import Workbook

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

COLUMNS = ["file", "individual", "model", "scene", "label", "split"]


def load_config(path=CONFIG_PATH):
    """读取配置，相对路径转为基于工程根目录的绝对路径。"""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("extract_dir", "output_xlsx", "log_file"):
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(PROJECT_ROOT, cfg[key])
    return cfg


def setup_logging(log_file):
    """同时输出到控制台和日志文件。"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, encoding="utf-8")],
    )
    return logging.getLogger("Q1.stage1")


def parse_file(name):
    """解析文件名 -> (individual, model, scene)。

    例：A1_IN_S0_slice_1.mat -> ("A1", "A", "IN_S0")
        background_slice_1.mat -> ("background", "BG", None)
    """
    stem = name[:-4] if name.lower().endswith(".mat") else name
    parts = stem.split("_")
    ident = parts[0]
    if ident.lower() == "background":
        return ident, "BG", None
    model = ident[0].upper() if ident else "?"
    scene = None
    for i in range(1, len(parts) - 1):
        if parts[i] in ("IN", "OUT"):
            if parts[i + 1].startswith("S"):
                scene = "%s_%s" % (parts[i], parts[i + 1])
            else:
                scene = parts[i]
            break
    return ident, model, scene


def scan_files(extract_dir, log):
    """扫描目录下所有 .mat 文件，构建基础记录列表。"""
    if not os.path.isdir(extract_dir):
        raise SystemExit("目录不存在：%s" % extract_dir)
    records = []
    for name in sorted(os.listdir(extract_dir)):
        if not name.lower().endswith(".mat"):
            continue
        ident, model, scene = parse_file(name)
        records.append({
            "file": name,
            "individual": ident,
            "model": model,
            "scene": scene,
            "label": ident,
            "split": None,
        })
    log.info("扫描到 %d 个 .mat 文件", len(records))
    return records


def assign_splits(records, cfg, log):
    """按完整采集批次分配 train/val/test，避免同源 slice 跨集合。"""
    train_scenes = set(cfg["scene_split"]["train_scenes"])
    val_scenes = set(cfg["scene_split"].get("val_scenes", []))
    test_scenes = set(cfg["scene_split"]["test_scenes"])

    bg_records = []
    for rec in records:
        scene = rec["scene"]
        if scene is None:
            bg_records.append(rec)
        elif scene in train_scenes:
            rec["split"] = "train"
        elif scene in val_scenes:
            rec["split"] = "val"
        elif scene in test_scenes:
            rec["split"] = "test"
        else:
            log.warning("未知场景 %s（文件 %s），跳过", scene, rec["file"])

    if bg_records and cfg.get("background", {}).get("include", False):
        seed = cfg["background"]["seed"]
        ratio = cfg["background"]["train_ratio"]
        rng = random.Random(seed)
        order = list(range(len(bg_records)))
        rng.shuffle(order)
        n_train = int(round(len(bg_records) * ratio))
        for rank, idx in enumerate(order):
            bg_records[idx]["split"] = "train" if rank < n_train else "test"
        log.info("background %d 个：train %d / test %d（seed=%d）",
                 len(bg_records), n_train, len(bg_records) - n_train, seed)
    elif bg_records:
        log.info("background %d 个：主任务排除，留作检测+识别扩展", len(bg_records))

    assigned = [r for r in records if r["split"] is not None]
    dropped = len(records) - len(assigned)
    if dropped:
        log.warning("有 %d 个文件未分配划分，已排除", dropped)
    return assigned


def write_xlsx(path, records):
    """写出划分清单。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "split"
    ws.append(COLUMNS)
    for rec in records:
        ws.append([rec[c] for c in COLUMNS])
    wb.save(path)


def log_summary(log, records):
    """打印划分统计。"""
    split_counter = Counter(r["split"] for r in records)
    log.info("划分统计：%s", dict(split_counter))
    scene_split = Counter((r["scene"], r["split"]) for r in records)
    for (scene, split), cnt in sorted(scene_split.items(), key=lambda x: str(x[0])):
        log.info("  场景 %s -> %s：%d", scene, split, cnt)


def main():
    cfg = load_config()
    log = setup_logging(cfg["log_file"])
    log.info("配置：%s", cfg)

    records = scan_files(cfg["extract_dir"], log)
    records = assign_splits(records, cfg, log)
    write_xlsx(cfg["output_xlsx"], records)
    log.info("清单已写出：%s（共 %d 行）", cfg["output_xlsx"], len(records))
    log_summary(log, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
