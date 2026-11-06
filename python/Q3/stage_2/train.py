# -*- coding: utf-8 -*-
"""Q3 stage_2：STFT 灰度图 CNN 的训练与评估。

用法::

    python python/Q3/stage_2/train.py                          # 按 config.json 跑
    python python/Q3/stage_2/train.py --epochs 2               # 小跑验证流程
    python python/Q3/stage_2/train.py --weights result/Q3/weights/cnn4_best.pt
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

# PyTorch 要求在首次创建 CUDA 上下文前设置该变量，以提高 cuBLAS 复现性。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import q3_common as q3        # noqa: E402
from stage_2 import dataset as q3_ds     # noqa: E402
from stage_2 import model_zoo            # noqa: E402

CFG_PATH = Path(__file__).with_name("config.json")


def parse_args():
    """命令行只覆盖常用训练超参，其余走 config.json。"""
    parser = argparse.ArgumentParser(description="Q3 CNN 训练与评估")
    parser.add_argument("--arch", default=None, help="覆盖 config 的架构 YAML")
    parser.add_argument("--weights", default=None, help="加载预训练权重 .pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--limit-batches", type=int, default=0,
                        help="调试用：每个 epoch 只跑前 N 个 batch")
    parser.add_argument("--tag", default="", help="权重文件名后缀，便于区分实验")
    parser.add_argument("--eval-only", action="store_true",
                        help="只评估 --weights 指定的权重，不进行训练")
    return parser.parse_args()


def read_index(index_xlsx):
    """读缓存索引表，返回 (明细记录, 类别顺序)。"""
    wb = openpyxl.load_workbook(index_xlsx, read_only=True)
    info = {str(r[0]): r[1] for r in wb["说明"].iter_rows(min_row=2, values_only=True)}
    classes = [c.strip() for c in str(info.get("class_order", "")).split(",") if c.strip()]
    rows = wb["明细"].iter_rows(values_only=True)
    header = [str(c).strip() for c in next(rows)]
    records = [dict(zip(header, r)) for r in rows if r and r[0] is not None]
    wb.close()
    return records, classes


def build_loaders(cfg, records):
    """按原始文件划分 train/val，并保留独立 test 做最终评估。"""
    cache = q3.PROJECT_ROOT / cfg["cache_dir"]
    batch = int(cfg["batch_size"])
    workers = int(cfg["num_workers"])
    dtype, floor = cfg["dtype"], cfg["db_floor"]
    seeds = cfg.get("seeds", {})
    train_records = [r for r in records if str(r["split"]) == "train"]
    direct_val_records = [r for r in records if str(r["split"]) == "val"]
    test_idx = [int(r["index"]) for r in records if str(r["split"]) == "test"]

    val_ratio = float(cfg.get("val_ratio", 0.1))
    split_rng = random.Random(int(seeds.get("split", 42)))
    val_files = set()
    val_cache_name = "X_val.npy"
    if direct_val_records:
        train_idx = [int(r["index"]) for r in train_records]
        val_idx = [int(r["index"]) for r in direct_val_records]
    elif val_ratio > 0:
        def record_key(file_name):
            name = str(file_name)
            # 背景数据没有采集批次标识，只能把每个 .mat 作为独立样本；
            # 无人机数据必须按“个体+场景+批次”整组划分。
            return name if name.lower().startswith("background") else name.split("_slice_")[0]

        by_class = {}
        for rec in train_records:
            key = record_key(rec["file"])
            by_class.setdefault(str(rec["model"]), set()).add(key)
        val_groups = set()
        for groups in by_class.values():
            groups = sorted(groups)
            split_rng.shuffle(groups)
            n_val = min(len(groups) - 1, max(1, round(len(groups) * val_ratio)))
            val_groups.update(groups[:n_val])
        train_idx = [int(r["index"]) for r in train_records
                     if record_key(r["file"]) not in val_groups]
        val_idx = [int(r["index"]) for r in train_records
                   if record_key(r["file"]) in val_groups]
        val_cache_name = "X_train.npy"
    else:
        train_idx = [int(r["index"]) for r in train_records]
        val_idx = []

    ds_train = q3_ds.StftCacheDataset(cache / "X_train.npy", cache / "y_train.npy",
                                      train_idx, dtype, floor)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(seeds.get("loader", 0)))
    pin = str(cfg.get("device", "cpu")).startswith("cuda")
    loaders = {"train": DataLoader(ds_train, batch_size=batch, shuffle=True,
                                   num_workers=workers, generator=loader_gen,
                                   pin_memory=pin),
               "val": None, "test": None}
    if val_idx:
        y_val_name = val_cache_name.replace("X_", "y_")
        ds_val = q3_ds.StftCacheDataset(cache / val_cache_name, cache / y_val_name,
                                        val_idx, dtype, floor)
        loaders["val"] = DataLoader(ds_val, batch_size=batch, shuffle=False,
                                    num_workers=0, pin_memory=pin)
    if test_idx:
        ds_test = q3_ds.StftCacheDataset(cache / "X_test.npy", cache / "y_test.npy",
                                         test_idx, dtype, floor)
        loaders["test"] = DataLoader(ds_test, batch_size=batch, shuffle=False,
                                     num_workers=0, pin_memory=pin)
    logging.info("样本数 train %d / val %d / test %d（按原始文件划分）",
                 len(train_idx), len(val_idx), len(test_idx))
    return loaders


def run_epoch(model, loader, criterion, optimizer, device, scaler, amp=False,
              limit_batches=0):
    """跑一个训练 epoch，返回 (平均 loss, 准确率)。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for step, (x, y) in enumerate(loader):
        if limit_batches and step >= limit_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * x.size(0)
        correct += int((out.argmax(1) == y).sum().item())
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估一个 loader，返回 (平均 loss, 准确率, 真实标签, 预测标签)。"""
    model.eval()
    total_loss, ys, ps = 0.0, [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        total_loss += float(criterion(out, y.to(device)).item()) * x.size(0)
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys) if ys else np.array([], dtype=int)
    y_pred = np.concatenate(ps) if ps else np.array([], dtype=int)
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    return total_loss / max(y_true.size, 1), acc, y_true, y_pred


def macro_f1(y_true, y_pred, n_class):
    """逐类算 F1 后取算术平均。"""
    scores = []
    for c in range(n_class):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def confusion(y_true, y_pred, n_class):
    """混淆矩阵，行=真实，列=预测。"""
    cm = np.zeros((n_class, n_class), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def save_ckpt(path, model, arch_path, classes, epoch, eval_f1):
    """保存权重及复现所需元信息。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": str(arch_path),
                "classes": list(classes), "epoch": int(epoch),
                "eval_f1": float(eval_f1)}, path)


def load_ckpt(path, model, device):
    """加载权重，返回 checkpoint 内的元信息。"""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return ckpt if isinstance(ckpt, dict) else {}


def write_metrics(path, history, test_res, classes):
    """写指标表：'说明' + '逐轮' + '测试总览' + '每类指标'。"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "说明"
    ws.append(["工作表/字段", "含义"])
    for key, desc in [
        ("逐轮", "每个 epoch 的训练指标与测试集指标"),
        ("train_loss/train_acc", "训练集平均交叉熵与准确率"),
        ("eval_acc/eval_f1", "监控集合（优先 val，否则 test）的准确率与 Macro-F1，用于挑选最佳权重"),
        ("测试总览", "用测试集上最优权重重新评估的结果"),
        ("每类指标", "逐类 precision / recall / f1 / support"),
        ("class_order", ", ".join(classes)),
    ]:
        ws.append([key, desc])

    wh = wb.create_sheet("逐轮")
    wh.append(["epoch", "train_loss", "train_acc", "eval_acc", "eval_f1", "耗时(s)"])
    for rec in history:
        wh.append([rec["epoch"], rec["train_loss"], rec["train_acc"],
                   rec["eval_acc"], rec["eval_f1"], rec["seconds"]])

    wt = wb.create_sheet("测试总览")
    wt.append(["指标", "数值"])
    wt.append(["accuracy", test_res["acc"]])
    wt.append(["macro_f1", test_res["f1"]])
    wt.append(["n_samples", int(test_res["y_true"].size)])

    wc = wb.create_sheet("每类指标")
    wc.append(["class", "precision", "recall", "f1", "support"])
    for row in test_res["per_class"]:
        wc.append(row)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    wb.close()


def plot_curves(history, out_png, pic, monitor_name="test"):
    """训练 loss 与监控集合 acc/F1 曲线。"""
    title = ("Test set (used for model selection)" if monitor_name == "test"
             else "Validation set")
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history],
                 color=pic["palette"]["primary"], marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("Training loss")
    axes[1].plot(epochs, [h["eval_acc"] for h in history],
                 color=pic["palette"]["accent"], marker="o", label="eval acc")
    axes[1].plot(epochs, [h["eval_f1"] for h in history],
                 color=pic["palette"]["secondary"], marker="s", label="eval macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(title)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=pic["axes"]["grid_alpha"], linestyle=pic["axes"]["grid_linestyle"])
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=pic["figure"]["dpi"], bbox_inches=pic["save"]["bbox_inches"])
    plt.close(fig)


def plot_confusion(cm, classes, out_png, pic):
    """混淆矩阵热图，格内标数字。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "#262626")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=pic["figure"]["dpi"], bbox_inches=pic["save"]["bbox_inches"])
    plt.close(fig)


def per_class_rows(y_true, y_pred, classes):
    """逐类 precision/recall/f1/support。"""
    rows = []
    for c, name in enumerate(classes):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append([name, round(prec, 4), round(rec, 4), round(f1, 4), int(tp + fn)])
    return rows


def main():
    """训练 → 每轮验证 → 取最优权重做 test 评估 → 落盘指标与图。"""
    args = parse_args()
    cfg = q3.load_json(CFG_PATH)
    for key, val in (("epochs", args.epochs), ("batch_size", args.batch_size),
                     ("lr", args.lr)):
        if val:
            cfg[key] = val
    if args.arch:
        cfg["arch"] = args.arch
    weights = args.weights or cfg.get("weights")

    q3.setup_logging(q3.PROJECT_ROOT / cfg["log_file"])
    save_dir = q3.PROJECT_ROOT / cfg.get("save_dir", "result/Q3/weights")
    q3.ensure_dirs(save_dir, q3.IMAGE_DIR)
    pic = q3.load_json(q3.PIC_CFG_PATH)

    # CPU 训练默认只用一半核心，这里显式用满；OMP_NUM_THREADS 优先
    threads = int(os.environ.get("OMP_NUM_THREADS", 0)) or (os.cpu_count() or 1)
    torch.set_num_threads(threads)
    logging.info("CPU 线程 %d", threads)

    seeds = cfg.get("seeds", {})
    init_seed = int(seeds.get("init", 42))
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    logging.info("随机种子 split=%s init=%s loader=%s", seeds.get("split"),
                 seeds.get("init"), seeds.get("loader"))

    records, classes = read_index(q3.PROJECT_ROOT / cfg["index_xlsx"])
    n_class = len(classes)
    loaders = build_loaders(cfg, records)

    arch_path = q3.PROJECT_ROOT / cfg["arch"]
    arch = model_zoo.load_arch(arch_path)
    device = torch.device(cfg["device"])
    model = model_zoo.build_model(arch, n_class).to(device)
    logging.info("架构 %s | 参数 %d | 设备 %s", arch.get("name", arch_path.stem),
                 model_zoo.count_params(model), device)
    if weights:
        meta = load_ckpt(q3.PROJECT_ROOT / weights, model, device)
        logging.info("已加载权重 %s（epoch=%s）", weights, meta.get("epoch"))

    criterion = nn.CrossEntropyLoss()
    if args.eval_only:
        if not weights:
            raise SystemExit("--eval-only 必须同时提供 --weights")
        if loaders["test"] is None:
            raise SystemExit("没有 test 子集，无法评估")
        _, test_acc, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
        test_f1 = macro_f1(y_true, y_pred, n_class)
        logging.info("eval-only test: acc %.4f | macro-F1 %.4f | 权重 %s",
                     test_acc, test_f1, weights)
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                                 weight_decay=float(cfg["weight_decay"]))
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    sched_cfg = cfg.get("lr_scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 2)),
        min_lr=float(sched_cfg.get("min_lr", 1e-5)))
    logging.info("混合精度 %s | 初始学习率 %.2g", amp, float(cfg["lr"]))
    tag = ("_" + args.tag) if args.tag else ""
    best_path = save_dir / ("%s%s_best.pt" % (arch_path.stem, tag))
    last_path = save_dir / ("%s%s_last.pt" % (arch_path.stem, tag))

    monitor_name = "val" if loaders.get("val") else "test"
    history, best_f1 = [], -1.0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        start = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer,
                                    device, scaler, amp, args.limit_batches)
        _, ev_acc, y_t, y_p = evaluate(model, loaders[monitor_name], criterion, device)
        ev_f1 = macro_f1(y_t, y_p, n_class)
        scheduler.step(ev_f1)
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 5),
                        "train_acc": round(tr_acc, 4), "eval_acc": round(ev_acc, 4),
                        "eval_f1": round(ev_f1, 4),
                        "seconds": round(time.time() - start, 1)})
        logging.info("epoch %d | train_loss %.4f acc %.3f | %s acc %.3f f1 %.3f | %.1fs",
                     epoch, tr_loss, tr_acc, monitor_name, ev_acc, ev_f1,
                     time.time() - start)
        save_ckpt(last_path, model, arch_path, classes, epoch, ev_f1)
        if ev_f1 > best_f1:
            best_f1 = ev_f1
            save_ckpt(best_path, model, arch_path, classes, epoch, ev_f1)
            logging.info("  → 新的最优模型（f1 %.4f）已保存", ev_f1)

    if loaders["test"] is None:
        logging.warning("没有 test 子集，跳过最终评估")
        return
    load_ckpt(best_path, model, device)
    _, test_acc, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
    test_f1 = macro_f1(y_true, y_pred, n_class)
    test_res = {"acc": round(test_acc, 4), "f1": round(test_f1, 4),
                "y_true": y_true, "per_class": per_class_rows(y_true, y_pred, classes)}
    logging.info("test: acc %.4f | macro-F1 %.4f | best 权重 %s", test_acc, test_f1, best_path)

    def tagged(rel_path):
        path = q3.PROJECT_ROOT / rel_path
        return path.with_name(path.stem + tag + path.suffix)

    metrics_path = tagged(cfg["metrics_xlsx"])
    curve_path = tagged(cfg["loss_curve_png"])
    confusion_path = tagged(cfg["confusion_png"])
    write_metrics(metrics_path, history, test_res, classes)
    plot_curves(history, curve_path, pic, monitor_name)
    plot_confusion(confusion(y_true, y_pred, n_class), classes,
                   confusion_path, pic)
    logging.info("指标 %s | 曲线 %s | 混淆矩阵 %s", metrics_path,
                 curve_path, confusion_path)


if __name__ == "__main__":
    main()
