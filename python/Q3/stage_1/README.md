# Q3 · stage_1：STFT 灰度图缓存构建

把 `.mat` 里的 I/Q 转成 STFT 灰度图，按类别限量后缓存成 npy，供 stage_2 训练直接读取。
STFT 计算复用 Q2（`python/Q2/stage_1/stft_explore.py` 的 `load_iq` / `stft_db`），不重复实现。

## 1. 固定参数（不可调）

| 项 | 值 | 说明 |
|---|---|---|
| 帧长 | 2048 点 | 频率分辨率 = fs/2048 = 39.06 kHz |
| 截取时长 | 50 ms | 每个片段统一取开头 4,000,000 点 |
| 通道 | 单通道灰度 | 由 dB 幅度图直接得到 |

## 2. 可调参数（`config.json`）

| 参数 | 含义 | 默认 |
|---|---|---|
| `hop_ratio` | 帧移 / 帧长 | 0.5（hop = 1024） |
| `frames_per_image` | 每张图的时间帧数 | 1560 |
| `dtype` | 缓存精度 `uint8` / `float16` / `float32` | uint8 |
| `db_floor` | 灰度映射下限（dB，相对峰值） | −80 |
| `bg_resample` | 某类不足 `max_per_class` 时是否循环补齐 | false |
| `max_per_class` | 每类最多取多少张图，0 = 不限 | 0 |
| `splits` | 构建哪些子集 | train / test |
| `fs_hz` | 采样率 | 80 MHz |
| `window` | 窗函数 | hann |

## 3. 运行

```bash
python python/Q3/stage_1/build_cache.py
python python/Q3/stage_1/build_cache.py --max-per-class 4   # 小样本试跑
```

## 4. 切片规则

- 50 ms 内总帧数 = `(4,000,000 − 2048) / hop + 1`，hop=1024 时是 **3905 帧**
- 每张图取 `frames_per_image` 帧，余数丢弃：3905 // 1560 = **2 张/片段**（丢 785 帧）
- 同一片段只算一次 STFT，切出的多张图共享这次计算

## 5. 输出

| 位置 | 内容 |
|---|---|
| `data/_q3_cache/X_train.npy`、`X_test.npy` | memmap 数组，形状 `(N, frames_per_image, 2048)`，D 盘 |
| `data/_q3_cache/y_train.npy`、`y_test.npy` | 标签下标（int16） |
| `result/Q3/cache_index.xlsx` | 索引表：“说明”含字段与参数含义，“明细”每张图一行 |
| `log/Q3/build_cache.log` | 运行日志 |

## 6. 体积参考（uint8）

| 配置 | 张数 | 占用 |
|---|---|---|
| 全量 7 类，hop 0.5，1560 帧/图 | 2170 + 1062 | ≈ 9.62 GiB |

C 盘只剩约 6 GiB，所以缓存放在 `data/_q3_cache`（junction 指向 D 盘）。
