## CoReGate (MMSA-min)

这是一个从 `baselines/MMSA` 精简抽取出来的最小可运行版本，仅保留 **CoReGate**（`model_name=coregate`）及其训练/评测所需的核心代码。

### 运行

在本目录下直接运行（无需额外 `cd` 到别处）：

```bash
python -m MMSA -h
python -m MMSA -d mosi -m coregate -s 1111 -g 0
python -m MMSA -d mosei -m coregate -s 1111 -g 0
```

### 数据

默认数据路径由 `MMSA/config/config_regression.json` 里的 `dataset_root_dir` + 各数据集的 `featurePath` 拼接得到。
请确保对应的 `*.pkl` 特征文件存在。

### 依赖

见 `requirements.txt`（依赖 PyTorch / transformers / sklearn 等）。

