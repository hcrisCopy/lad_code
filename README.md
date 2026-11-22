# 逻辑异常检测

先测一个问题：暗光下认错一个零件时，保留第二种解释能少报多少正常品？产品真的缺件时，又会漏掉多少？运行目录统一为 `lad_code/`。首轮看 VID-AD 的 Fruits、Tools、Dishes，分别检查数量与类型、分格放置、左右顺序。

## 环境和下载

```bash
conda create -n lad python=3.11 -y
conda activate lad
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m lad.papers
python -m lad.prepare repo vid weights
```

论文在 `../lad_research/papers/`，LogiCo 源码在 `../lad_research/repos/LogiCo/`，VID-AD 在 `../lad_data/VID-AD_dataset/`。权重需要 Meta 授权。按提示粘贴 ViT-L/16 和 dino.txt 两个原生 `.pth` 文件的完整下载地址，保存到 `../lad_data/weights/`。也可预设 `DINOV3_L_URL`、`DINOTXT_URL`。[Meta 权重申请](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) · [所需文件名](https://github.com/cnulab/LogiCo/blob/ac724ffb0c36392b443eecc41b58ba152b4298a0/ckpt/README.md)

做 LOCO 时一并下载原图、作者分割图和 ViT-B/16。按提示粘贴 MVTec 完整数据包与 Meta ViT-B/16 的下载地址，或预设 `LOCO_URL`、`DINOV3_B_URL`。[MVTec 数据申请](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad)

```bash
python -m lad.prepare loco maps baseline-weight
python -m lad.index --dataset loco
python -m lad.index --dataset vid
python -m unittest discover -s tests -v
```

索引在 `../lad_data/index/`，校验完整目录、总图数和跨划分重复文件。只跑 VID 时执行最后两条即可。

## LogiCo 作者基线

以早餐盒为例。作者发布了分割图，仓库没有提供训练好的早餐盒检测头下载入口。沿用作者模型、合成器和损失，训练 200 轮后用原版评测脚本打分。训练按 epoch 续跑，权重、loss 曲线和可视化在 `../lad_data/logico/mvtec_loco/breakfast_box/`。

```bash
R=../lad_research/repos/LogiCo
python -m lad.train_logico --category breakfast_box --epochs 200 --batch-size 14 --limit 0 --out ../lad_data/logico
python "$R/evaluation.py" --dataset mvtec_loco --category breakfast_box --data_root ../lad_data --ckpt_root ../lad_data/weights --config_root "$R/configs" --save_dir ../lad_data/logico --batch_size 16 --ckpt_file_name LogiCo_epoch_200_ckpt.pkl
```

## 首轮实验

先提取干净、降亮度、模糊三份候选，再用同一批候选和正常参考图比较所有方法。8 改为 1 即可单卡跑。调试时两条命令都加 `--limit 4`，每个场景、条件、划分和子类型各取 4 张。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 -m lad.extract --dataset vid --categories Fruits Tools Dishes --views clean dark blur --size 512 --crop-batch 8 --temperature 0.03 --min-area 0.001 --proposal-ratio 0.8 --limit 0
python -m lad.experiment --dataset vid --protocol within --categories Fruits Tools Dishes --views clean dark blur --top-k 3 --penalty 3 --spatial 1 --quantile 0.95 --limit 0
```

候选在 `../lad_data/candidates/vid/`。结果目录打印到终端，位于 `../lad_data/runs/vid/within/`。中断后重跑原命令会复用已完成的图片；加 `--clean` 重算本次选中的样本。更改视觉参数时给两步指定同一个新 `--cache`。

| 先看什么 | 看哪个文件 |
|---|---|
| 正常图变暗后，联合判定是否比硬判定、单独调阈值少误报 | `ambiguity.csv` 的 FPR |
| 各场景、拍摄条件和违规类型的 AUROC、AP、固定阈值 F1、漏报率 | `per_task.csv` |
| 同一批图上的 AUROC 差值与 95% 重采样区间 | `paired.csv` |
| 哪张图从误报变正确，哪张异常被放过 | `scores.csv` 的 id、score、threshold、alarm |

## 人工组件复核

```bash
python -m lad.review --dataset vid --categories Fruits Tools Dishes --per-stratum 3 --output ../lad_data/review/vid.jsonl
```

逐张看原图，把 `reviewer`、`violation` 和全部可见组件填回 JSONL。组件格式为 `{"name":"metal bolt","box":[0.10,0.20,0.25,0.40]}`，框为归一化左、上、右、下坐标，名称从该行词表选。确实没有组件才填空列表。异常原因可填 `missing`、`wrong_type`、`wrong_position`、`mixed` 或 `other`，正常填 `good`。同图扰动沿用原标注。

```bash
python -m lad.experiment --dataset vid --protocol within --categories Fruits Tools Dishes --views clean dark blur --top-k 3 --penalty 3 --spatial 1 --quantile 0.95 --limit 0 --oracle ../lad_data/review/vid.jsonl
```

`perception.csv` 看漏检与错认，`paired.csv` 看人工组件替换后的收益，`reviewed_recall.csv` 看真实缺件是否被放过。人工替换沿用原规则和原硬判定阈值。

## 全量与跨环境

两条首轮命令去掉 `--categories`、保留 `--views clean` 可跑 VID 官方 50 个任务。改为 `--dataset loco` 可跑 LOCO 全五类。LOCO 用官方正常验证集定阈值；VID 每个任务用全部正常训练图建库，以留一法的正常分数定阈值。

跨环境实验需要核实过的实例或采集组表 `id,group`；id 对应索引里的相对路径，同一物理实例在不同条件下用同一组号。先完成 VID 全量候选提取，再运行：

```bash
python -m lad.index --dataset vid --groups ../lad_data/review/vid_groups.csv
python -m lad.experiment --dataset vid --protocol cross --views clean --top-k 3 --penalty 3 --spatial 1 --quantile 0.95 --limit 0
```

每个场景只用 Original 建库和定阈值。混合五条件与仅未见四条件分开写入 `pooled.csv`。方法定义、论文对照和继续条件见[阅读笔记](../lad_research/论文阅读与首轮实验.md)。
