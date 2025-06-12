# CFQA: A Benchmark Dataset for Compressed Feature Quality Assessment

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This repository hosts the **first benchmark dataset** for **Compressed Feature Quality Assessment (CFQA)**, enabling the evaluation of semantic degradation in compressed deep features without relying on downstream inference. It supports research on codec optimization, transmission control, and feature reuse across tasks and architectures.
This project is the official implementation of the paper titled **“[Compressed Feature Quality Assessment: Dataset and Baselines](https://arxiv.org/abs/2506.07412)”**. 

## 🔍 Introduction

Feature coding plays a vital role in cloud-edge AI systems by enabling the transmission and storage of intermediate representations instead of raw inputs. However, traditional signal distortion metrics (e.g., MSE, PSNR) fail to capture **semantic degradation**, which leads to unpredictable downstream performance.

This repository introduces the CFQA problem and provides:
- A multi-task benchmark dataset
- Ground-truth semantic distortion labels
- Evaluation protocols and baseline metrics

## 🧩 Applications

CFQA is applicable to:
- **Codec training**: As a supervisory signal guiding semantic-preserving compression.
- **Edge-cloud systems**: For real-time estimation of compressed feature quality before transmission.
- **Rate control & caching**: Supporting adaptive encoding and transmission decisions.

![Application Scenarios](readme/application.png)

## 📦 Dataset Overview

### Total: **300 original features**, **12000 compressed features**, 3 tasks, 4 codecs.

|   Task   |   Source   | Num. of Org. Feat. |   Propocessing   |                    Feature Codecs                     | Num of Comp. Feat. |       Feature Shape       |  GT Distortion   |
|:--------:|:----------:|:------------------:|:----------------:|:-----------------------------------------------------:|:------------------:|:-------------------------:|:----------------:|
|   **Cls**    |  ImageNet  |        100         |      Resize      | HM, VTM<br>Multi-task Hyperprior<br>Task-specific Hyperprior |        4000         |         257×1536          |      Rank        |
|   **Seg**    |  VOC 2012  |        100         |  Flip and Crop   | HM, VTM<br>Multi-task Hyperprior<br>Task-specific Hyperprior |        4000         |       2×1370×1536         | mIoU difference  |
|   **Dpt**    |   NYUv2    |        100         |      Flip        | HM, VTM<br>Multi-task Hyperprior<br>Task-specific Hyperprior |        4000         |     2×4×161×1536          | RMSE difference  |


## 🛠️ Dataset Structure

```
cfqa_dataset/
├── org_feat/
│   ├── cls/
│   ├── seg/
│   └── dpt/
├── compressed_feat/
│   ├── hm/
│     ├── cls/
│     ├── seg/
│     └── dpt/
│   ├── vtm/
│     ├── cls/
│     ├── seg/
│     └── dpt/
│   ├── hyperprior_task_specific/
│     ├── cls/
│     ├── seg/
│     └── dpt/
│   ├── hyperprior_multi_task/
│     ├── cls/
│     ├── seg/
│     └── dpt/
```

## 📊 Baseline Metrics

We evaluate three representative metrics:

| Metric | Description |
|--------|-------------|
| MSE    | Measures element-wise signal distortion |
| Cosine Similarity | Captures angular relationships in feature space |
| CKA (Centered Kernel Alignment) | Measures structural similarity between token-level features |

Evaluation is conducted using **PLCC** and **SROCC**:
- PLCC: Linear correlation with task-specific degradation
- SROCC: Monotonic ranking consistency

## 🚀 Usage
### 1. Decode Features
For HM Codec
```bash
cd coding/hm; python hm.py
```
For VTM Codec
```bash
cd coding/vtm; python vtm.py
```
For Hyperprior Codecs
```bash
cd coding/CompressAI; python run_batch.py
```
Remember to configure the directories accordingly.

### 2. Evaluate Compressed Features on Various Tasks

```bash
cd machines/dinov2/; python cls.py; python seg.py; python dpt.py
```
Remember to configure the directories accordingly.

### 3. Evaluate Baseline Metrics

```bash
python corr/corr.py 
```
Remember to configure the directories accordingly.

<!-- ### 4. Analyze Results
The script will output PLCC and SROCC scores for each codec and task combination. -->

## 📈 Results Snapshot

|        **Codec**         |  **Task**  | **MSE PLCC** | **MSE SROCC** | **Cosine PLCC** | **Cosine SROCC** | **CKA PLCC** | **CKA SROCC** |
|:------------------------:|:----------:|:------------:|:-------------:|:---------------:|:----------------:|:------------:|:-------------:|
|       **HM**             |    Cls     |    0.6641    |    0.9113     |     -0.7368     |     -0.9116      |   -0.5788    |   -0.9089     |
|       **HM**                    |    Seg     |   -0.6210    |   -0.6729     |      0.6866     |      0.6729      |    0.6133    |    0.6726     |
|       **HM**                    |    Dpt     |    0.8601    |    0.9036     |     -0.8799     |     -0.9036      |   -0.8829    |   -0.9027     |
|       **VTM**            |    Cls     |    0.6220    |    0.8939     |     -0.7089     |     -0.8939      |   -0.5499    |   -0.8932     |
|       **VTM**                   |    Seg     |   -0.4784    |   -0.5213     |      0.5131     |      0.5213      |    0.4885    |    0.5213     |
|       **VTM**                   |    Dpt     |    0.8344    |    0.8750     |     -0.8572     |     -0.8747      |   -0.8478    |   -0.8754     |
| **Hyperprior (Multi-Task)** |   Cls     |   -0.0281    |    0.4707     |     -0.8907     |     -0.7165      |   -0.3277    |   -0.6593     |
| **Hyperprior (Multi-Task)**                         |    Seg     |    0.0595    |   -0.0251     |      0.3496     |      0.0604      |   -0.0059    |    0.0847     |
| **Hyperprior (Multi-Task)**                         |    Dpt     |    0.6466    |    0.6675     |     -0.4752     |     -0.6528      |   -0.5876    |   -0.6004     |
| **Hyperprior (Task-Specific)** | Cls   |   -0.0220    |    0.1303     |     -0.5486     |     -0.6084      |   -0.2622    |   -0.3365     |
| **Hyperprior (Task-Specific)**                         |    Seg     |    0.3259    |    0.3235     |     -0.0552     |     -0.1669      |   -0.3138    |   -0.2187     |
| **Hyperprior (Task-Specific)**                         |    Dpt     |   -0.0787    |    0.4915     |     -0.8427     |     -0.8258      |   -0.5483    |   -0.5638     |

📌 Cosine similarity generally performs best across tasks and codecs.

## 📚 Citation

If you use our dataset or evaluation tools, please cite the following paper:

```bibtex
@inproceedings{gao2025cfqa,
  title={Compressed Feature Quality Assessment: Dataset and Baselines},
  author={Gao, Changsheng and Zhou, Wei and Lin, Guosheng and Lin, Weisi},
  booktitle={Proceedings of ACM Multimedia (ACMMM)},
  year={2025}
}
```

## 📄 License

This dataset is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

## 🔗 Related Links

- 📘 [Paper (Preprint)](https://arxiv.org/html/2506.07412v1)
- 📁 [OneDrive Dataset Mirror](https://entuedu-my.sharepoint.com/:f:/r/personal/changsheng_gao_staff_main_ntu_edu_sg/Documents/CFQA_Dataset?csf=1&web=1&e=Bt5a3q)
<!-- - 🎬 [Demo Video (Coming Soon)](https://youtu.be/XXXXXX) -->

---

> Maintained by [@chansongoal](https://github.com/chansongoal) • Contributions welcome!
