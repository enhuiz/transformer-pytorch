# TorchNMT

## Introduction

A simple implementation of the neural machine translation framework using pytorch.

## Models

- Transformer (https://arxiv.org/abs/1706.03762)

## Datasets

- **Xiaoshi**: A Chinese to Chinese traditional poetry dataset (https://github.com/enhuiz/XiaoShi).

- **Multi30K (de-en)**: WMT'16 Multimodal Translation task (de-en) (http://www.statmt.org/wmt16/multimodal-task.html).

## Example

### Prerequisites

```bash
pip install -r requirements.txt
```

### Train

```bash
python scripts/train.py config/xiaoshi/transformer.yml
```

### Test

```bash
python scripts/test.py config/xiaoshi/transformer.yml
```

<!-- ## Results -->
<!-- TODO -->

## Acknowledgement

- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://github.com/OpenNMT/OpenNMT-py
