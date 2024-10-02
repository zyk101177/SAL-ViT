# SAL-ViT
Repository for SAL-ViT: Towards latency-efficient private inference on ViT using Selective Attention Search with a learnable softmax approximation.

---

## Running the Codes

This guide provides instructions for running different configurations of SAS-L and SAS-H models on CIFAR-10/100 and TinyImageNet datasets. The guide also includes detailed descriptions of the attention mechanisms and softmax approximations used in these models.

### Run SAS-L on CIFAR-10/100
To run SAS-L (Self-Attention + External Attention) on the CIFAR-10/100 dataset, use the following command:
```bash
python3 hybridvit_train_cifar10_100.py --dataset cifar10 --attention_mechanism hybrid --softmax_approx trainablequad2cd --self_attn_limit 3 --NAS_epoch 600 --num_heads 4
```

### Run SAS-H on CIFAR-10/100
To run SAS-H (Self-Attention + External Attention) on the CIFAR-10/100 dataset, use the following command:
```bash
python3 hybridvit_train_cifar10_100_headwise.py --dataset cifar10 --attention_mechanism hybridHeadWise --softmax_approx trainablequad2cd --self_attn_limit 12 --NAS_epoch 600
```

### Run SAS-L on TinyImageNet
To run SAS-L on the TinyImageNet dataset, use the following command:
```bash
python3 hybridvit_train_tinyimagenet.py --dataset tinyimagenet --attention_mechanism hybrid --softmax_approx trainablequad2cd --self_attn_limit 3 --n_attn_layers 9 --NAS_epoch 100
```

### Run SAS-H on TinyImageNet
To run SAS-H on the TinyImageNet dataset, use this command:
```bash
python3 hybridvit_train_tinyimagenet_headwise.py --dataset tinyimagenet --attention_mechanism hybridHeadWise --softmax_approx trainablequad2cd --self_attn_limit 12 --NAS_epoch 100 --num_heads 4
```

---

## Parameters

### Attention Mechanism:
- `hybrid` – SAS-L (combination of Self-Attention + External Attention)
- `original` – All Self-Attention (SA)
- `externalattention` – All External Attention (EA)
- `hybridHeadWise` – SAS-H (Head-wise combination of Self-Attention + External Attention)

### Softmax Approximation:
- `original` – Standard Softmax
- `relusoftmax` – 2ReLU approximation of Softmax
- `quad2` – 2Quad approximation of Softmax
- `trainablequad2cd` – Element-wise L2Q' approximation

---

## Citation

If you find this work useful in your research, please cite our paper:

```
@inproceedings{zhang2023sal,
  title={{SAL-ViT: Towards Latency Efficient Private Inference on ViT using Selective Attention Search with a Learnable Softmax Approximation}},
  author={\textbf{Yuke Zhang*} and {Dake Chen*} and {Souvik Kundu*} and {Chenghao Li} and {Peter A Beerel}},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5116--5125},
  year={2023}
}
```
