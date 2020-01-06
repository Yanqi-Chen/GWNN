GWNN ![License](https://img.shields.io/github/license/Yanqi-Chen/GWNN?style=plastic)
============================================

GNN课程大作业，对

> *Graph Wavelet Neural Network*. Bingbing Xu, Huawei Shen, Qi Cao, Yunqi Qiu, Xueqi Cheng. *ICLR*, 2019.
> [[Paper]](https://openreview.net/forum?id=H1ewdiR5tQ)

的复现，包含了Appendix D提到的利用Чебышёв多项式近似快速求小波基。

## 依赖

Python版本为3.7.5，主要包版本如下所示

```
cudatoolkit               10.1.243
numpy                     1.17.3
pytorch                   1.3.1
torchvision               0.4.2
scikit-learn              0.21.3
scipy                     1.3.1
tensorboard               2.0.0
tensorflow                2.0.0
tensorflow-base           2.0.0
```

## 参数选项
### 输入参数

```
  --dataset				STR		Which dataset to use.				'cora', 'citeseer' or 'pubmed'. Default is 'cora'.
  --save-path			STR		Target directory for saving models.	Default is './models'
```

### 模型参数

```
  --epochs				INT		Number of training epochs.			Default is 200.
  --hidden				INT		Number of units in hidden layer.	Default is 16.
  --weight-decay		FLOAT	Adam weight decay.					Default is 5e-4.
  --learning-rate		FLOAT	Learning rate.						Default is 0.01.
  --dropout				FLOAT	Dropout probability.				Default is 0.5.
  --approximation-order	INT		Chebyshev polynomial order.			Default is 3.
  --threshold			FLOAT	Sparsification parameter.			Default is 1e-4.
  --scale				FLOAT	Scaling parameter.					Default is 1.0.
  --fast				BOOL	Use fast graph wavelets with Chebyshev polynomial approximation.
```

## 例子

```bash
python train.py --dataset cora --fast --approximation-order 3
```

