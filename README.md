# CPR (updaiting..)

The official code of CPR: Classifier-Projection Regularization for Continual Learning (ICLR 2021) [[arxiv]](https://arxiv.org/pdf/2006.07326.pdf)

## Quick Start

### 1. Requirements

```
$ pip install -r requirements.txt
$ mkdir weights data result_data
```

### 2. Prepare Datasets

1) Download datasets from [[this google drive link]]()
2) Locate downloaded datasets to './data' directory

```
./data
      /Permuted_Omniglot_task50.pt
      /binary_split_cub200_new
      /binary_split_cifar100
      /binary_cifar10
      /binary_omniglot
```

### 3.  Run .sh file

#### 3-1) Train 'CIFAR' scenarios using \[EWC, SI, MAS, Rwalk, AGS-CL\] with and without CPR

```
$ ./train_cifar.sh
```

#### 3-2) Train 'Omniglot' scenario using \[EWC, SI, MAS, Rwalk, AGS-CL\] with and without CPR

```
$ ./train_omniglot.sh
```

#### 3-3) Train 'Cub200' scenario using \[EWC, SI, MAS, Rwalk, AGS-CL\] with and without CPR

```
$ ./train_cub200.sh
```

## QnA
### 1. How to apply CPR to another CL algorithm?

: The implementation for CPR is quite simple. As shown in Equation (3) of the paper, you can implement CPR by maximizing an entropy of a model's softmax output (in other words, minimizing KL divergence between the model's softmax output and uniform distribution). Note that a lambda (the hyperparameter for entropy maximization) should be selected carefully.


## Citation

```
@inproceedings{
  cha2021cpr,
  title={{\{}CPR{\}}: Classifier-Projection Regularization for Continual Learning},
  author={Sungmin Cha and Hsiang Hsu and Taebaek Hwang and Flavio Calmon and Taesup Moon},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=F2v4aqEL6ze}
}
```

## Reference



