# CPR

The official code of CPR: Classifier-Projection Regularization for Continual Learning (ICLR 2021) [[arxiv]](https://arxiv.org/pdf/2006.07326.pdf)

## Quick Start

### 1. Requirements

```
$ pip install -r requirements.txt
$ mkdir weights data
```

### 2. Prepare Datasets

1) Download datasets (CIFAR[1], Omniglot[2] and CUB200[3]) from [[this google drive link]](https://drive.google.com/file/d/19UaTcjGYj8YUBlj69mPK7zcVvFUR8bso/view?usp=sharing)
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

#### 3-3) Train 'CUB200' scenario using \[EWC, SI, MAS, Rwalk, AGS-CL\] with and without CPR

```
$ ./train_cub200.sh
```

### 3.  Analyze experimental results

1) Check './result_analysis_code/'. There are example ipython files to anayze the experimental results of [EWC, MAS, SI, Rwalk, AGS-CL] with or without CPR in CIFAR100. Note that the analysis results are for experiments conducted on only single seed.

2) You can easily transform and use these files to analyze other results!


## QnA
### 1. How to apply CPR to another CL algorithm?

: The implementation for CPR is quite simple. As shown in Equation (3) of the paper, you can implement CPR by maximizing an entropy of a model's softmax output (in other words, minimizing KL divergence between the model's softmax output and uniform distribution). Note that a lambda (the hyperparameter for entropy maximization) should be selected carefully. As an example, check Line 222 at './approaches/ewc_cpr.py'.


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
[1] Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images." (2009): 7.

[2] Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "Human-level concept learning through probabilistic program induction." Science 350.6266 (2015): 1332-1338.

[3] Welinder, Peter, et al. "Caltech-UCSD birds 200." (2010).


