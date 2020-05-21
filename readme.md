# Classical ML Equations in LaTeX

A collection of classical ML equations in Latex . Some of them are provided with simple notes and paper link. Hopes to help writings such as papers and blogs.

- [Classical ML Equations in LaTeX](#classical-ml-equations-in-latex)
  - [Model](#model)
    - [RNNs(LSTM, GRU)](#rnnslstm-gru)
    - [Attentional Seq2seq](#attentional-seq2seq)
      - [Bahdanau Attention](#bahdanau-attention)
      - [Luong(Dot-Product) Attention](#luongdot-product-attention)
    - [transformer](#transformer)
      - [Scaled Dot-Product attention](#scaled-dot-product-attention)
      - [Multi-head attention](#multi-head-attention)
  - [Activations](#activations)
    - [Sigmoid](#sigmoid)
    - [Softmax](#softmax)
    - [Relu](#relu)
  - [Loss](#loss)
    - [Regression](#regression)
      - [Mean Absolute Error(MAE)](#mean-absolute-errormae)
      - [Mean Squared Error(MSE)](#mean-squared-errormse)
      - [Huber loss](#huber-loss)
    - [Classification](#classification)
      - [Cross Entropy](#cross-entropy)
      - [Negative Loglikelihood](#negative-loglikelihood)
      - [Hinge loss](#hinge-loss)
      - [KL/JS divergence](#kljs-divergence)
    - [Regularization](#regularization)
      - [L1 regularization](#l1-regularization)
      - [L2 regularization](#l2-regularization)
  - [Metrics](#metrics)
    - [Classification](#classification-1)
      - [Accuracy, Precision, Recall, F1](#accuracy-precision-recall-f1)
      - [Sensitivity, Specificity and AUC](#sensitivity-specificity-and-auc)
    - [Regression](#regression-1)
    - [Clustering](#clustering)
      - [(Normalized) Mutual Information (NMI)](#normalized-mutual-information-nmi)
    - [Ranking](#ranking)
      - [(Mean) Average Precision(MAP)](#mean-average-precisionmap)
    - [Similarity/Relevance](#similarityrelevance)
      - [Cosine](#cosine)
      - [Jaccard](#jaccard)
      - [Pointwise Mutual Information(PMI)](#pointwise-mutual-informationpmi)
  - [Notes](#notes)
  - [Reference](#reference)

## Model

### RNNs(LSTM, GRU)

encoder hidden state  ![math](https://render.githubusercontent.com/render/math?math=h_t)  at time step  ![math](https://render.githubusercontent.com/render/math?math=t) 


![math](https://render.githubusercontent.com/render/math?math=h_t%20%3D%20RNN_%7Benc%7D%28x_t%2C%20h_%7Bt-1%7D%29)



decoder hidden state  ![math](https://render.githubusercontent.com/render/math?math=s_t)  at time step  ![math](https://render.githubusercontent.com/render/math?math=t) 



![math](https://render.githubusercontent.com/render/math?math=s_t%20%3D%20RNN_%7Bdec%7D%28y_t%2C%20s_%7Bt-1%7D%29)



```
h_t = RNN_{enc}(x_t, h_{t-1})
s_t = RNN_{dec}(y_t, s_{t-1})
```

The  ![math](https://render.githubusercontent.com/render/math?math=RNN_%7Benc%7D) ,  ![math](https://render.githubusercontent.com/render/math?math=RNN_%7Bdec%7D)  are usually either 

- LSTM (paper: [Long short-term memory](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf)) 

-  GRU (paper: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)).

### Attentional Seq2seq

The attention weight  ![math](https://render.githubusercontent.com/render/math?math=%5Calpha_%7Bij%7D) , the  ![math](https://render.githubusercontent.com/render/math?math=i) th decoder step over the  ![math](https://render.githubusercontent.com/render/math?math=j) th encoder step, resulting in context vector  ![math](https://render.githubusercontent.com/render/math?math=c_i) 



![math](https://render.githubusercontent.com/render/math?math=c_i%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7BT_x%7D%20%5Calpha_%7Bij%7Dh_j)





![math](https://render.githubusercontent.com/render/math?math=%5Calpha_%7Bij%7D%20%3D%20%5Cfrac%7B%5Cexp%28e_%7Bij%7D%29%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BT_x%7D%20%5Cexp%28e_%7Bik%7D%29%7D)





![math](https://render.githubusercontent.com/render/math?math=e_%7Bik%7D%20%3D%20a%28s_%7Bi-1%7D%2C%20h_j%29)



```
c_i = \sum_{j=1}^{T_x} \alpha_{ij}h_j

\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}

e_{ik} = a(s_{i-1}, h_j)
```

 ![math](https://render.githubusercontent.com/render/math?math=a)  is an specific attention function, which can be

#### Bahdanau Attention

Paper: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)



![math](https://render.githubusercontent.com/render/math?math=e_%7Bik%7D%20%3D%20v%5ET%20tanh%28W%5Bs_%7Bi-1%7D%3B%20h_j%5D%29)



#### Luong(Dot-Product) Attention

Paper: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

If  ![math](https://render.githubusercontent.com/render/math?math=s_i)  and  ![math](https://render.githubusercontent.com/render/math?math=h_j)  has same number of dimension.



![math](https://render.githubusercontent.com/render/math?math=e_%7Bik%7D%20%3D%20s_%7Bi-1%7D%5ET%20h_j)



otherwise



![math](https://render.githubusercontent.com/render/math?math=e_%7Bik%7D%20%3D%20s_%7Bi-1%7D%5ET%20W%20h_j)



```
e_{ik} = s_{i-1}^T h_j

e_{ik} = s_{i-1}^T W h_j
```

Finally, the output  ![math](https://render.githubusercontent.com/render/math?math=o_i)  is produced by:



![math](https://render.githubusercontent.com/render/math?math=s_t%20%3D%20tanh%28W%5Bs_%7Bt-1%7D%3By_t%3Bc_t%5D%29)




![math](https://render.githubusercontent.com/render/math?math=o_t%20%3D%20softmax%28Vs_t%29)



```
s_t = tanh(W[s_{t-1};y_t;c_t])
o_t = softmax(Vs_t)
```

### transformer

Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

#### Scaled Dot-Product attention



![math](https://render.githubusercontent.com/render/math?math=Attention%28Q%2C%20K%2C%20V%29%20%3D%20softmax%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%29V)



```
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
```

where  ![math](https://render.githubusercontent.com/render/math?math=%5Csqrt%7Bd_k%7D)  is the dimension of the key vector  ![math](https://render.githubusercontent.com/render/math?math=k)  and query vector  ![math](https://render.githubusercontent.com/render/math?math=q)  .

#### Multi-head attention



![math](https://render.githubusercontent.com/render/math?math=MultiHead%28Q%2C%20K%2C%20V%20%29%20%3D%20Concat%28head_1%2C%20...%2C%20head_h%29W%5EO)



where 


![math](https://render.githubusercontent.com/render/math?math=head_i%20%3D%20Attention%28Q%20W%5EQ_i%2C%20K%20W%5EK_i%2C%20V%20W%5EV_i%29)



```
MultiHead(Q, K, V ) = Concat(head_1, ..., head_h)W^O

head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
```

## Activations

### Sigmoid
Related to *Logistic Regression*. For single-label/multi-label binary classification.



![math](https://render.githubusercontent.com/render/math?math=%5Csigma%28z%29%20%3D%20%5Cfrac%7B1%7D%20%7B1%20%2B%20e%5E%7B-z%7D%7D)



```
\sigma(z) = \frac{1} {1 + e^{-z}}
```

### Softmax

For multi-class single label classification.



![math](https://render.githubusercontent.com/render/math?math=%5Csigma%28z_i%29%20%3D%20%5Cfrac%7Be%5E%7Bz_%7Bi%7D%7D%7D%7B%5Csum_%7Bj%3D1%7D%5EK%20e%5E%7Bz_%7Bj%7D%7D%7D%20%5C%20%5C%20%5C%20for%5C%20i%3D1%2C2%2C%5Cdots%2CK)



```
\sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K
```

### Relu



![math](https://render.githubusercontent.com/render/math?math=Relu%28z%29%20%3D%20max%280%2C%20z%29)



```
Relu(z) = max(0, z)
```

## Loss

### Regression

Below  ![math](https://render.githubusercontent.com/render/math?math=x)  and  ![math](https://render.githubusercontent.com/render/math?math=y)  are  ![math](https://render.githubusercontent.com/render/math?math=D)  dimensional vectors, and  ![math](https://render.githubusercontent.com/render/math?math=x_i)  denotes the value on the  ![math](https://render.githubusercontent.com/render/math?math=i) th dimension of  ![math](https://render.githubusercontent.com/render/math?math=x) .

#### Mean Absolute Error(MAE) 



![math](https://render.githubusercontent.com/render/math?math=%5Csum_%7Bi%3D1%7D%5E%7BD%7D%7Cx_i-y_i%7C)



```
\sum_{i=1}^{D}|x_i-y_i|
```

#### Mean Squared Error(MSE)



![math](https://render.githubusercontent.com/render/math?math=%5Csum_%7Bi%3D1%7D%5E%7BD%7D%28x_i-y_i%29%5E2)



```
\sum_{i=1}^{D}(x_i-y_i)^2
```

#### Huber loss

It’s less sensitive to outliers than the MSE as it treats error as square only inside an interval.



![math](https://render.githubusercontent.com/render/math?math=L_%7B%5Cdelta%7D%3D%0A%20%20%20%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%0A%20%20%20%20%20%20%20%20%5Cfrac%7B1%7D%7B2%7D%28y%20-%20%5Chat%7By%7D%29%5E%7B2%7D%20%26%20if%20%5Cleft%20%7C%20%28y%20-%20%5Chat%7By%7D%29%20%20%5Cright%20%7C%20%3C%20%5Cdelta%5C%5C%0A%20%20%20%20%20%20%20%20%5Cdelta%20%28%28y%20-%20%5Chat%7By%7D%29%20-%20%5Cfrac1%202%20%5Cdelta%29%20%26%20otherwise%0A%20%20%20%20%5Cend%7Bmatrix%7D%5Cright.)



```
L_{\delta}=
    \left\{\begin{matrix}
        \frac{1}{2}(y - \hat{y})^{2} & if \left | (y - \hat{y})  \right | < \delta\\
        \delta ((y - \hat{y}) - \frac1 2 \delta) & otherwise
    \end{matrix}\right.
```

### Classification

#### Cross Entropy

- In binary classification, where the number of classes  ![math](https://render.githubusercontent.com/render/math?math=M)  equals 2, Binary Cross-Entropy(BCE) can be calculated as:



![math](https://render.githubusercontent.com/render/math?math=-%7B%28y%5Clog%28p%29%20%2B%20%281%20-%20y%29%5Clog%281%20-%20p%29%29%7D)



- If  ![math](https://render.githubusercontent.com/render/math?math=M%20%3E%202)  (i.e. multiclass classification), we calculate a separate loss for each class label per observation and sum the result.



![math](https://render.githubusercontent.com/render/math?math=-%5Csum_%7Bc%3D1%7D%5EMy_%7Bo%2Cc%7D%5Clog%28p_%7Bo%2Cc%7D%29)



```
-{(y\log(p) + (1 - y)\log(1 - p))}

-\sum_{c=1}^My_{o,c}\log(p_{o,c})
```

> M - number of classes 
> 
> log - the natural log
> 
> y - binary indicator (0 or 1) if class label c is the correct classification for observation o
> 
> p - predicted probability observation o is of class c

#### Negative Loglikelihood



![math](https://render.githubusercontent.com/render/math?math=NLL%28y%29%20%3D%20-%7B%5Clog%28p%28y%29%29%7D)



Minimizing negative loglikelihood 



![math](https://render.githubusercontent.com/render/math?math=%5Cmin_%7B%5Ctheta%7D%20%5Csum_y%20%7B-%5Clog%28p%28y%3B%5Ctheta%29%29%7D)




is equivalent to Maximum Likelihood Estimation(MLE).




![math](https://render.githubusercontent.com/render/math?math=%5Cmax_%7B%5Ctheta%7D%20%5Cprod_y%20p%28y%3B%5Ctheta%29)



Here  ![math](https://render.githubusercontent.com/render/math?math=p%28y%29)  is a *scaler* instead of *vector*. It is the value of the single dimension where the ground truth  ![math](https://render.githubusercontent.com/render/math?math=y)  lies. It is thus equivalent to cross entropy (See [wiki](https://en.wikipedia.org/wiki/Cross_entropy)).\

```
NLL(y) = -{\log(p(y))}

\min_{\theta} \sum_y {-\log(p(y;\theta))}

\max_{\theta} \prod_y p(y;\theta)
```

#### Hinge loss

Used in Support Vector Machine(SVM).



![math](https://render.githubusercontent.com/render/math?math=max%280%2C%201%20-%20y%20%5Ccdot%20%5Chat%7By%7D%29)



#### KL/JS divergence



![math](https://render.githubusercontent.com/render/math?math=KL%28%5Chat%7By%7D%20%7C%7C%20y%29%20%3D%20%5Csum_%7Bc%3D1%7D%5E%7BM%7D%5Chat%7By%7D_c%20%5Clog%7B%5Cfrac%7B%5Chat%7By%7D_c%7D%7By_c%7D%7D)





![math](https://render.githubusercontent.com/render/math?math=JS%28%5Chat%7By%7D%20%7C%7C%20y%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%28KL%28y%7C%7C%5Cfrac%7By%2B%5Chat%7By%7D%7D%7B2%7D%29%20%2B%20KL%28%5Chat%7By%7D%7C%7C%5Cfrac%7By%2B%5Chat%7By%7D%7D%7B2%7D%29%29)



### Regularization

The  ![math](https://render.githubusercontent.com/render/math?math=Error)  below can be any of the above loss.

#### L1 regularization

A regression model that uses L1 regularization technique is called *Lasso Regression*.



![math](https://render.githubusercontent.com/render/math?math=Loss%20%3D%20Error%28Y%20-%20%5Cwidehat%7BY%7D%29%20%2B%20%5Clambda%20%5Csum_1%5En%20%7Cw_i%7C)



#### L2 regularization

A regression model that uses L1 regularization technique is called *Ridge Regression*.



![math](https://render.githubusercontent.com/render/math?math=Loss%20%3D%20Error%28Y%20-%20%5Cwidehat%7BY%7D%29%20%2B%20%20%5Clambda%20%5Csum_1%5En%20w_i%5E%7B2%7D)



## Metrics

Some of them overlaps with loss, like MAE, KL-divergence.

### Classification

#### Accuracy, Precision, Recall, F1



![math](https://render.githubusercontent.com/render/math?math=Accuracy%20%3D%20%5Cfrac%7BTP%2BTF%7D%7BTP%2BTF%2BFP%2BFN%7D)





![math](https://render.githubusercontent.com/render/math?math=Precision%20%3D%20%5Cfrac%7BTP%7D%7BTP%2BFP%7D)





![math](https://render.githubusercontent.com/render/math?math=Recall%20%3D%20%5Cfrac%7BTP%7D%7BTP%2BFN%7D)





![math](https://render.githubusercontent.com/render/math?math=F1%20%3D%20%5Cfrac%7B2%2APrecision%2ARecall%7D%7BPrecision%2BRecall%7D%20%3D%20%5Cfrac%7B2%2ATP%7D%7B2%2ATP%2BFP%2BFN%7D)



```
Accuracy = \frac{TP+TF}{TP+TF+FP+FN}
Precision = \frac{TP}{TP+FP}
Recall = \frac{TP}{TP+FN}
F1 = \frac{2*Precision*Recall}{Precision+Recall} = \frac{2*TP}{2*TP+FP+FN}
```

#### Sensitivity, Specificity and AUC



![math](https://render.githubusercontent.com/render/math?math=Sensitivity%20%3D%20Recall%20%3D%20%5Cfrac%7BTP%7D%7BTP%2BFN%7D)





![math](https://render.githubusercontent.com/render/math?math=Specificity%20%3D%20%5Cfrac%7BTN%7D%7BFP%2BTN%7D)



```
Sensitivity = Recall = \frac{TP}{TP+FN}
Specificity = \frac{TN}{FP+TN}
```

AUC is calculated as the Area Under the  ![math](https://render.githubusercontent.com/render/math?math=Sensitivity) (TPR)- ![math](https://render.githubusercontent.com/render/math?math=%281-Specificity%29) (FPR) Curve.

### Regression

MAE, MSE, equation [above](#loss).

### Clustering

#### (Normalized) Mutual Information (NMI)

The Mutual Information is a measure of the similarity between two labels of the same data. Where  ![math](https://render.githubusercontent.com/render/math?math=%7CU_i%7C)  is the number of the samples in cluster  ![math](https://render.githubusercontent.com/render/math?math=U_i)  and  ![math](https://render.githubusercontent.com/render/math?math=%7CV_i%7C)  is the number of the samples in cluster  ![math](https://render.githubusercontent.com/render/math?math=V_i)  , the Mutual Information between cluster  ![math](https://render.githubusercontent.com/render/math?math=U)  and  ![math](https://render.githubusercontent.com/render/math?math=V)  is given as:



![math](https://render.githubusercontent.com/render/math?math=MI%28U%2CV%29%3D%5Csum_%7Bi%3D1%7D%5E%7B%7CU%7C%7D%20%5Csum_%7Bj%3D1%7D%5E%7B%7CV%7C%7D%20%5Cfrac%7B%7CU_i%5Ccap%20V_j%7C%7D%7BN%7D%0A%5Clog%5Cfrac%7BN%7CU_i%20%5Ccap%20V_j%7C%7D%7B%7CU_i%7C%7CV_j%7C%7D)



```
MI(U,V)=\sum_{i=1}^{|U|} \sum_{j=1}^{|V|} \frac{|U_i\cap V_j|}{N}
\log\frac{N|U_i \cap V_j|}{|U_i||V_j|}
```

Normalized Mutual Information (NMI) is a normalization of the Mutual Information (MI) score to scale the results between 0 (no mutual information) and 1 (perfect correlation). In this function, mutual information is normalized by some generalized mean of H(labels_true) and H(labels_pred)), See [wiki](https://en.wikipedia.org/wiki/Mutual_information#Normalized_variants).

Skip [RI, ARI](https://en.wikipedia.org/wiki/Rand_index) for complexity.

Also skip metrics for related tasks (e.g. modularity for community detection[graph clustering], coherence score for topic modeling[soft clustering]).


### Ranking

Skip nDCG (Normalized Discounted Cumulative Gain) for its complexity.

#### (Mean) Average Precision(MAP)

Average Precision is calculated as:



![math](https://render.githubusercontent.com/render/math?math=%5Ctext%7BAP%7D%20%3D%20%5Csum_n%20%28R_n%20-%20R_%7Bn-1%7D%29%20P_n)



```
\text{AP} = \sum_n (R_n - R_{n-1}) P_n
```

where  ![math](https://render.githubusercontent.com/render/math?math=R_n)  and  ![math](https://render.githubusercontent.com/render/math?math=P_n)  are the precision and recall at the  ![math](https://render.githubusercontent.com/render/math?math=n) th threshold,

MAP is the mean of AP over all the queries.

### Similarity/Relevance

#### Cosine



![math](https://render.githubusercontent.com/render/math?math=Cosine%28x%2Cy%29%20%3D%20%5Cfrac%7Bx%20%5Ccdot%20y%7D%7B%7Cx%7C%7Cy%7C%7D)



```
Cosine(x,y) = \frac{x \cdot y}{|x||y|}
```

#### Jaccard

Similarity of two sets  ![math](https://render.githubusercontent.com/render/math?math=U)  and  ![math](https://render.githubusercontent.com/render/math?math=V) .



![math](https://render.githubusercontent.com/render/math?math=Jaccard%28U%2CV%29%20%3D%20%5Cfrac%7B%7CU%20%5Ccap%20V%7C%7D%7B%7CU%20%5Ccup%20V%7C%7D)



```
Jaccard(U,V) = \frac{|U \cap V|}{|U \cup V|}
```

#### Pointwise Mutual Information(PMI)

Relevance of two events  ![math](https://render.githubusercontent.com/render/math?math=x)  and  ![math](https://render.githubusercontent.com/render/math?math=y) .



![math](https://render.githubusercontent.com/render/math?math=PMI%28x%3By%29%20%3D%20%5Clog%7B%5Cfrac%7Bp%28x%2Cy%29%7D%7Bp%28x%29p%28y%29%7D%7D)



```
PMI(x;y) = \log{\frac{p(x,y)}{p(x)p(y)}}
```

For example,  ![math](https://render.githubusercontent.com/render/math?math=p%28x%29)  and  ![math](https://render.githubusercontent.com/render/math?math=p%28y%29)  is the frequency of word  ![math](https://render.githubusercontent.com/render/math?math=x)  and  ![math](https://render.githubusercontent.com/render/math?math=y)  appearing in corpus and  ![math](https://render.githubusercontent.com/render/math?math=p%28x%2Cy%29)  is the frequency of the co-occurrence of the two.

## Notes

This repository now only contains simple equations for ML. They are mainly about deep learning and NLP now due to personal research interests.

For time issues, elegant equations in traditional ML approaches like SVM, SVD, PCA, LDA are not included yet.

Moreover, there is a trend towards more complex metrics, which have to be calculated with complicated program (e.g. BLEU, ROUGE, METEOR), iterative algorithms (e.g. PageRank), optimization (e.g. Earth Mover Distance), or even learning based (e.g. BERTScore). They thus cannot be described using simple equations.

## Reference

[Pytorch Documentation](https://pytorch.org/docs/master/nn.html)

[Scikit-learn Documentation](https://scikit-learn.org/stable/modules/classes.html)

[Machine Learning Glossary](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)

[Wikipedia](https://en.wikipedia.org/)

Thanks for [a-rodin's solution](https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b) to show Latex in Github markdown, which I have wrapped into `latex2pic.py`.