# Classical ML Equations in LaTeX

A collection of classical ML equations in Latex . Some of them are provided with simple notes and paper link. Hopes to help writings such as papers and blogs.

Better viewed at https://blmoistawinde.github.io/ml_equations_latex/

- [Classical ML Equations in LaTeX](#classical-ml-equations-in-latex)
  - [Model](#model)
    - [RNNs(LSTM, GRU)](#rnnslstm-gru)
    - [Attentional Seq2seq](#attentional-seq2seq)
      - [Bahdanau Attention](#bahdanau-attention)
      - [Luong(Dot-Product) Attention](#luongdot-product-attention)
    - [Transformer](#transformer)
      - [Scaled Dot-Product attention](#scaled-dot-product-attention)
      - [Multi-head attention](#multi-head-attention)
    - [Generative Adversarial Networks(GAN)](#generative-adversarial-networksgan)
      - [Minmax game objective](#minmax-game-objective)
    - [Variational Auto-Encoder(VAE)](#variational-auto-encodervae)
      - [Reparameterization trick](#reparameterization-trick)
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

encoder hidden state $h_t$ at time step $t$
$$h_t = RNN_{enc}(x_t, h_{t-1})$$

decoder hidden state $s_t$ at time step $t$

$$s_t = RNN_{dec}(y_t, s_{t-1})$$

```
h_t = RNN_{enc}(x_t, h_{t-1})
s_t = RNN_{dec}(y_t, s_{t-1})
```

The $RNN_{enc}$, $RNN_{dec}$ are usually either 

- LSTM (paper: [Long short-term memory](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf)) 

-  GRU (paper: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)).

### Attentional Seq2seq

The attention weight $\alpha_{ij}$, the $i$th decoder step over the $j$th encoder step, resulting in context vector $c_i$

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij}h_j$$

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

$$
e_{ik} = a(s_{i-1}, h_j)
$$

```
c_i = \sum_{j=1}^{T_x} \alpha_{ij}h_j

\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}

e_{ik} = a(s_{i-1}, h_j)
```

$a$ is an specific attention function, which can be

#### Bahdanau Attention

Paper: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

$$e_{ik} = v^T tanh(W[s_{i-1}; h_j])$$

```
e_{ik} = v^T tanh(W[s_{i-1}; h_j])
```

#### Luong(Dot-Product) Attention

Paper: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

If $s_i$ and $h_j$ has same number of dimension.

$$e_{ik} = s_{i-1}^T h_j$$

otherwise

$$e_{ik} = s_{i-1}^T W h_j$$

```
e_{ik} = s_{i-1}^T h_j

e_{ik} = s_{i-1}^T W h_j
```

Finally, the output $o_i$ is produced by:

$$s_t = tanh(W[s_{t-1};y_t;c_t])$$
$$o_t = softmax(Vs_t)$$

```
s_t = tanh(W[s_{t-1};y_t;c_t])
o_t = softmax(Vs_t)
```

### Transformer

Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

#### Scaled Dot-Product attention

$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

```
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
```

where $\sqrt{d_k}$ is the dimension of the key vector $k$ and query vector $q$ .

#### Multi-head attention

$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$$

where 
$$
head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
$$

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
```

### Generative Adversarial Networks(GAN)

Paper: [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

#### Minmax game objective

$$
\min_{G}\max_{D}\mathbb{E}_{x\sim p_{\text{data}}(x)}[\log{D(x)}] +  \mathbb{E}_{z\sim p_{\text{generated}}(z)}[1 - \log{D(G(z))}]
$$

```
\min_{G}\max_{D}\mathbb{E}_{x\sim p_{\text{data}}(x)}[\log{D(x)}] +  \mathbb{E}_{z\sim p_{\text{generated}}(z)}[1 - \log{D(G(z))}]
```


### Variational Auto-Encoder(VAE)

Paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

#### Reparameterization trick

To produce a latent variable z such that $z \sim q_{\mu, \sigma}(z) = \mathcal{N}(\mu, \sigma^2)$, we sample $\epsilon \sim \mathcal{N}(0,1)$, than z is produced by 

$$z = \mu + \epsilon \cdot \sigma$$

```
z \sim q_{\mu, \sigma}(z) = \mathcal{N}(\mu, \sigma^2)
\epsilon \sim \mathcal{N}(0,1)
z = \mu + \epsilon \cdot \sigma
```

Above is for 1-D case. For a multi-dimensional (vector) case we use:

$$
\vec{\epsilon} \sim \mathcal{N}(0, \textbf{I})
$$

$$
\vec{z} \sim \mathcal{N}(\vec{\mu}, \sigma^2 \textbf{I})
$$

```
\epsilon \sim \mathcal{N}(0, \textbf{I})
\vec{z} \sim \mathcal{N}(\vec{\mu}, \sigma^2 \textbf{I})
```

## Activations

### Sigmoid
Related to *Logistic Regression*. For single-label/multi-label binary classification.

$$\sigma(z) = \frac{1} {1 + e^{-z}}$$

```
\sigma(z) = \frac{1} {1 + e^{-z}}
```

### Softmax

For multi-class single label classification.

$$\sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K$$

```
\sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K
```

### Relu

$$Relu(z) = max(0, z)$$

```
Relu(z) = max(0, z)
```

## Loss

### Regression

Below $x$ and $y$ are $D$ dimensional vectors, and $x_i$ denotes the value on the $i$th dimension of $x$.

#### Mean Absolute Error(MAE) 

$$\sum_{i=1}^{D}|x_i-y_i|$$

```
\sum_{i=1}^{D}|x_i-y_i|
```

#### Mean Squared Error(MSE)

$$\sum_{i=1}^{D}(x_i-y_i)^2$$

```
\sum_{i=1}^{D}(x_i-y_i)^2
```

#### Huber loss

It’s less sensitive to outliers than the MSE as it treats error as square only inside an interval.

$$
L_{\delta}=
    \left\{\begin{matrix}
        \frac{1}{2}(y - \hat{y})^{2} & if \left | (y - \hat{y})  \right | < \delta\\
        \delta ((y - \hat{y}) - \frac1 2 \delta) & otherwise
    \end{matrix}\right.
$$

```
L_{\delta}=
    \left\{\begin{matrix}
        \frac{1}{2}(y - \hat{y})^{2} & if \left | (y - \hat{y})  \right | < \delta\\
        \delta ((y - \hat{y}) - \frac1 2 \delta) & otherwise
    \end{matrix}\right.
```

### Classification

#### Cross Entropy

- In binary classification, where the number of classes $M$ equals 2, Binary Cross-Entropy(BCE) can be calculated as:

$$-{(y\log(p) + (1 - y)\log(1 - p))}$$

- If $M > 2$ (i.e. multiclass classification), we calculate a separate loss for each class label per observation and sum the result.

$$-\sum_{c=1}^My_{o,c}\log(p_{o,c})$$

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

$$NLL(y) = -{\log(p(y))}$$

Minimizing negative loglikelihood 

$$\min_{\theta} \sum_y {-\log(p(y;\theta))}$$


is equivalent to Maximum Likelihood Estimation(MLE).


$$\max_{\theta} \prod_y p(y;\theta)$$

Here $p(y)$ is a *scaler* instead of *vector*. It is the value of the single dimension where the ground truth $y$ lies. It is thus equivalent to cross entropy (See [wiki](https://en.wikipedia.org/wiki/Cross_entropy)).\

```
NLL(y) = -{\log(p(y))}

\min_{\theta} \sum_y {-\log(p(y;\theta))}

\max_{\theta} \prod_y p(y;\theta)
```

#### Hinge loss

Used in Support Vector Machine(SVM).

$$max(0, 1 - y \cdot \hat{y})$$

```
max(0, 1 - y \cdot \hat{y})
```

#### KL/JS divergence

$$KL(\hat{y} || y) = \sum_{c=1}^{M}\hat{y}_c \log{\frac{\hat{y}_c}{y_c}}$$

$$JS(\hat{y} || y) = \frac{1}{2}(KL(y||\frac{y+\hat{y}}{2}) + KL(\hat{y}||\frac{y+\hat{y}}{2}))$$

```
KL(\hat{y} || y) = \sum_{c=1}^{M}\hat{y}_c \log{\frac{\hat{y}_c}{y_c}}

JS(\hat{y} || y) = \frac{1}{2}(KL(y||\frac{y+\hat{y}}{2}) + KL(\hat{y}||\frac{y+\hat{y}}{2}))
```

### Regularization

The $Error$ below can be any of the above loss.

#### L1 regularization

A regression model that uses L1 regularization technique is called *Lasso Regression*.

$$Loss = Error(Y - \widehat{Y}) + \lambda \sum_1^n |w_i|$$

```
Loss = Error(Y - \widehat{Y}) + \lambda \sum_1^n |w_i|
```

#### L2 regularization

A regression model that uses L1 regularization technique is called *Ridge Regression*.

$$Loss = Error(Y - \widehat{Y}) +  \lambda \sum_1^n w_i^{2}$$

```
Loss = Error(Y - \widehat{Y}) +  \lambda \sum_1^n w_i^{2}
```

## Metrics

Some of them overlaps with loss, like MAE, KL-divergence.

### Classification

#### Accuracy, Precision, Recall, F1

$$Accuracy = \frac{TP+TF}{TP+TF+FP+FN}$$

$$Precision = \frac{TP}{TP+FP}$$

$$Recall = \frac{TP}{TP+FN}$$

$$F1 = \frac{2*Precision*Recall}{Precision+Recall} = \frac{2*TP}{2*TP+FP+FN}$$

```
Accuracy = \frac{TP+TF}{TP+TF+FP+FN}
Precision = \frac{TP}{TP+FP}
Recall = \frac{TP}{TP+FN}
F1 = \frac{2*Precision*Recall}{Precision+Recall} = \frac{2*TP}{2*TP+FP+FN}
```

#### Sensitivity, Specificity and AUC

$$Sensitivity = Recall = \frac{TP}{TP+FN}$$

$$Specificity = \frac{TN}{FP+TN}$$

```
Sensitivity = Recall = \frac{TP}{TP+FN}
Specificity = \frac{TN}{FP+TN}
```

AUC is calculated as the Area Under the $Sensitivity$(TPR)-$(1-Specificity)$(FPR) Curve.

### Regression

MAE, MSE, equation [above](#loss).

### Clustering

#### (Normalized) Mutual Information (NMI)

The Mutual Information is a measure of the similarity between two labels of the same data. Where $|U_i|$ is the number of the samples in cluster $U_i$ and $|V_i|$ is the number of the samples in cluster $V_i$ , the Mutual Information between cluster $U$ and $V$ is given as:

$$
MI(U,V)=\sum_{i=1}^{|U|} \sum_{j=1}^{|V|} \frac{|U_i\cap V_j|}{N}
\log\frac{N|U_i \cap V_j|}{|U_i||V_j|}
$$

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

$$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$$

```
\text{AP} = \sum_n (R_n - R_{n-1}) P_n
```

where $R_n$ and $P_n$ are the precision and recall at the $n$th threshold,

MAP is the mean of AP over all the queries.

### Similarity/Relevance

#### Cosine

$$Cosine(x,y) = \frac{x \cdot y}{|x||y|}$$

```
Cosine(x,y) = \frac{x \cdot y}{|x||y|}
```

#### Jaccard

Similarity of two sets $U$ and $V$.

$$Jaccard(U,V) = \frac{|U \cap V|}{|U \cup V|}$$

```
Jaccard(U,V) = \frac{|U \cap V|}{|U \cup V|}
```

#### Pointwise Mutual Information(PMI)

Relevance of two events $x$ and $y$.

$$PMI(x;y) = \log{\frac{p(x,y)}{p(x)p(y)}}$$

```
PMI(x;y) = \log{\frac{p(x,y)}{p(x)p(y)}}
```

For example, $p(x)$ and $p(y)$ is the frequency of word $x$ and $y$ appearing in corpus and $p(x,y)$ is the frequency of the co-occurrence of the two.

## Notes

This repository now only contains simple equations for ML. They are mainly about deep learning and NLP now due to personal research interests.

For time issues, elegant equations in traditional ML approaches like SVM, SVD, PCA, LDA are not included yet.

Moreover, there is a trend towards more complex metrics, which have to be calculated with complicated program (e.g. BLEU, ROUGE, METEOR), iterative algorithms (e.g. PageRank), optimization (e.g. Earth Mover Distance), or even learning based (e.g. BERTScore). They thus cannot be described using simple equations.

## Reference

[Pytorch Documentation](https://pytorch.org/docs/master/nn.html)

[Scikit-learn Documentation](https://scikit-learn.org/stable/modules/classes.html)

[Machine Learning Glossary](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)

[Wikipedia](https://en.wikipedia.org/)

https://blog.floydhub.com/gans-story-so-far/

https://ermongroup.github.io/cs228-notes/extras/vae/

Thanks for [a-rodin's solution](https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b) to show Latex in Github markdown, which I have wrapped into `latex2pic.py`.