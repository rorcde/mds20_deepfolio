<h2 align="center"> Transformer Hawkes Process </h2>

<h4 align="center"> Content </h4>

<p align="center">
  <a href="#architecture">Architecture</a> •
  <a href="#loss-function">Loss function</a> •
  <a href="#implementation-details">Implementation details</a> •
  <a href="#files">Files</a> •
  <a href="#model-weights">Model weights</a>
</p>

## Architecture

<p align="center">
  <img width="350" height="300" src="https://github.com/rodrigorivera/mds20_deepfolio/blob/main/images/THP-arch.png">
</p>

Input event sequence consists of times and event types. First, it passes through two embedding layers - deterministic temporal encoding is used for the
time sequences, while learnable neural embedding is used for the event types:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?X&space;=&space;Z&space;&plus;&space;U" />
</p>
where <a href=""><img src="https://latex.codecogs.com/gif.latex?X" /></a> - input to the Transformer model,
<a href=""><img src="https://latex.codecogs.com/gif.latex?Z&space;\in&space;\mathbb{R}^{S&space;\times&space;E}" /></a> - temporal encoding,
<a href=""><img src="https://latex.codecogs.com/gif.latex?U&space;\in&space;\mathbb{R}^{S&space;\times&space;E}" /></a> - event types embedding,
<a href=""><img src="https://latex.codecogs.com/gif.latex?S" /></a> - sequence length, 
<a href=""><img src="https://latex.codecogs.com/gif.latex?E" /></a> - embedding dimension.

Secondly, <a href=""><img src="https://latex.codecogs.com/gif.latex?X" /></a> passes through N stacked transformer encoder layers, consisting of multihead attention
and position-wise feed forward layers. Architecture of transformer encoder layers is the same as in the <a href="https://arxiv.org/abs/1706.03762" > original </a> paper.
As a result, hidden representations of the input are obtained.

After that, hidden states are used to produce the conditional intensity function:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\lambda&space;(t|H_t)&space;=&space;\sum\limits_{k=1}^K&space;\lambda_k&space;(t|H_t)" />
</p>
where <a href=""><img src="https://latex.codecogs.com/gif.latex?k" /></a> - event types, 
<a href=""><img src="https://latex.codecogs.com/gif.latex?t" /></a> - time,
<a href=""><img src="https://latex.codecogs.com/gif.latex?H_t&space;=&space;\{&space;(t_j,&space;k_j)&space;:&space;t_j&space;<&space;t&space;\}" /></a> - history up to time <a href=""><img src="https://latex.codecogs.com/gif.latex?t" /></a>,
<a href=""><img src="https://latex.codecogs.com/gif.latex?\lambda_k&space;(t|H_t)" /></a> - type-specific intensity function, which can be defined as:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\lambda_k&space;(t|H_t)&space;=&space;f_k&space;\left(&space;\alpha_k&space;\frac{t&space;-&space;t_j}{t}&space;&plus;&space;w_k^T&space;h(t_j)&space;&plus;&space;b_k&space;\right)" />
</p>
where <a href=""><img src="https://latex.codecogs.com/gif.latex?f_k" /></a> - softplus function,
<a href=""><img src="https://latex.codecogs.com/gif.latex?\alpha_k" /></a> - constant, which controls interpolation strength,
<a href=""><img src="https://latex.codecogs.com/gif.latex?t&space;\in&space;[t_j;&space;t_{j&plus;1})" /></a>. Rest of the sum is the output of a linear layer, input
to which are the hidden representations from the Transformer.

Finally, the conditional intensity function can be used to obtain predictions for the next time stamp and event type in the sequence:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?p(t|H_j)&space;=&space;\lambda&space;(t|H_t)&space;\exp&space;\left(&space;-\int\limits_{t_{j-1}}^t&space;\lambda&space;(\tau|H_{\tau})&space;d\tau&space;\right)" title="p(t|H_j) = \lambda (t|H_t) \exp \left( -\int\limits_{t_{j-1}}^t \lambda (\tau|H_{\tau}) d\tau \right)" />
</p>
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\hat{t}_{j&plus;1}&space;=&space;\int\limits_{t_j}^{\infty}&space;t&space;p(t|H_j)&space;dt" title="\hat{t}_{j+1} = \int\limits_{t_j}^{\infty} t p(t|H_j) dt" />
</p>
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\hat{k}_{j&plus;1}&space;=&space;\underset{k}{\text{argmax}}&space;\frac{\lambda_k&space;(t_{j&plus;1}|H_{j&plus;1})}{\lambda&space;(t_{j&plus;1}|H_{j&plus;1})}" title="\hat{k}_{j+1} = \underset{k}{\text{argmax}} \frac{\lambda_k (t_{j+1}|H_{j+1})}{\lambda (t_{j+1}|H_{j+1})}" />
</p>
where <a href=""><img src="https://latex.codecogs.com/gif.latex?\hat{t}" /></a> - time prediction,
<a href=""><img src="https://latex.codecogs.com/gif.latex?\hat{k}" /></a></a> - event type prediction.

Another way is to use two separate linear layers (with no bias) to make time and event type predictions, which can be taught by combining the negative log likelihood loss
for the sequence with regression and classification losses for predictions. This loss function will be shown in the next section.

## Loss function

To train this neural network, standard negative log likelihood for the sequence is used. Since we employ two linear layers for predictions, then we also need additional
terms for time regression and event type classification. Hence, the final loss function has the following form:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?L&space;(\lambda,&space;\hat{t},&space;\hat{k})&space;=&space;\int\limits_{t_1}^{t_S}&space;\lambda&space;(t|H_t)&space;dt&space;-&space;\sum\limits_{j=1}^S&space;\log&space;\lambda(t_j|H_j)&space;&plus;&space;\sum\limits_{j=2}^S&space;\left[&space;\left(&space;\hat{t}_{j-1}&space;-&space;t_j&space;\right)^2&space;-&space;k_{j}&space;\log&space;\hat{k}_{j-1}&space;\right]" title="L (\lambda, \hat{t}, \hat{k}) = \int\limits_{t_1}^{t_S} \lambda (t|H_t) dt - \sum\limits_{j=1}^S \log \lambda(t_j|H_j) + \sum\limits_{j=2}^S \left[ \left( \hat{t}_{j-1} - t_j \right)^2 - k_{j} \log \hat{k}_{j-1} \right]" />
</p>
This loss function is minimized to train the described network.

## Implementation details

List of key differences compared to the <a href="https://github.com/SimiaoZuo/Transformer-Hawkes-Process">authors' implementation</a>:
* PyTorch's *nn.TransformerEncoder* was used for the transformer encoder layers;
* Addition of temporal encoding is performed only once instead of each time per layer;
* <a href="https://en.wikipedia.org/wiki/LogSumExp">LogSumExp</a> trick was used in the temporal encoding for better numerical stability;
* No RNN layers since we found that on our dataset it does not help;
* ReLU activation in the feed-forward network;
* Additional vectorizations in some parts, combined with using PyTorch layers for Transformer, made our model almost 3 times faster (tested on the financial dataset
from NHP): <a href="https://github.com/rodrigorivera/mds20_deepfolio/blob/main/images/our_thp.png">ours</a>,
<a href="https://github.com/rodrigorivera/mds20_deepfolio/blob/main/images/original_thp.png">original</a>.

## Files

* **model.py** - contains THP architecture (with two linear layers as predictors),
* **train.py** - contains train functions,
* **dataset_wrapper.py** - contains code for the dataset wrapper which feeds data in a required manner to the network,
* **utils.py** - utility functions,
* **THP_experiment_1.ipynb** - contains experiment on the financial data from NHP paper

## Model weights

Model weights can be obtained from this <a href="https://drive.google.com/drive/folders/1bzcug2lOx7qUVpq1bUJSbRTSzbt_MNgK?usp=sharing">Google drive</a>.
