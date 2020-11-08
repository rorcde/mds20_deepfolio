<h2 align="center"> Existing implementations </h2>

<h4 align="center"> Content </h4>

<p align="center">
  <a href="#deeplob">DeepLOB</a> •
  <a href="#unipoint">UNIPoint</a> •
  <a href="#hawkes-transformer">Hawkes Transformer</a>
</p>

## DeepLOB

Folder **deeplob** contains <a href="https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books" target="_blank">original Keras implementation</a> of Zhang et al. work. Since the project does not focus on DeepLOB (it is already implemented in the DeepFolio paper and we do not need to reimplement the model for this project), this implementation is listed purely for consistency.

## UNIPoint

## Hawkes Transformer

Folder **hawkes-transformer** contains the <a href="https://github.com/SimiaoZuo/Transformer-Hawkes-Process" target="_blank">original implementation</a> of Zuo et al. work. The network architecture is presented on the image below (taken from the original paper):

<p align="center">
  <img width="350" height="300" src="https://github.com/rodrigorivera/mds20_deepfolio/blob/main/images/THP-arch.png">
</p>

The input event sequence passes through two encoding modules - its timestamps are encoded using temporal encoding procedure, which is essentially the same as the positional encoding proposed in the original <a href="https://arxiv.org/abs/1706.03762" target="_blank">Transformer</a> paper; additionally, learned neural embeddings are used for the event types. Encoded input is then processed through N stacked Transformer encoder layers, each of which consists of multiheaded attention module and two layer position-wise feed-forward neural network.

**Implementation details**:
* Authors implement the temporal encoding by directly rewriting the formula with PyTorch tensors, while learned embeddings for events are created through PyTorch's Embedding layers.
* Main bulk of the codebase is related to implementation of the encoding layers of Transformer architecture, where authors implement Multihead Attention and Feedforward layers from scratch.
* Multihead attention is implemented exactly the same as in the original Transformer paper, its structure can be seen on the following illustration (taken from <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a>):
<p align="center">
  <img width="300" height="450" src="https://github.com/rodrigorivera/mds20_deepfolio/blob/main/images/multihead_attention.png">
</p>
where Scaled Dot-Product Attention (taken from the same paper):
<p align="center">
  <img width="200" height="400" src="https://github.com/rodrigorivera/mds20_deepfolio/blob/main/images/scaled_dot-product-attention.png">
</p>
Q, K, V are query, key and value matrices, which are obtained by multiplying the encoded input with respective learnable weight matrices.

* Result of the Multihead Attention module is fed to the Position FeedForward module. The latter is implemented as a residual block with two linear layers with GELU (Gaussian Error Linear Unit) activation function for the first layer and no activation for the second one, also dropout is applied after each layer. Additionally, layer normalization can be applied either at the beginning or in the end.
* The problem of "peaking into the future" is solved by masking, where all future positions are set to "inf".
* Optionally, authors suggest using additional LSTM layers for the Transformer output. This approach is inspired by the fact that this helps to get better results in other sequential data problems, such as language modelling.
* Negative log-likelihood is used as a loss function for the sequence, where non-event part of the log-likelihood is computed using Monte Carlo integration.
* Additionally, authors calculate event prediction loss using either Cross Entropy or Label Smoothing loss functions. Time prediction loss is handled straightforwardly by applying Mean Squared Error (MSE) loss. Since the time prediction loss is usually large, authors scaled it by dividing on 100 to stabilize training.
* Sum of all three loss functions is used for calculation of gradients and backpropagation.

**What will be done differently**:
* Since it looks like all the Transformer modules have the same structure as in the original paper, then in our version we will use the PyTorch implementations of the Transformer blocks such as <a href="https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html" target="_blank">Multihead attention</a>. This approach will likely result in better speed-wise performance, as PyTorch layers are highly optimized.
* Calculation of negative log likelihood loss requires calculation of the integral - aside from Monte Carlo, it can be estimated through numerical integration (e.g. linear interpolation or trapezoidal rule).
* LSTM layers will most likely be ommitted as they are not mentioned in the paper itself and there is no evidence of drastic quality improvements.
* It is also unclear why GELU activation is used, while in the paper they stated that they used ReLU, hence both variants will be tried.
* Work <a href="https://arxiv.org/pdf/1603.05027.pdf" target="_blank">Identity Mappings in Deep Residual Networks</a> explores different approaches towards activation functions for the residual blocks. Although, the performance is usually very similar, taking some ideas from there could help to further stabilize training and improve results.
