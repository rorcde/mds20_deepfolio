<h2 align="center"> <a href="https://arxiv.org/pdf/1612.09328.pdf">Neural Hawkes Process</a> </h2>

<h4 align="center"> Content </h4>

<p align="center">
  <a href="#architecture">Architecture</a> •
  <a href="#loss-function">Loss function</a> •
  <a href="#implementation-details">Implementation details</a> •
  <a href="#files">Files</a> 
</p>

## Architecture

<p align="center">
<img src="https://github.com/rodrigorivera/mds20_deepfolio/blob/main/images/NHP_model.png" />
</p>


At the input of the model we have  <a href=""><img src="https://latex.codecogs.com/svg.latex?(k_i,%20t_i)" /></a>, 
where  <a href=""><img src="https://latex.codecogs.com/svg.latex?k_i" /></a> - type of event 
and <a href=""><img src="https://latex.codecogs.com/svg.latex?t_i" /></a> - time of event occurrence, 
and we are interested in modeling <a href=""><img src="https://latex.codecogs.com/svg.latex?\lambda_k%20(t)" /></a> - intensity function, 
wich "jumps" at each new event and then increases or discreases towards some value.

Model architecture is similar to classic LSTM - the dymanics of <a href=""><img src="https://latex.codecogs.com/svg.latex?\lambda_k%20(t)" /></a> 
is controlled by hidden state vector - <a href=""><img src="https://latex.codecogs.com/svg.latex?\bold{h}(t)%20\in%20(-1,1)^D" /></a>,
wich is depend on memory cells vector - <a href=""><img src="https://latex.codecogs.com/svg.latex?\bold{c}(t)%20\in%20\mathbb{R}^D" /></a>.
The difference with LSTM is that after event happened each memory cell <a href=""><img src="https://latex.codecogs.com/svg.latex?c" /></a>
exponentially decays at some rate <a href=""><img src="https://latex.codecogs.com/svg.latex?\delta" /></a>
toward some steady-state value <a href=""><img src="https://latex.codecogs.com/svg.latex?\bar{c}" /></a>.

At each time <a href=""><img src="https://latex.codecogs.com/svg.latex?t" /></a> 
we obtain the intensity <a href=""><img src="https://latex.codecogs.com/svg.latex?\lambda_k%20(t)" /></a>
by the following equations:

<p align="center">
<img src="https://github.com/rodrigorivera/mds20_deepfolio/blob/main/images/intens_eq.png" />
</p>

In this architecture <a href=""><img src="https://latex.codecogs.com/svg.latex?\bold{h}(t)" /></a>
summarizes not only the past event sequence <a href=""><img src="https://latex.codecogs.com/svg.latex?(k_1,%20...,%20k_{i-i})" /></a>
but also the interarrival times <a href=""><img src="https://latex.codecogs.com/svg.latex?(t_1%20-%200,%20t_2%20-%20t_1,%20...,%20t%20-%20t_{i-1})" /></a>
and when in time <a href=""><img src="https://latex.codecogs.com/svg.latex?t_i" /></a>
event <a href=""><img src="https://latex.codecogs.com/svg.latex?k_i" /></a> occures, 
model reads <a href=""><img src="https://latex.codecogs.com/svg.latex?(k_i,%20t_i)" /></a>
and updates the current (decayed) hidden cells <a href=""><img src="https://latex.codecogs.com/svg.latex?\bold{c}(t)" /></a>
to new initial values <a href=""><img src="https://latex.codecogs.com/svg.latex?\bold{c}_{i-1}" /></a>,
based on the current (decayed) hidden state <a href=""><img src="https://latex.codecogs.com/svg.latex?\bold{h}(t)" /></a>,
according to the update formulas below:

<p align="center">
<img src="https://github.com/rodrigorivera/mds20_deepfolio/blob/main/images/CTLSTM.png" />
</p>

The input vector <a href=""><img src="https://latex.codecogs.com/svg.latex?k_i%20\in%20\{0,%201\}^K" /></a>
a one-hot encoding of the new event <a href=""><img src="https://latex.codecogs.com/svg.latex?k_i" /></a>
with non-zero value only at the entry indexed by <a href=""><img src="https://latex.codecogs.com/svg.latex?k_i" /></a>.
Equations from the first column are similiar to classic LSTM, the updates do not depend on the “previous” hidden state from 
just after time <a href=""><img src="https://latex.codecogs.com/svg.latex?t_{i-1}" /></a>, but
rather its value <a href=""><img src="https://latex.codecogs.com/svg.latex?\bold{h}(t_{i})" /></a>,
after it has decayed.

Equations in the second column are new.  They define how in future, as
<a href=""><img src="https://latex.codecogs.com/svg.latex?t > t_{i}" /></a> increases, 
the elements of <a href=""><img src="https://latex.codecogs.com/svg.latex?\bold{c}(t)" /></a>
will continue to deterministically decay 
from <a href=""><img src="https://latex.codecogs.com/svg.latex?c_{i+1}" /></a>
toward targets <a href=""><img src="https://latex.codecogs.com/svg.latex?\bar{c}_{i+1}" /></a>
Specifically, <a href=""><img src="https://latex.codecogs.com/svg.latex?\bold{c}(t)" /></a>, 
will continue to control <a href=""><img src="https://latex.codecogs.com/svg.latex?\bold{h}(t)" /></a>
and thus <a href=""><img src="https://latex.codecogs.com/svg.latex?\lambda_k(t)" /></a>
according to this equation:

<p align="center">
<img src="https://github.com/rodrigorivera/mds20_deepfolio/blob/main/images/c(t).png" />
</p>

Finally, the intensity function can be used to obtain predictions for the next time stamp and event type in the sequence:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?p(t)&space;=&space;\lambda&space;(t)&space;\exp&space;\left(&space;-\int\limits_{t_{j-1}}^t&space;\lambda&space;(\tau)&space;d\tau&space;\right)" title="p(t|H_j) = \lambda (t) \exp \left( -\int\limits_{t_{j-1}}^t \lambda (\tau) d\tau \right)" />
</p>
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\hat{t}_{j&plus;1}&space;=&space;\int\limits_{t_j}^{\infty}&space;t&space;p(t)&space;dt" title="\hat{t}_{j+1} = \int\limits_{t_j}^{\infty} t p(t) dt" />
</p>
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\hat{k}_{j&plus;1}&space;=&space;\underset{k}{\text{argmax}}&space;\frac{\lambda_k&space;(t_{j&plus;1})}{\lambda&space;(t_{j&plus;1})}" title="\hat{k}_{j+1} = \underset{k}{\text{argmax}} \frac{\lambda_k (t_{j+1})}{\lambda (t_{j+1})}" />
</p>
where <a href=""><img src="https://latex.codecogs.com/gif.latex?\hat{t}" /></a> - time prediction,
<a href=""><img src="https://latex.codecogs.com/gif.latex?\hat{k}" /></a></a> - event type prediction.

Another way is to use two separate linear layers (with no bias) to make time and event type predictions, 
which can be taught by combining the negative log likelihood loss for the sequence with regression and classification losses for predictions. 
This loss function will be shown in the next section.

## Loss function

To train this neural network, standard negative log likelihood for the sequence was suggested by authors. 

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L&space;(\lambda)&space;=&space;\int\limits_{t_1}^{t_S}&space;\lambda&space;(t)&space;dt&space;-&space;\sum\limits_{j=1}^S&space;\log&space;\lambda(t_j)&space;"  />
</p>

However, since we employ two linear layers for predictions, we also tried to add to original loss two additional
terms for time regression and event type classification:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?L&space;(\lambda,&space;\hat{t},&space;\hat{k})&space;=&space;\int\limits_{t_1}^{t_S}&space;\lambda&space;(t)&space;dt&space;-&space;\sum\limits_{j=1}^S&space;\log&space;\lambda(t_j)&space;&plus;&space;\sum\limits_{j=2}^S&space;\left[&space;\left(&space;\hat{t}_{j-1}&space;-&space;t_j&space;\right)^2&space;-&space;k_{j}&space;\log&space;\hat{k}_{j-1}&space;\right]" title="L (\lambda, \hat{t}, \hat{k}) = \int\limits_{t_1}^{t_S} \lambda (t) dt - \sum\limits_{j=1}^S \log \lambda(t_j) + \sum\limits_{j=2}^S \left[ \left( \hat{t}_{j-1} - t_j \right)^2 - k_{j} \log \hat{k}_{j-1} \right]" />
</p>

## Implementation details

* Updates to LSTM states was done using *nn.Linear* layers with combination of corresponding activation functions:
<a href=""><img src="https://latex.codecogs.com/svg.latex?\sigma" /></a> - *torch.sigmoid*, 
<a href=""><img src="https://latex.codecogs.com/svg.latex?2\sigma" /></a> - *torch.tanh*, 
<a href=""><img src="https://latex.codecogs.com/svg.latex?f" /></a> - *F.softplus*, 

* Computation of log likelihood loss was done using Monte-Carlo method, according to authors suggestions in Appendix B1-B2
* In addition for prediction time and type of next event, instead of using probability function, two *nn.Linear* layers were added for this purpose, 
which seems to the better choice (according to target metrics).

## Files

* **model.py** - contains NHP architecture (with two linear layers as predictors),
* **train.py** - contains train functions,
* **DataWrapper.py.py** - contains code for the dataset wrapper which feeds data in a required manner to the network,
* **utils.py** - utility functions,
* **LOB_exp.ipynb** - contains training and evaluating code on LOB data
* folder **weights** - contains model weights

