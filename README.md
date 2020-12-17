<h1 align="center"> DeepFolio: Hawkes Transformer for LOB Data </h1>

<h4 align="center"> Team members: Aiusha Sangadiev, Kirill Stepanov, Kirill Bubenchikov, Andrey Poddubny </h4>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#dependencies">Dependencies</a> •
  <a href="#setup">Setup</a> •
  <a href="#content">Content</a>
</p>

## Description

Usually, when one deals with <a href="https://www.investopedia.com/terms/l/limitorderbook.asp" target="_blank"> Limit Order Book (LOB) </a> data, the common way is to treat it as time series or tabular data. This project investigates another possible approach, which is treatment of LOB data as long event sequences with two possible events (price going up or down), which allows one to use a rich mathematical apparatus developed for temporal point processes on LOB data.

We are going to consider three main models - <a href="https://arxiv.org/abs/1612.09328">Neural Hawkes Process</a> (baseline), <a href="https://arxiv.org/abs/2007.14082">UNIPoint</a>, and <a href="https://arxiv.org/abs/2002.09291">Transformer Hawkes Process</a>. The aforementioned models will be implemented from scratch, adapted towards usage on LOB data, tuned and tested on our self-collected limit order book dataset consisting of five tokens - Ethereum (ETH), Litecoin (LTC), EOSIO (EOS), Ripple (XRP), Binance coin (BNB). We intentionally skip the most popular crypto asset - Bitcoin (BTC), because we are interested in robustness of the models when dealing with data coming from less liquid markets. In addition to that, we also perform an out-of-sample test on Stellar coin (XLM) to see the generalization capability of the models and how they would react to an unknown coin being introduced, i.e. whether there are general features in the LOB event sequences that could exploited, opening access for e.g. transfer learning in the future.

## Dependencies

* Python 3
* <a href="https://hub.packtpub.com/python-data-stack/" target="_blank"> Python data stack </a>
* <a href="https://pytorch.org/" target="_target"> PyTorch </a>
* <a href="https://github.com/tqdm/tqdm" target="_target"> tqdm </a>

## Setup

Clone GitHub repository:

```
git clone https://github.com/rodrigorivera/mds20_deepfolio
```

Make sure that all dependencies are installed and run the respective notebooks, containing the experiments for each model.

## Content

This repository contains the codebase for the project, its structure is the following:
* **datasets** folder contains the code that was used to download and transform raw data into LOB, as well as all the required preprocessing of the data;
* **models** folder contains codebase for the models used in this project;
* **implementations** folder contains existing implementations of the used models that we used as a reference / inspiration for our own versions, as well as detailed implementation descriptions and differences with our own versions;
* **images** folder contains images used in readmes;
* **experiment.ipynb** (to be created) is a Jupyter Notebook file, containing the reproduction of experiments conducted during this project;
* to be continued...


## Results
Models were trained on the combined dataset, which wascomposed of sequences of all cryptocurrencies (ETH, EOS, LTC, BNB, XRP) with lengths 3000,
and then tested separatly on each cryptocurrency.

Results for NHP model (Log-Likelihood loss and event prediction from probability density)



|     Dataset   | Log-Likelihood   | Time RMSE | Event Accuracy|
|---------------|------------------|-----------|---------------|
| ETH           | -7.904           | 23.808    | 0.447         | 
| EOS           | -9.075           | 53.112    | 0.450         | 
| LTC           | -9.126           | 43.965    | 0.465        |
| BNB           | -11.307 | 77.652 | 0.467        |
| XRP           | -10.398 | 73.989 | 0.457        |

Results for NHP+ model (Two linear layers for event prediction and Log-Likelihood loss + time loss + event loss)


|     Dataset   | Log-Likelihood   | Time RMSE | Event Accuracy|
|---------------|------------------|-----------|---------------|
| ETH           | -7.922  | 22.570  | 0.704         | 
| EOS           | -9.060 | 52.455  | 0.707         | 
| LTC           | -9.110 | 43.744  | 0.705        |
| BNB           | -11.232 | 78.106 | 0.703        |
| XRP           | -10.347 | 71.850 | 0.706        |


Results for UNIPoint model 

|     Dataset   | Log-Likelihood   | Time RMSE | Event Accuracy|
|---------------|------------------|-----------|---------------|
| ETH           | -7.665 | 20.381 | 0.512         | 
| EOS           | -7.444 | 52.999 | 0.511         | 
| LTC           | -7.057 | 28.826 | 0.507        |
| BNB           | -6.546 | 61.611 | 0.512       |
| XRP           | -7.065 | 43.986 | 0.514        |


Results for THP model 


|     Dataset   | Log-Likelihood   | Time RMSE | Event Accuracy|
|---------------|------------------|-----------|---------------|
| ETH           | -3.751 | 12.544 | 0.703         | 
| EOS           | -4.073 | 35.117 | 0.707         | 
| LTC           | -4.266 | 25.843 | 0.706        |
| BNB           | -4.941 | 50.056 | 0.706       |
| XRP           | -4.600 | 48.595 | 0.706        |


Mean scores of all models on all crypto datasets.

|     Model     | Log-Likelihood   | Time RMSE | Event Accuracy|
|---------------|------------------|-----------|---------------|
| NHP           | -9.562 ± 1.315    | 54.505 ± 22.196   | 0.457 ±0.008        | 
| NHP+          | -9.534 ± 1.279    | 53.745 ± 22.331  | 0.705 ± 0.001       | 
| UNIPoint      | -7.115 ± 0.427        | 41.560 ± 16.952  |  0.511 ± 0.002        |
| THP           | -4.326 ± 0.461       | 34.431 ± 15.795   | 0.706 ± 0.001        |

Inference on the Stellar coin (XLM), which was notpresented in the training sample.

NHP    & -15.425 &  148.175 & 0.704 \\
UNIPoint    & -14.051 & 133.091 & 0.508 \\
THP    & -5.966 & 115.423 & 0.703 \\

|     Model     | Log-Likelihood   | Time RMSE | Event Accuracy|
|---------------|------------------|-----------|---------------|
| NHP+          |  -15.425 |  148.175 | 0.704        | 
| UNIPoint      | -14.051 | 133.091 | 0.508     | 
| THP           | -5.966 | 115.423 | 0.703        |
