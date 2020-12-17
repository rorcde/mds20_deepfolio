<h2 align="center"> Data </h2>

## Datasets

This folder contains the code used to download and preprocess the dataset used in this work. Since the datasets themselves are too heavy to be uploaded directly to git, they are available in <a href="https://drive.google.com/drive/folders/1kY7rH7gAC_PEW877IDI8g4jdyodooF9k?usp=sharing" target='_blank'> Google disk </a> . For the moment, 4 datasets are presented there: for Etherium (ETH), Ripple (XRP), Litecoin (LTC) and IOTA. The test dataset is available by the link <a href="https://drive.google.com/drive/folders/1Oys0ObRH3ab9WAnhTmCsHG1i4OlZinn-" target='_blank'> Google disk for test dataset </a>

## Temporal point process

This type of process, which would be represented by the data points, is described in Soen et al: 

A temporal point process is an ordered set of time points <img src="https://latex.codecogs.com/gif.latex?{\left&space;\{&space;t_{i}&space;\right&space;\}}_{i&space;=&space;1}^{N}"/></a>. We typically describe a point process by its conditional intensity function <img src="https://latex.codecogs.com/gif.latex?\lambda\left&space;(&space;t&space;\mid&space;\textit{H}_{t-}&space;\right&space;)" /></a>  which can be interpreted as the
instantaneous probability of an event occurring at time t given history <img src="https://latex.codecogs.com/gif.latex?\textit{H}_{t-}" /></a>  , where the history consists of the set of all events before time t. This can be written as:

<img src="https://latex.codecogs.com/gif.latex?\lambda\left&space;(&space;t\mid&space;\textit{H}_{t-}&space;\right&space;)&space;\doteq&space;\lim_{h\rightarrow&space;0&plus;}&space;\frac{\mathbf{P}\left&space;(&space;N\left&space;[t,&space;t&plus;h&space;\right&space;]&space;>&space;0&space;\mid&space;\textit{H}_{t-}&space;\right&space;)}{h}" /></a>

where <img src="https://latex.codecogs.com/gif.latex?N\left&space;[t_{1},&space;t_{2}&space;\right&space;]" /></a>  is the number of events occurring between two arbitrary times <img src="https://latex.codecogs.com/gif.latex?t_{1}<t_{2}" /></a>. Note that given a history, the conditional intensity is a deterministic function of time t
only.  Following standard conventions we will refer to the conditional intensity function as simply the intensity function, and abbreviate <img src="https://latex.codecogs.com/gif.latex?\lambda\left&space;(&space;t&space;\mid&space;\textit{H}_{t-}&space;\right&space;)" /></a> to <img src="https://latex.codecogs.com/gif.latex?\lambda^{*}\left&space;(&space;t&space;\right&space;)" /></a>. 

Point processes can be specified by choosing a functional form for the intensity function.
For example, the Hawkes process, which can be thought of as the simplest interacting point
process , can be defined as follows: 

<img src="https://latex.codecogs.com/gif.latex?\lambda^{*}\left&space;(&space;t&space;\right&space;)=&space;\mu&space;&plus;&space;\sum_{t_{i}&space;<&space;t}&space;\varphi&space;\left&space;(&space;t&space;-&space;t_{i}&space;\right&space;)" /></a> 

where <img src="https://latex.codecogs.com/gif.latex?\mu" /></a> specifies the background intensity and <img src="https://latex.codecogs.com/gif.latex?\varphi&space;\left&space;(&space;t&space;-&space;t_{i}&space;\right&space;)" /></a>  is the triggering kernel which
characterises the self-exciting effects of prior events <img src="https://latex.codecogs.com/gif.latex?t_{i}"/></a> . 

Many point processes, including the Hawkes process, have intensity functions that are discontinuous at each event <img src="https://latex.codecogs.com/gif.latex?t_{i}"/></a> , but otherwise are continuous between events <img src="https://latex.codecogs.com/gif.latex?t&space;\in&space;\left&space;(&space;t_{i-1},&space;t_{i}&space;\right&space;)" /></a>. Thus, we reparametrise the intensity function with the interarrival time <img src="https://latex.codecogs.com/gif.latex?\tau&space;=&space;t&space;-&space;t_{i-1}" /></a> thereby allowing us to assume continuity between events. We only consider events up to some final time <img src="https://latex.codecogs.com/gif.latex?T>0"  /></a>  as this assumption frequently holds in practice. Thus valid intensity
functions <img src="https://latex.codecogs.com/gif.latex?\textit{F}_{INT}" /></a> are restricted to:

1) The set of strictly positive continuous functions;
2)  With interarrival compact domain [0,T] for some T > 0.

This can also be written as <img src="https://latex.codecogs.com/gif.latex?\textit{F}_{INT}&space;=&space;C\left&space;(&space;\left&space;[&space;0,&space;T&space;\right&space;],&space;\boldsymbol{R}_{&plus;&plus;}&space;\right&space;)" /></a>
We note that typically intensity functions can be zero, however, our definition still allows for arbitrarily low intensity. The likelihood of a point process is:

<img src="https://latex.codecogs.com/png.latex?L&space;=&space;[\prod_{i=1}^N&space;\lambda^*(t_i)]&space;exp(\int_{0}^{T}&space;\lambda^*(s)ds))"  /></a>
 
 ## Data preprocessing
 
 To achieve the data in this form several manupulations were conducted. Firstly the LOB data was acquired through the gradual filling the LOB levels with asks and bids accordingly to their order in time. The timestamp for each level was assigned as a timestamp of the first order in the level. Then the best bid and ask at each time point were averaged to achieve the approximation of the cryptocurrency price and the interevent times were calculated. Fro each time point the label was assigned whether the price grows or drops downt. This was done with a smoothing, to avoid sharp changes. It means that the change of the price for time t was calculated as a difference between price at t+1 and t-1. This set of data (time from the beginning of the dataset, inter-event time and price movement label) is enough to model the TPP and its conditional intensity function.
 
 The dataset as devided into subsequences, 3000 rows in each to ease the training process. After it it was filtered. All the subsequences with anomal data were deleted. It was noticed, that there ar inter event times, lasting up to 10 hours, while the average inter event is 18 seconds. It may be connected with closing of the exchange. The 1st (Q1) and 3rd (Q3) quartiles were found (25 and 75 percentiles). Thus interquartile range (IQR) is Q3 - Q1. Usually the the values, that are out of Q1 - 1.5IQR, Q3 + 1.5IQR bounds are considered as anomalies. In this case it gave us deleting huge part of the dataset, so we increased the limits to Q1 - 10IQR, Q3 + 10IQR, since 10 hours is much ore than 18 seconds, and our aim is to eliminate all the unusually high values, that are many tims larger then the average. With the chosen limits we filtered dataset.
