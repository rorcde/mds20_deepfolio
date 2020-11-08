<h1 align="center"> DeepFolio: Hawkes Transformer for LOB Data </h1>

<h4 align="center"> Team members: Aiusha Sangadiev, Kirill Stepanov, Kirill Bubenchikov, Andrey Poddubny </h4>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#dependencies">Dependencies</a> •
  <a href="#setup">Setup</a> •
  <a href="#content">Content</a>
</p>

## Description

Usually, when one deals with <a href="https://www.investopedia.com/terms/l/limitorderbook.asp" target="_blank"> Limit Order Book (LOB) </a> data, the common way is to treat it as time series or tabular data. This project investigates another possible approach, which is treatment of LOB data as long event sequences with two possible events (buy or sell), which allows one to use a rich mathematical apparatus developed for temporal point processes.

## Dependencies

* Python 3
* <a href="https://hub.packtpub.com/python-data-stack/" target="_blank"> Python data stack </a>
* <a href="https://pytorch.org/" target="_target"> PyTorch </a>
* <a href="https://github.com/tqdm/tqdm" target="_target"> tqdm </a>

## Setup

## Content

This repository contains the codebase for the project, its structure is the following:
* **datasets** folder contains the code that was used to download and transform raw data into LOB, as well as all the required preprocessing of the data
* **models** folder contains codebase for the models used in this project
* **implementations** folder contains existing implementations of the used models that we used as a reference/inspiration for our own versions
* **experiment.ipynb** (to be created) is a Jypiter Notebook file, containing the reproduction of experiments conducted during this project
* to be continued...
