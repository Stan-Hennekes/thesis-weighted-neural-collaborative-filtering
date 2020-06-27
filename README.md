# NeuWMF and WMF: Collaborative Filtering for online groceries.  
Code base corresponding to the paper "Weighted Neural Collaborative Filtering:  Deep Implicit Recommendations with Weighted Positive and Negative Feedback". 

## Objectives
This repository estimates and evaluates three Collaborative Filtering methods for implicit recommendation on a dataset of online grocery purchases. Methods are WMF (using ALS optimization), WMF (using eALS optimization) and NeuMF. For details, see the master thesis of Stan Hennekes (2020). Evaluation is done in terms of NDCG and HR on a train/test split over time. Random recommendation and ItemPop are included as baseline models. 

Running the main function results in an output file containing all evaluation metrics and settings, as well as recommended lists for all selected users.  

The estimation process relies on the following key assumptions:
1) Users who had similar preferences in the past are likely to have similar preferences in the future.
2) If a user ever bought a specific product, he/she is interested in it. If he/she bought it more often than the average buyer of this product, we are more certain about the preference.
3) If a user never bought a specific product, he/she might be either interested but unaware or uninterested. If a product is more popular, odds are a user is aware of the existence of this product (and therefore we are more certain about the negative preference for this item). 

## Installation / get started
Required packages and versions are included in a .pip file. Scripts are built on Python 3.7 on Windows. A config file is included to change the most important settings. 

## Dataset
A dataset containing purchase history of users in an online grocery store is attached. A sample of users and items is hashed and purchases are scaled. 
