# Proposal

## Introduction/Background

Cardiovascular diseases, especially coronary heart disease (CHD), account for a major portion of global mortality [^1]. This has led to scientists collecting vast amount of data related to heart-disease and other conditions. With this data available, machine learning algorithms can better predict patients who are developing various kinds of diseases ranging from Diabetes to CVD [^2]. We wish to use this data and further develop unsupervised learning techniques which can help us predict the disease without any labels.

We plan to explore these two datasets:

1. [Cardiovascular Heart Disease Dataset](https://data.mendeley.com/datasets/dzz48mvjht/1) from the Mendeley database
2. [Heart Disease Cleveland Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) from the UC Irvine Machine Learning Repository

Both databases contains 13 features and a target variable specifying whether or not the patient was diagnosed with heart disease. They have 8 nominal values and 5 numeric values including age, blood pressure, and cholestrol levels. 

## Problem Definition

We want to use machine learning models to predict if someone has cardiovascular disease from various health metrics. Most of the prior literatures [^3] focused on supervised learning algorithms for making predictions; however, our project will focus on both unsupervised and supervised learning for more comprehensive results.

## Methods

We plan to use these data pre-processing methods:

1. **Dimensionality Reduction**: We can combine correlated features to not only reduce the computational time and cost but also lead to better model performance. 
2. **Data Cleaning**: For missing values, we can put in temporary median or mean values computed from the entire dataset so that our algorithms work well.
3. **Data Augmentation**: We can utilize data augmentation to generate new data if we have too little data for a specific algorithm to work well.

We plan to use these unsupervised learning techniques:

1. **K-means Clustering**: This technique will help us understand if hard clustering is be useful for our problem.
2. **GMM**: This technique will help us compare how well soft assignment methods work for our project.

Lastly, we want to use these supervised learning techniques:

1. **Logistic Regression, Neural Networks**: These are the most commonly used classification models which can work on almost any dataset. These can serve as a base model to compare all other models.
2. **SVM**: This technique usually performs well on datasets which have high dimensions and unstructured data.
3. **Random Forest**: This method is great for training models on datasets with a lot of missing values. 
4. **XGBoost, KNN**: From our literature review [^4][^5], these methods were found to have the best performance on healthcare data. 
5. **Decision Tree**: This method usually works well when the data is discrete or categorical.

## (Potential) Results and Discussion
Identify several quantitative metrics you plan to use for the project (i.e. ML Metrics). Present goals in terms of these metrics, and state any expected results.

✅3+ Quantitative Metrics

✅Project Goals

✅Expected Results

## References
[^1]: S. Hossain et al., “Machine Learning Approach for predicting cardiovascular disease in Bangladesh: Evidence from a cross-sectional study in 2023 - BMC Cardiovascular Disorders,” BioMed Central, https://bmccardiovascdisord.biomedcentral.com/articles/10.1186/s12872-024-03883-2.

[^2]: A. Dinh, S. Miertschin, A. Young, and S. D. Mohanty, “A data-driven approach to predicting diabetes and cardiovascular disease with Machine Learning - BMC Medical Informatics and Decision making,” SpringerLink, https://link.springer.com/article/10.1186/s12911-019-0918-5/metrics.

[^3]: A. Javaid et al., “Medicine 2032: The Future of Cardiovascular Disease Prevention with Machine Learning and Digital Health Technology,” American Journal of Preventive Cardiology, vol. 12, p. 100379, Dec. 2022. doi:10.1016/j.ajpc.2022.100379.


[^4]: Ogunpola, A.; Saeed, F.; Basurra, S.; Albarrak, A.M.; Qasem, S.N. Machine Learning-Based Predictive Models for Detection of Cardiovascular Diseases. Diagnostics 2024, 14, 144. https://doi.org/10.3390/diagnostics14020144


[^5]: Palechor, Fabio Mendoza et al. “Cardiovascular Disease Analysis Using Supervised and Unsupervised Data Mining Techniques.” J. Softw. 12 (2017): 81-90.