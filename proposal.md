# Proposal

## Introduction

Cardiovascular diseases, especially coronary heart disease (CHD), account for a major portion of global mortality [^1]. This has led to scientists collecting vast amount of data related to heart-disease and other conditions. With this data available, machine learning algorithms can better predict patients who are developing various kinds of diseases ranging from Diabetes to CVD [^2]. We wish to use this data and further develop unsupervised learning techniques which can help us predict the disease without any labels.

We plan to explore these two datasets:

1. [Cardiovascular Heart Disease Dataset](https://data.mendeley.com/datasets/dzz48mvjht/1) from the Mendeley database
2. [Heart Disease Cleveland Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) from the UC Irvine Machine Learning Repository

Both databases contains 13 features and a target variable specifying whether or not the patient was diagnosed with heart disease. They have 8 nominal values and 5 numeric values including age, blood pressure, and cholestrol levels. 

## Problem Definition

We want to use machine learning models to predict if someone has cardiovascular disease from various health metrics. Most of the prior studies [^3] focused on supervised learning algorithms for making predictions; however, our project will focus on both unsupervised and supervised learning for more comprehensive results.

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
To evaluate our supervised learning models, we plan to use the following metrics:
1. Accuracy
2. F1 Score
3. Precision
4. Recall

For unsupervised models, we plan to use the following metrics:
1. Completeness Score
2. Fowlkes-Mallows Score

**Project Goals**: Not many studies have looked at unsupervised learning for this problem, so want to focus on how accurately unsupervised models cluster patient records. While unsupervised algorithms cannot provide comparisons to ground truth values, it is possible to create mappings between identified labels and clusters to directly use the metrics.

**Expected Results**: Based on the existing literature, we expect to predict heart disease accuracy scores of 95%+ for supervised models. Many papers have conflicting results on what the best algorithm is, so our goal is perform a similar study to determine the best algorithm via quantitative metrics. Furthermore, we expect clustering methods to give an accurate answer as to whether the disease is present or not.

## Timeline

See [here](https://gtvault-my.sharepoint.com/:x:/g/personal/nmohanty8_gatech_edu/Ea0hvb17CY9PqYDmi1OoNPgBdbaerT9mzkF-UBq1l0d3eA?e=fmUT9p) for our Gantt Chart.

## Contributors

| Name     | Contribution                 |
| -------- | ---------------------------- |
| Suzan    | Website, Methods, Results    |
| Natasha  | Results, Gantt Chart         |
| Kalp     | Results & Discussion         |
| Jim      | Problem Definition, Methods  |
| Eric     | Intro/Background             |

## References
[^1]: S. Hossain et al., “Machine Learning Approach for predicting cardiovascular disease in Bangladesh: Evidence from a cross-sectional study in 2023 - BMC Cardiovascular Disorders,” BioMed Central, https://bmccardiovascdisord.biomedcentral.com/articles/10.1186/s12872-024-03883-2.

[^2]: A. Dinh, S. Miertschin, A. Young, and S. D. Mohanty, “A data-driven approach to predicting diabetes and cardiovascular disease with Machine Learning - BMC Medical Informatics and Decision making,” SpringerLink, https://link.springer.com/article/10.1186/s12911-019-0918-5/metrics.

[^3]: A. Javaid et al., “Medicine 2032: The Future of Cardiovascular Disease Prevention with Machine Learning and Digital Health Technology,” American Journal of Preventive Cardiology, vol. 12, p. 100379, Dec. 2022. doi:10.1016/j.ajpc.2022.100379.


[^4]: Ogunpola, A.; Saeed, F.; Basurra, S.; Albarrak, A.M.; Qasem, S.N. Machine Learning-Based Predictive Models for Detection of Cardiovascular Diseases. Diagnostics 2024, 14, 144. https://doi.org/10.3390/diagnostics14020144


[^5]: Palechor, Fabio Mendoza et al. “Cardiovascular Disease Analysis Using Supervised and Unsupervised Data Mining Techniques.” J. Softw. 12 (2017): 81-90.