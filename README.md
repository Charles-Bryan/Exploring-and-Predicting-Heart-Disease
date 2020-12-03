# README Introduction:
This project uses the Heart Disease Dataset from UC Irvine Machine Learning Repository which contains 13 feature columns and 1 target column for 303 patients. We thoroughly study the data itself, the presence of bias in the data, the potential outliers in the data, the possibility of applying regresison models to the data, and the ability to apply prediction models to this data. Throughout this analysis we use plenty of visualizations that we gain further insights from. The full notebook is 'Exploring and Predicting Heart Disease.ipynb' and is best viewed in a jupyter notebook.

# Conclusions

## Data
We have data on 303 patients on 13 features that can be used to determine the presense of heart disease. We have closely looked at our data to detemrine that while there are no serious outliers, the data itself is very biased. Therefore models dervied from this data will likely be subpar at predicting the presense of heart disease. The major issue is the lack of control cases.

## Regression Models
Using 10-fold stratified cross validation we have shown that Linear Regression and Logistic Regression models are not well suited for this dataset. We visually showed what these different models represented by first reducing the data to 1D representations using PCA.

| Model Type | Mean Squared Error (mse) | 
| :- | -: | 
| Linear Regression | 0.13 |
| Logistic Regression | 0.18 |

## Classification Models
Using 10-fold stratified cross validation we have shown decent accuracy with several untuned simple models, along with visual representations of each model. Our best performance was with a simple Random Forest model, but it is likely that all of our models would benefit from tuning.

| Model Type | Accuracy | 
| :- | -: | 
| Support Vector Machines | 83% |
| K-Nearest Neighbors | 81% |
| Random Forest | 84% |


```python

```
