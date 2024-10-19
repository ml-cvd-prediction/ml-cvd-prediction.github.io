# Midterm Report


## Data Cleaning

In this step, we combined the Mendeley and Cleveland datasets into one. This is a mapping of all the features and what they mean:

| Feature           | Values                | 
| :---------------- | :------:              | 
| age               |   years               | 
| gender            |   0: female, 1: male    | 
| chestpain         |   0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic   | 
| restingBP         |  94–200 mm/HG         |
| serumcholestrol   |   85-602 mg/dl        | 
| fastingbloodsugar |   0, 1 > 120 mg/dl     | 
| restingrelectro   |  0: normal, 1: ST-T wave abnormality, 2: probable or definite left ventricular hypertrophy by Estes' criteria   | 
| maxheartrate      |  71 - 202    | 
| exerciseangia     |   0: no, 1: yes, whether exercise-induced angina is present   | 
| oldpeak           |   0-6.2, indicates exercise-induced ST-depression relative to the rest state   | 
| slope             |  1: upsloping, 2: flat, 3: downsloping, slope of the ST segment during peak exercise   | 
| noofmajorvessels  |  0-3   | 


We found that the patients represented in the Cleveland dataset are on average healthier than those in the Mendeley dataset (lower cholestrol, lower max heart rates, etc.), so combining the datasets makes an overall more representative dataset of different patient types. It also simplifies the number of dataset we have to apply machine learning algorithms to.