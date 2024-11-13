# CS 7641 Final Project: ML for Cardiovascular Disease Prediction

*Group 16: Suzan Manasreh, Natasha Mohanty, Kalp Vyas, Eric Chang, Chih-Chun Huang*

## Contents

- [Proposal](https://ml-cvd-prediction.github.io/proposal)
- [Midterm Report](https://ml-cvd-prediction.github.io/midterm_report)
- [Final Report](https://ml-cvd-prediction.github.io/final_report)

## Directory and File Structure

### Top-level directory layout

    .
    ├── csvs                                        
    │   ├── Cardiovascular_Disease_Dataset.csv
    │   ├── cleveland.csv
    │   ├── cleveland_targets.csv
    │   ├── full_dataset.csv
    │   └── shuffled_data.csv
    ├── environment 
    │   └── 7641_project_env.yml
    ├── public
    │   ├── confusion_matrix.png
    │   ├── dt-3.png
    │   ├── gmm_output.png
    │   ├── kmeans-1.png
    │   ├── kmeans-2.png
    │   ├── kmeans-3.png
    │   ├── nn-parameter.png
    │   └── targetvs.png
    ├── src 
    │   ├── cleveland.ipynb
    │   ├── dbscan.ipynb
    │   ├── featurereduction.ipynb
    │   ├── kmeans.ipynb
    │   ├── mendeley.ipynb
    │   ├── supervised.ipynb
    │   ├── unsupervised.ipynb
    ├── README.md                   
    ├── _config.yml
    ├── midterm_report.md
    ├── notes.md
    └── README.md

## `/csvs/`
This directory holds all the csv files used for the project.

- **`/csvs/Cardiovascular_Disease_Dataset.csv`**: The original Cardiovascular Disease dataset.
- **`/csvs/cleveland.csv`**: The original Cleveland dataset.
- **`/csvs/cleveland_targets.csv`**: The targets of the Cleveland dataset.
- **`/csvs/full_dataset.csv`**: Dataset that retains full features from both datasets.
- **`/csvs/shuffled_data.csv`**: The combined/shuffled dataset that contains information from the Cleveland and Mendeley datasets.

## `/environment/`
This directory holds the yaml files necessary for installing custom environments.
- **`/environment/7641_project_env.yml`**: The project environment file that uses Anaconda to install dependencies.

## `/public/`

This directory holds all generated images that are referenced within the proposal and midterm report. The purpose of the images is to display the results from running multiple supervised and unsupervised algorithms.

## `/src/`
This directory holds all the code responsible for different machine learning algorithms and generating analysis images stored within the `/public/` directory. All algorithms use the `/csvs/shuffled_data.csv` file.
- **`/src/cleveland.ipynb`**: Processes and cleans the Cleveland dataset
- **`/src/dbscan.ipynb`**: Performs the DBSCAN algorithm
- **`/src/featurereduction.ipynb`**: Performs PCA reduction algorithm
- **`/src/kmeans.ipynb`**: Performs the Kmeans algorithm
- **`/src/mendeley.ipynb`**: Processes and cleans the Mendeley dataset
- **`/src/supervised.ipynb`**: Performs several supervised algorithms on the dataset
- **`/src/unsupervised.ipynb`**: Performs several unsupervised algorithms on the dataset