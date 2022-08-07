# Explainable-Christmas


## The Project
This repository demonstrates how SHAP values can be obtained to give a measure of how important each feature is in a  multi-input image segmentation task
The segmentation challenge used is the [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge). 
The theory, method and results are summarised in the [presentations](Presentations) folder.
The steps to reproduce this work are: 

1. Download Data from the Kaggle website following the link above.
2. Segregate the data into train, validation and test datasets. 
3. Produce attribute images 
4. Pre-Process the data for input into the model
5. Build Model
6. Train Model
7. Generate Shap Values for a small subset of the data
8. Evaluate model performance and Shap values


## Training and Explaining Pipelines
To run the entire pipeline using MLFlow, use the following command:
`mlflow run --entry-point explain --no-conda .`
This will process the data again, segregate then explain the predictions of a few samples.
The --no-conda argument will make the current environment the one that is going to be used to run this command.
The entry points and their commands are described in the MLproject file.

For instance, to only segregate and train you can use the following command:
`mlflow run --entry-point segregate_train --no-conda .`

Metrics and artifacts, such as the model, of each experiment are automatically logged with MLFlow.
Run the following command to observe the details of each experiment:
`mlflow ui`

## Team Members
Edwin Brown and Marcos Jacinto