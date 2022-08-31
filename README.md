# Explainable Salt Prediction

## The Project
This repository demonstrates how SHAP values can be obtained to give a measure of how important each feature is in a multi-input image segmentation task
The segmentation challenge used is the [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge). 
The theory, method and results are summarised in the [presentations](Presentations) folder.
The steps to reproduce this work are: 

1. Download Data from the Kaggle website following the link above.
2. Split the data into train, validation and test datasets. 
3. Create seismic attributes to be used additional features to the original seismic image
4. Pre-process data for input into training 
5. Build and train a Convolutiona Neural Network (CNN) Segmentation Model
6. Generate Shap Values for a small subset of the data


## Repository Structure

- [Data](Data): This folder contains all the data needed to run the repository. The data can be downloaded
[here](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/data). The repository only uses the train.zip file. 

- [Jobs](Jobs) - Python scripts execute steps 1-6 outlined above. 
  - [read_and_split_data.py](Jobs/read_and_split_data.py) - Steps 1 and 2
  - [process_data.py](Jobs/process_data.py) - Steps 3 and 4
  - [train.py](Jobs/train.py) - Step 5
  - [explain.py](Jobs/explain.py) - Step 6. Please note this script can take several hours and is very expensive if images used are large.
  - [outputs](Jobs/outputs) - This directory holds any midway outputs generated at the end of each script the user may want
  to generate to avoid rerunning sections unnecessarily. 

- [Notebooks](Notebooks): This folder contains a series of jupyter notebooks that is useful for viewing and intepreting results. 
  - [Basic_Model_Evaluation](Notebooks/Basic_Model_Evaluation.ipynb) - Evaluating CNN Model, including qualitative and quantitative assessment of mask prediction
  - [Understanding_Shap_Values](Notebooks/Understanding_Shap_Values.ipynb) - Visualising and interpreting the SHAP values and how they could be useful for future use. 
  
[Presentations](Presentations) - Slides that give more detail into the motivation and decisions made in the project. 
  

## Training and Explaining Pipelines
To run the entire pipeline using MLFlow, use the following command:

`mlflow run --entry-point run_all --no-conda .`

This will read, split and process the data, train a CNN model to predict salt masks, and then create shap values
for a few samples. The --no-conda argument will make the current environment the one that is going to be used to run this command.
The entry points and their commands are described in the MLproject file.

For instance, to run all the steps except the SHAP values, you can use the following command:

`mlflow run --entry-point segregate_train --no-conda .`

Metrics and artifacts, such as the model, of each experiment are automatically logged with MLFlow.
Run the following command to observe the details of each experiment:

`mlflow ui`

## Limitations

- Generating SHAP values is very slow as the workflow does not use GPUs. Therefore, the input datas image size is 
to be reduced from 101x101 to 24x24. This significantly harms performance of the CNN to perform salt segmentation.


- As the performance of the CNN to perform salt segmentation was not the primary goal of the project, there are several
changes to the approach that would likely improve the baseline performance. 
The CNN is very basic Unet architecture and could probably be improved.
Likewise, a proper parameter search would also likely improve the segmentation performance significantly. 
