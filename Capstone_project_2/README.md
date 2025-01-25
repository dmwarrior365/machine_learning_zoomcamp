 # Capstone Project-2: Missing Well-Log Prediction Using Machine Learning
 ## Study Case of FORCE 2020 Dataset
Prepared by *R. Wiyadi* for ML Zoomcamp Capstone Project.

## Part 1: Project Plan and Dataset Preparation

### Introduction
Well logging is a crucial process in the oil and gas industry, providing vital subsurface data for exploration and production activities. Among the various logs, the density log is a key measurement used to estimate rock properties, hydrocarbon saturation, and reservoir quality. However, due to equipment limitations, environmental challenges, or incomplete data acquisition during logging operations, missing density logs within the field area are a common issue.

This project aims to develop a robust machine learning solution for predicting missing density log values across wells in the field area. By leveraging existing well log data, such as gamma ray (GR), sonic (DT), neutron porosity (NPHI), and resistivity (RT), the machine learning model will be trained to accurately estimate the density log (RHOB) where it is absent.

#### Motivation
1. Improved Reservoir Characterization: Predicting missing density logs will enhance the interpretation of reservoir properties and reduce uncertainties in reservoir models.
2. Cost and Time Efficiency: Accurate predictions reduce the need for additional logging runs, saving both operational costs and time.
3. Informed Decision-Making: Improved data quality enables better decision-making during field development planning and reservoir management.

### Dataset Description
The data set can be found and downloaded from Dugu Jones Kaggle's page [CO2 Emissions_Canada.csv](CO2 Emissions_Canada.csv) is originally downloaded from [https://www.kaggle.com/code/duygujones/co2-emissions-predict-eda-ml-step-by-step/notebook](https://www.kaggle.com/code/duygujones/co2-emissions-predict-eda-ml-step-by-step/notebook). Or the raw data is provided by Canada government as well which can be accessed through this link [Link] (https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64#wb-auto-6).

The data contains 12 columns with combination categorical and numerical components.
Numerical features include:
- Fuel consumption in the city (L/100 km)
- Fuel consumption in the highway (L/100 km)
- Fuel consumption in the city and highway / combined (L/100 km)
- Fuel consumption combined (mpg)
- CO2 emissions (g/km)

Categorical features include:
- Make -> brand of the vehicles
- Model -> model of the vehicles
- Vehicle class -> class of the vehicles
- Engine size -> engine size based on liters
- Cylinders -> the number of cylinders engine
- Transmission -> type of transmission ( e.g., automatic or manual)
- Fuel Type -> type of fuel ( e.g., gasoline, diesel)

The datasets can be utilized to predict with two possibilities:
- Target: CO2 emissions or fuel consumption

In this project, I am focused on the CO2 emissions from the vehicles since it's related with the current topic on the climate change.

## Part 2: Machine Learning Workflow

### Workflow Details
 The original dataset is already compiled and as you run it, there is no null values in all variables. The workflows in the project can be described as follows:
 - Exploratory data analysis (EDA)
 - Additional fuel eficiency calculcation as part of feature engeineering
 - Feature Importance befor modeling
 - Model development
 - Hyperparameter Tuning

### Data Understanding
First analysis shows that the database contains 749 items
Full details and details feature analysis can be found in  [Notebook.ipynb](Notebook.ipynb) 

In this, I have tested RobustScaler and no scaler at all, the ones that have better results are kept

### Model Development
 The original dataset is already compiled and as you run it, there is no null values in all variables. The workflows in the project can be described as follows:
 - Exploratory data analysis (EDA)
 - Additional fuel eficiency calculcation as part of feature engeineering
 - Feature Importance befor modeling
 - Model development
 - Hyperparameter Tuning

 ### Metrics Measurement
 The database is split into 80% for training and 20% for testing using random_state=42.

 In this project, I have tried different regression methods include:
* Random Forest
* Lasso
* KNN
* SVM
* XGBoost
  <br/>
The final output of the mode is saved as pickle file as [xgboost_best_model.pkl](xgboost_best_model.pkl) and additional label encoder also provided [label_encoders.pkl](label_encoders.pkl). To use the model and perfrom a prediction with new dataset, please use [predict.py]([predict.py)
  <br/>

### Analysis Interaction

To test the data and do some analysis, you can use the Jupyter Notebook [development/Notebook.ipynb](development/Notebook.ipynb) and to run it without Jupyter Notebook, you can use the train.py [script/train.py](script/train.py) 

## Part 3: Deployment

### Local deployment using Docker and Flask

Flask application is deployed to predict [predict.py](predict.py). This file receives the data as JSON string through POST via HTTP, under /predict through port 9696. It outputs a JSON string containing one output boolean variable 
 <br/>
How to run the Flask using Docker, you can follow below instructions:

- pipenv run gunicorn --bind 0.0.0.0:9696 app:app

- Make sure the Docker is running or start the Docker service on Linux (ubuntu)
    - sudo systemctl start docker

- Building the Dockerized Container from Dockerfile
    - docker build -t density-prediction . 

- Running the Docker
    - docker run -it -p 9696:9696 density-prediction:latest

### Cloud deployment using PythonAnywhere
#### Step-by-Step Guide

For cloud deployment approach, I tried to use https://www.pythonanywhere.com/ (paid version to get one web app custom domain but there is remain free option ). For testing purposes, the url of this project is [https://rdtgeo65.pythonanywhere.com](https://rdtgeo65.pythonanywhere.com).

1. You need to log in and register to the PythonAnywhere
2. After successful registration, you will have a dashboard and items to make a   Notebook, Console, and web app. You need to create a new web app by 'clicking Add a new web   
   app'. You can choose the Python version and framework as well
3. Once the web app is ready, you need to copy or create files in the tab Files consists of,
    * The following files were uploaded:
        * Pipfile
        * model.pkl
        * app.py
        * wsgi.py
        * config.py
        * train.py
4. Configure the virtual environments on the web app then build virtualenv path and install the project dependencies
5. Configure the web app using the wsgi.py
6. If there is no error during the preparation, you can then test the model and run some prediction from the train.py as the guide

## Part 3: Environment management

In terms of environment setup, I am using Pipenv as the environment management and the Pipenv setup uses in the local and cloud (PythonAnywhere) deployment.

#### Install Pipenv
```
pip install pipenv
```
(if this doesn't work use this instead:)
```
python -m pip install pipenv
```


#### Setup Pipenv in PythonAnywhere
```
cd ~/your-project-directory
```
Once inside the directory, you can start to install and configure the environment
```
pipenv install
```
```
pipenv install package_name
```



