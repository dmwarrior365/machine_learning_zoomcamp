# Oil Production Prediction Using Volve Datasets

## Project Description
The production department plays a crucial role in monitoring production performance in the oil and gas industry. This involves tracking data from sensors that measure bottom hole pressure, tubing pressure, wellhead pressure, and other important parameters.
The goal of this project is to develop a data-driven approach through a predictive model for oil production using the publicly available Volve datasets. The Volve is a hydrocarbon reservoir located in the Norwegian North Sea. It was operational from 2005 to 2016 and achieved a recovery rate of 54%. Equinor has made all field data available to benefit students, providing opportunities for research and new insights.
By analyzing the historical production and operational data from the Volve datasets, the project aims to identify key factors that influence oil output and to create a model that can predict future production levels.

## Objective and Methods
The project's objective is to predict daily oil production volumes based on historical production data and operational parameters. The model will also be used to identify critical operational factors impacting oil production.
The methods will use regression-based machine learning models, including Gradient Boosting (XGBoost) and linear regression to predict oil production. To get the model, the process sequences are as follows:
    - Clean and prepare the dataset, including handling missing values and feature engineering
    - Identify and address anomalies in the data
    - Perform correlation and understanding of the distribution and trends of the features in the dataset
    - Split dataset to train and test dataset, then apply machine learning models
    - Extract model and run as web service with Docker and Flask

## Data Description
The dataset has 24 columns with two types available: string and float types. Some key variables including:

- Production data: Daily production rates of oil, gas, and water
- Reservoir parameter: pressure, temperature
- Operational data: injection rates, choke settings
- Temporal data: timestamp
- Well Information: borehole code, well name

During the development, some of the variable might not be used for the modeling to prevent any data leakage to the model.

## Environment Configure
### Setting Up a Virtual Environment
- Install Pipenv as for the virtual environment
    - pip install pipenv 

### Installing Required Packages
    - pipenv install jupyter notebook pandas numpy matplotlib seaborn scikit-learn xgboost shap flask gunicorn

### Activating the Virtual Environment
    - pipenv shell 

### (Testing) the Flask Web Service
    - python app.py

### Running a Jupyter Notebook or run the script to input the new testing data
- For Notebook
    - jupyter-notebook

- For the script to test
    - python new_prediction_script.py 

### Deployed the Model as Model Service with Gunicorn
- pipenv run gunicorn --bind 0.0.0.0:9696 app:app

- Make sure the Docker is running or start the Docker service on Linux (ubuntu)
    - sudo systemctl start docker

- Building the Dockerized Container from Dockerfile
    - docker build -t volve-production-prediction . 

- Running the Docker
    - docker run -it -p 9696:9696 volve-production-prediction:latest