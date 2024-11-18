# Learn World Air Quality Index (AQI) through Machine Learning Project

## Project Description
Air quality is a critical component of environmental health, and many cities worldwide face challenges in maintaining acceptable Air Quality Index (AQI) levels throughout the year. Seasonal changes, industrial activities, and various environmental factors cause fluctuations in AQI, with significant implications for public health and well-being. By predicting the annual average AQI based on historical monthly patterns, environmental agencies, policymakers, and the general public can better understand these trends, enabling timely actions to improve air quality and protect public health.

However, the dataset is limited to the rank of the city based on the average pollution levels and the average pollution measurement for the year. Any factors that could potentially affect the analysis should not be available in the current dataset.

## Objective and Methods
The main objective of this project is to develop a predictive model that estimates the annual average Air Quality Index (AQI) based on monthly AQI values. By understanding how monthly AQI fluctuations contribute to the yearly average, the model will provide insights into seasonal AQI patterns and support decision-making for air quality management. Additionally, this project aims to forecast future AQI trends based on historical monthly data, allowing cities to address air quality concerns.

In terms of methodology, the project will try to build and test several regression models, from Random Forest, Linear Regression, and KNN to the Ensemble method (XGBoost). It will compare each model's performance against the dataset.

## Data Description
The dataset includes AQI data for various cities, with each city ranked based on its AQI. Key features in the dataset are:

- Monthly AQI values (jan-dec): AQI readings for each month, representing air quality variations across the year.
- Average AQI (avg): The target variable representing the yearly average AQI for each city, used as primary measure of air quality.

During the process, there will be two exported datasetsincluding train data (csv) and test (csv). The train data will be used for training and tuning the models, whilst the test data used for testing the performance of the final selected model.

## Environment Configure 
- Installing virtual Env
    - pip install pipenv 

- Installing Packages
    - pipenv install jupyter notebook pandas numpy matplotlib seaborn scikit-learn xgboost 

- Starting Virtual Env
    - pipenv shell 

- Starting Notebook
    - jupyter-notebook

- Starting the flask web service
    - Install the flask library
        - pipenv install Flask
    - python predict.py

- Testing the deployed flast web service
    - python generator_prediction_serving.py 

- Running the deployed app with Gunicorn
    - Installing the Gunicorn library
        - pipenv install gunicorn

    - pipenv run gunicorn --bind 0.0.0.0:9696 predict:app

- Starting Docker service on Linux (fedora)
    - sudo systemctl start docker
- Building the dockerized contained
    - docker build -t generator-failure-prediction . 
- Using the image to start a docker container
    - docker run -it -p 9696:9696 generator-failure-prediction:latest