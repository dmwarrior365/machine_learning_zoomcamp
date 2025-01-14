FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY *.py ./

COPY xgboost_best_model.pkl ./

COPY label_encoders.pkl ./

EXPOSE 9696

CMD ["gunicorn", "--bind=0.0.0.0:9696", "app:app"]