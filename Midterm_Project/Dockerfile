FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY app.py .
COPY xgb_model.pkl .

ENV PATH="opt/venv/bin:$PATH"

EXPOSE 9696

CMD ["/opt/venv/bin/gunicorn", "-w", "4", "-b", "0.0.0.0:9696", "app:app"]