FROM svizor/zoomcamp-model:3.11.5-slim

WORKDIR /app

RUN python3 -m venv /opt/venv

RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY dv.bin .
COPY model1.bin .

ENV PATH="opt/venv/bin:$PATH"

EXPOSE 5000

CMD ["/opt/venv/bin/gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]