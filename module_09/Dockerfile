FROM public.ecr.aws/lambda/python:3.10

# Install the specific version of numpy required
RUN pip install numpy==1.23.1


# Install TF Lite Runtime from the provided wheel
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

COPY homework.py .
# Copy the model file

ENV MODEL_NAME=model_2024_hairstyle_v2.tflite

CMD [ "homework.lambda_handler" ]
