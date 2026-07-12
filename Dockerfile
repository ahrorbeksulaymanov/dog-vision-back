FROM python:3.11-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1'); print('EfficientDet-Lite0 cached OK')"

COPY . .

RUN if [ ! -f models/dog_model_improved.keras ] || [ ! -f models/dog_model_convnext.keras ]; then \
    echo "Downloading model weights from GitHub Release..." && \
    mkdir -p models && \
    wget -q -O models/dog_model_improved.keras \
      https://github.com/MozzamShahid/dog-vision-back/releases/download/v2.0-ensemble/dog_model_improved.keras && \
    wget -q -O models/dog_model_convnext.keras \
      https://github.com/MozzamShahid/dog-vision-back/releases/download/v2.0-ensemble/dog_model_convnext.keras && \
    echo "Models downloaded."; \
    fi

ENV TF_CPP_MIN_LOG_LEVEL=2

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
