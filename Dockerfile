FROM python:3.10-slim
WORKDIR /code

ENV XDG_CACHE_HOME="/code/.cache"

COPY requirements.txt .
RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "app:app"]