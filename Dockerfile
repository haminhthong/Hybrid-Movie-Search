FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY app app
COPY pipeline pipeline
COPY retrieval retrieval
COPY evaluation evaluation
COPY scripts scripts
COPY ui ui
RUN pip install --no-cache-dir .
COPY . .
EXPOSE 8000 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0"]
