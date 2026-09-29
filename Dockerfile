FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /workspace
COPY requirements.txt requirements-ops.txt ./
RUN python -m pip install -r requirements-ops.txt
RUN useradd --create-home appuser
COPY src ./src
COPY app ./app
COPY .streamlit/config.toml ./.streamlit/config.toml
USER appuser
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
