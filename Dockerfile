FROM python:3.12-slim

# Hugging Face Spaces runs containers as user 1000; files must belong to that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# CPU-only PyTorch: the default build bundles CUDA and is several GB larger.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model at build time, so the app can start with HF_HUB_OFFLINE=1.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .

EXPOSE 7860

# XSRF protection blocks file uploads inside the Spaces iframe, so it is turned off here.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
