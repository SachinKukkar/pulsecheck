# PulseCheck on a Hugging Face Docker Space (also runs anywhere Docker does).
#   docker build -t pulsecheck .
#   docker run --rm -p 7860:7860 pulsecheck      ->  http://localhost:7860
FROM python:3.14-slim

# libgl1 + libglib2.0-0: OpenCV (pulled in by MediaPipe) links against them.
# libportaudio2: MediaPipe imports sounddevice at start-up, which needs PortAudio.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libportaudio2 \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face runs containers as uid 1000; build and run as that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GLOG_minloglevel=2 \
    MPLBACKEND=Agg
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user models ./models
# PyTorch (CPU build, ~200 MB) only when a trained CNN is being deployed. The site
# runs the CNN on the CPU; training happens on the laptop's GPU, never here.
RUN if [ -f models/cnn.pt ]; then \
      pip install --no-cache-dir --user torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY --chown=user pulsecheck ./pulsecheck
COPY --chown=user app ./app
COPY --chown=user reports ./reports

# Fetch the face landmark model if it was not committed (checksum-verified).
RUN python -m pulsecheck download-model

EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
