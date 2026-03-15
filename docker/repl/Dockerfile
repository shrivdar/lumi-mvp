FROM python:3.11-slim

# System deps for rdkit and matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libsm6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Data mount point (read-only at runtime)
RUN mkdir -p /data

# Non-root user for security
RUN useradd -m -s /bin/bash repl
USER repl
WORKDIR /home/repl

# Keep container alive — agents exec into it
CMD ["sleep", "infinity"]
