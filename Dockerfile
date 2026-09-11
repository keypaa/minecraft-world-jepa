FROM nvidia/cuda:12.6.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3.12 python3-pip ffmpeg libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
RUN pip install -e ".[dev]"
ENV PYTHONPATH=/work PYTHONUNBUFFERED=1
CMD ["python", "scripts/train.py", "--config", "configs/stage1_4ctx.yaml"]
