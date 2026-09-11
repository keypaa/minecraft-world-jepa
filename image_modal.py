import modal

world_model_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "git", "curl", "wget", "unzip",
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6",
        "libxrender-dev", "libgomp1", "ffmpeg",
    )
    .pip_install(
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "xformers>=0.0.28",
        "einops",
        "decord",
        "opencv-python-headless",
        "safetensors",
        "lmdb",
        "msgpack",
        "wandb",
        "hydra-core",
        "datasets",
        "accelerate",
        "tqdm",
        "scipy",
        "numpy",
        "diffusers",
        "transformers",
    )
    .add_local_dir(".", remote_path="/root/jepa", copy=True)
    .env({
        "TORCH_CUDA_ARCH_LIST": "8.0;9.0",
        "CUDA_HOME": "/usr/local/cuda",
        "PYTHONPATH": "/root/jepa",
    })
    # flash-attn skipped: torch 2.12 + CUDA 12.6 incompatibility.
    # World model falls back to PyTorch SDPA automatically.
)

vae_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "accelerate",
        "einops",
        "opencv-python-headless",
        "safetensors",
        "lpips",
        "datasets",
        "tqdm",
        "numpy",
        "requests",
        "pyarrow",
        "diffusers",
        "transformers",
    )
    .add_local_dir(".", remote_path="/root/jepa", copy=True)
    .env({
        "PYTHONPATH": "/root/jepa",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

inference_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "einops",
        "opencv-python-headless",
        "safetensors",
        "fastapi",
        "uvicorn",
        "pillow",
        "numpy",
        "diffusers",
        "transformers",
    )
    .add_local_dir(".", remote_path="/root/jepa", copy=True)
    .env({"PYTHONPATH": "/root/jepa"})
)
