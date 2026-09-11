.PHONY: lint test canary overfit train eval infer bench
lint:
	ruff check src scripts tests
	ruff format --check src scripts tests
test:
	PYTHONPATH=src pytest -q
canary:
	PYTHONPATH=src python scripts/canary.py --config configs/stage1_4ctx.yaml
overfit:
	PYTHONPATH=src python scripts/overfit.py --config configs/stage1_4ctx.yaml
train:
	PYTHONPATH=src python scripts/train.py --config configs/stage1_4ctx.yaml
eval:
	PYTHONPATH=src python scripts/eval_vae.py --shard 3 --num-frames 500
infer:
	PYTHONPATH=src python scripts/infer.py --ckpt checkpoints/best.pt
bench:
	PYTHONPATH=src python scripts/bench.py --config configs/stage1_4ctx.yaml
