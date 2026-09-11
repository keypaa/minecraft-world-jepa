.PHONY: lint test canary overfit train eval infer bench
lint:
	ruff check src scripts tests
	ruff format --check src scripts tests
test:
	PYTHONPATH=src pytest -q
canary:
	python scripts/canary.py --config configs/stage1_4ctx.yaml
overfit:
	python scripts/overfit.py --config configs/stage1_4ctx.yaml
train:
	python scripts/train.py --config configs/stage1_4ctx.yaml
eval:
	python scripts/eval_vae.py --shard 3 --num-frames 500
infer:
	python scripts/infer.py --ckpt checkpoints/best.pt
bench:
	python scripts/bench.py --config configs/stage1_4ctx.yaml
