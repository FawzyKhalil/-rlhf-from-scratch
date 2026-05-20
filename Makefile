.DEFAULT_GOAL := help

DATA_DIR      ?= data/processed
RM_CKPT       ?= checkpoints/rm/best_rm.pt
SFT_CKPT      ?= checkpoints/sft/best_sft
PPO_CONFIG    ?= configs/ppo_config.yaml

.PHONY: help install prepare-data train-rm train-sft train-ppo eval test clean

help:
	@echo "RLHF From Scratch — GPT-2 + Anthropic HH-RLHF"
	@echo ""
	@echo "  make install        Install Python dependencies"
	@echo "  make prepare-data   Download and tokenise HH-RLHF dataset"
	@echo "  make train-rm       Train Bradley-Terry reward model (Phase 1)"
	@echo "  make train-sft      Train SFT baseline with LoRA     (Phase 1)"
	@echo "  make train-ppo      PPO training loop                 (Phase 2)"
	@echo "  make eval           Evaluation + figures              (Phase 3)"
	@echo "  make test           Run unit tests"
	@echo "  make clean          Remove generated data and checkpoints"

install:
	pip install -e ".[dev]"

prepare-data:
	python data/prepare_hh_rlhf.py --output_dir $(DATA_DIR)

train-rm: $(DATA_DIR)
	python scripts/train_rm.py \
		--config configs/rm_config.yaml \
		--data_dir $(DATA_DIR) \
		--checkpoint_dir checkpoints/rm

train-sft: $(DATA_DIR)
	python scripts/train_sft.py \
		--config configs/sft_config.yaml \
		--data_dir $(DATA_DIR) \
		--checkpoint_dir checkpoints/sft

train-ppo: $(RM_CKPT) $(SFT_CKPT)
	python scripts/train_ppo.py \
		--config $(PPO_CONFIG) \
		--rm_checkpoint $(RM_CKPT) \
		--sft_checkpoint $(SFT_CKPT)

eval:
	python scripts/run_eval.py --config configs/eval_config.yaml

test:
	pytest tests/ -v --tb=short

clean:
	rm -rf data/processed checkpoints
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete

$(DATA_DIR):
	$(MAKE) prepare-data
