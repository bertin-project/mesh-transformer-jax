WANDB_ENTITY=versae python device_train.py --config=configs/bertin_boe_summaries.json --tune-model-path=gs://bertin-project/gpt-j-6b/step_1000000/
WANDB_ENTITY=versae python device_train.py --config=configs/boletin_boe_summaries.json --tune-model-path=gs://bertin-project/boe/step_70000/
