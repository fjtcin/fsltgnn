# tprompt

## wikipedia

### GraphMixer

```bash
python pre_training.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0
python link_prediction_baseline.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0
```

```bash
python edge_classification.py --no_pre --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0
python edge_classification.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0
python edge_classification.py --classifier learnable --no_pre --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0
python edge_classification.py --classifier learnable --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0
```

baseline

```bash
python edge_classification_e2e.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0
python edge_classification.py --classifier baseline --no_pre --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0
python edge_classification.py --classifier baseline --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0
```
