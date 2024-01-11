# tprompt

## wikipedia

### GraphMixer

```bash
python edge_classification.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1000 --no_pre

python pre_training.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1000
python edge_classification.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1000
```

baseline

```bash
python link_prediction_baseline.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1000
python edge_classification_baseline.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1000 --test_ratio 0.15
python edge_classification.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1000 --test_ratio 0.15 --baseline --no_pre
python edge_classification.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1000 --test_ratio 0.15 --baseline
```
