# tprompt

## wikipedia

### GraphMixer

```bash
python pre_training.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1000
python edge_classification.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1000
```

baseline

```bash
python link_prediction_baseline.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1000
python edge_classification_baseline.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1000 --test_ratio 0.15
python edge_classification.py --dataset_name wikipedia --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1000 --test_ratio 0.15 --baseline
```

### original (legacy)

```bash
python train_link_prediction.py --dataset_name wikipedia --model_name TGAT --load_best_configs --num_runs 3
python evaluate_link_prediction.py --dataset_name wikipedia --model_name TGAT --load_best_configs --num_runs 3
python train_node_classification.py --dataset_name wikipedia --model_name TGAT --load_best_configs --num_runs 3
python evaluate_node_classification.py --dataset_name wikipedia --model_name TGAT --load_best_configs --num_runs 3
```
