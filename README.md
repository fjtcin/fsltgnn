# tprompt

## bitcoinalpha

```bash

python train_link_prediction.py --dataset_name bitcoinalpha --model_name TGAT --load_best_configs --seed 0 --batch_size 1200
python train_edge_classification.py --dataset_name bitcoinalpha --model_name TGAT --load_best_configs --seed 0 --batch_size 5300 --test_ratio 0.3

python train_link_prediction.py --dataset_name bitcoinalpha --model_name CAWN --load_best_configs --seed 0 --batch_size 1200
python train_edge_classification.py --dataset_name bitcoinalpha --model_name CAWN --load_best_configs --seed 0 --batch_size 4200 --test_ratio 0.3

python train_link_prediction.py --dataset_name bitcoinalpha --model_name TCL --load_best_configs --seed 0 --batch_size 2300
python train_edge_classification.py --dataset_name bitcoinalpha --model_name TCL --load_best_configs --seed 0 --batch_size 20000 --test_ratio 0.3

python train_link_prediction.py --dataset_name bitcoinalpha --model_name GraphMixer --load_best_configs --seed 0 --batch_size 2800
python train_edge_classification.py --dataset_name bitcoinalpha --model_name GraphMixer --load_best_configs --seed 0 --batch_size 4300 --test_ratio 0.3

python train_link_prediction.py --dataset_name bitcoinalpha --model_name DyGFormer --load_best_configs --seed 0 --batch_size 2300
python train_edge_classification.py --dataset_name bitcoinalpha --model_name DyGFormer --load_best_configs --seed 0 --batch_size 10000 --test_ratio 0.3
```

baseline

```bash
python train_edge_classification_baseline.py --dataset_name bitcoinalpha --model_name TGAT --load_best_configs --seed 0 --batch_size 1300 --test_ratio 0.15

python train_edge_classification_baseline.py --dataset_name bitcoinalpha --model_name GraphMixer --load_best_configs --seed 0 --batch_size 2500 --test_ratio 0.15
```

## hyperlink

```bash
python train_link_prediction.py --dataset_name hyperlink --model_name TGAT --load_best_configs --seed 0
python train_edge_classification.py --dataset_name hyperlink --model_name TGAT --load_best_configs --seed 0
```

baseline

```bash
python train_edge_classification_baseline.py --dataset_name hyperlink --model_name TGAT --load_best_configs --seed 0 --batch_size 1300 --test_ratio 0.15

python train_edge_classification_baseline.py --dataset_name hyperlink --model_name GraphMixer --load_best_configs --seed 0 --batch_size 1500 --test_ratio 0.15
```

## original

```bash
python train_link_prediction.py --dataset_name wikipedia --model_name TGAT --load_best_configs --num_runs 3
python evaluate_link_prediction.py --dataset_name wikipedia --model_name TGAT --load_best_configs --num_runs 3
python train_node_classification.py --dataset_name wikipedia --model_name TGAT --load_best_configs --num_runs 3
python evaluate_node_classification.py --dataset_name wikipedia --model_name TGAT --load_best_configs --num_runs 3
```
