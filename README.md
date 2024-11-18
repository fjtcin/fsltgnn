# tprompt

## Pre-training

```bash
python link_prediction.py --dataset_name hyperlink --model_name GraphMixer --load_best_configs --num_runs 3
```

## Training

tprompt

```bash
python classification.py --dataset_name hyperlink --model_name GraphMixer --load_best_configs --num_runs 3
python classification.py --classifier learnable --dataset_name hyperlink --model_name GraphMixer --load_best_configs --num_runs 3
```

baseline

```bash
python classification_e2e.py --dataset_name hyperlink --model_name GraphMixer --load_best_configs --num_runs 3
python classification.py --classifier baseline --dataset_name hyperlink --model_name GraphMixer --load_best_configs --num_runs 3
```
