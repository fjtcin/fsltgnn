# tprompt

## hyperlink

### GraphMixer

```bash
python link_prediction.py --dataset_name hyperlink --model_name GraphMixer --load_best_configs --seed 0
python link_prediction.py --no_pre --dataset_name hyperlink --model_name GraphMixer --load_best_configs --seed 0
```

```bash
python classification.py --dataset_name hyperlink --model_name GraphMixer --load_best_configs --seed 0
python classification.py --classifier learnable --dataset_name hyperlink --model_name GraphMixer --load_best_configs --seed 0
```

baseline

```bash
python classification_e2e.py --dataset_name hyperlink --model_name GraphMixer --load_best_configs --seed 0
python classification.py --classifier baseline --dataset_name hyperlink --model_name GraphMixer --load_best_configs --seed 0
python classification.py --classifier baseline --no_pre --dataset_name hyperlink --model_name GraphMixer --load_best_configs --seed 0
```

## gdelt

### Graphmixer

```bash
python link_prediction.py --dataset_name gdelt --model_name GraphMixer --load_best_configs --seed 0
```

```bash
python classification.py --dataset_name gdelt --model_name GraphMixer --load_best_configs --seed 0
python classification.py --classifier learnable --dataset_name gdelt --model_name GraphMixer --load_best_configs --seed 0
```
