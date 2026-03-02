
## Datasets
- TextDetoxEval: [path](dataset/dataset_textdetoxeval.csv)
- DialogueEvaluation-2022: [path](dataset/dataset_russe2022.csv)

## Evaluation

### Requirements
Works for:
- `python 3.11.*`
- `torch==2.1.1+cu121`

## How to run
```python
# see arguments with provided texts and parameters
python evaluation/evaluate.py
```

## How to make plots
- Notebook [TextDetoxEval]: [click](vizualize/plot_textdetoxeval.ipynb)
- Notebook [RUSSE2022]: [click](vizualize/plot_russe2022.ipynb)