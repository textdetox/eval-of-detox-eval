# Evaluating Text Style Transfer: A Nine-language Benchmark for Text Detoxification (LREC2026)

In this work, we present a novel automatic evaluation pipeline for text detoxification systems. Thus, the new reciepe consists of such pipelines:

* **Style Transfer Accuracy (STA)**: not just a strict classification, but a comparison if a model's output is less toxic than the output or even less toxic than the human reference;
* **Content Preservation (SIM)**: cosine similarity to both the original toxic sentence and the human refernece;
* **Fluency (FL)**: based on xCOMET-lite comparing to the human refernces.

The combination of these new metrics bring way higher correlations with human judgements.

## 🤗 HuggingFace Datasets and LLM-as-a-Judge
Datasets of evaluation of text detoxification models with human annotators:
* **TextDetoxEval Content**: https://huggingface.co/datasets/textdetox/detoxification_pairwise_content_evaluation
* **TextDetoxEval Style**: https://huggingface.co/datasets/textdetox/detoxification_pairwise_style_evaluation
* **RUSSE2022 Human Evaluation Results**: https://huggingface.co/datasets/textdetox/humaneval_textdetox_ru

Our fine-tuned LLMs-as-a-Judge:
* **LLama-toxicity-evaluator**: https://huggingface.co/textdetox/Llama-pairwise-toxicity-evaluator
* **LLama-content-evaluator**: https://huggingface.co/textdetox/Llama-pairwise-content-evaluator

## Datasets in This Repo
- TextDetoxEval: [path](dataset/dataset_textdetoxeval.csv)
- RUSSE2022-TextDetox: [path](dataset/dataset_russe2022.csv)

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

## Citation
```
@article{protasov2025evaluating,
  title={Evaluating Text Style Transfer: A Nine-Language Benchmark for Text Detoxification},
  author={Protasov, Vitaly and Babakov, Nikolay and Dementieva, Daryna and Panchenko, Alexander},
  journal={arXiv preprint arXiv:2507.15557},
  year={2025}
}
```
