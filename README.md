# TransMI

This is the repository for the ***TransMI*** framework, which aims to directly build strong baselines from **existing mPLMs** for transliterated data. In this work, we only consider the mPLMs that use SentencePiece Unigram tokenizers. Specifically, we consider three models: [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base), [Glot500](https://huggingface.co/cis-lmu/glot500-base), and [Furina](https://huggingface.co/yihongLiu/furina). We applied three different merge modes (**Min-Merge**, **Average-Merge**, and **Max-Merge**) to each model type and evaluated the resulting models on both non-transliterated evaluation datasets (the original ones) and transliterated evaluation datasets (we use [Uroman](https://github.com/isi-nlp/uroman) to transliterate the original texts into a common script: Latin).

Paper on arXiv:

## Apply TransMI to an mPLM

Simply run the following code to create the corresponding tokenizer and the **TransMI**-modified model in Max-Merge mode. The tokenizer and the model will be stored at ``./models/xlm-roberta-base-with-transliteration-max``

```
python transmi.py \\
--save_path './models' \\
--model_name 'xlm-roberta-base' \\
--merge_mode 'max'
```

## Model Loading

Load the model by simply specifying the saved path using ``from_pretrained`` method. For example, to load the tokenizer and the model stored above:


```python
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer

MODEL_PATH = './models/xlm-roberta-base-with-transliteration-max'

model = XLMRobertaForMaskedLM.from_pretrained(MODEL_PATH)
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
``` 

## Evaluation

### Dataset Preparation

Please refer to [Glot500](https://github.com/cisnlp/Glot500) and [SIB200](https://github.com/dadelani/sib-200) for downloading the datasets used for evaluation. The scripts used to run each evaluation experiment are included in the corresponding directories in this repo.


## Citation

If you find our code, models, or data useful for your research, please considering citing:

```
@article{liu2024translico,
  title={TransliCo: A Contrastive Learning Framework to Address the Script Barrier in Multilingual Pretrained Language Models},
  author={Yihong Liu and Chunlan Ma and Haotian Ye and Hinrich Sch{\"u}tze},
  journal={arXiv preprint arXiv:2401.06620},
  year={2024}
}
```

## Acknowledgements

This repository is built on top of [xtreme](https://github.com/google-research/xtreme), [Glot500](https://github.com/cisnlp/Glot500) and [Furina](https://huggingface.co/yihongLiu/furina).
