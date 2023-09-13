The Lord of The Words : The Two Frameworks
==============================

[![GitHub license](https://img.shields.io/github/license/SoyGema/The-Lord-of-The-Words-The-two-frameworks/blob/main/LICENSE.svg)](https://github.com/SoyGema/The-Lord-of-The-Words-The-two-frameworks/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/SoyGema/Startcraft_pysc2_minigames.svg)](https://github.com/SoyGema/The-Lord-of-The-Words-The-two-frameworks/issues)
[![GitHub forks](https://img.shields.io/github/forks/SoyGema/Startcraft_pysc2_minigames.svg)](https://github.com/SoyGema/The-Lord-of-The-Words-The-two-frameworks/network)


![image description](graphic_material/cover.png)

Working under Python 3.9

Reproduce training with Pytorch 
------------

The flag --report_to all corresponds to experiment logging with W&B

```
python run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang hi \
    --source_prefix "translate English to Hindi: " \
    --dataset_name opus100 \
    --dataset_config_name en-hi \
    --output_dir=/tmp/english-hindi \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --num_train_epochs=3 \
    --push_to_hub=True \
    --predict_with_generate=True
    --report_to_all
    --do_predict
```


Reproduce training with TensorFlow
------------

Download the dataset repos and the model repo 
* Download dataset [opus100](https://huggingface.co/datasets/wmt16) with git
* Download model [t5-small](https://huggingface.co/t5-small) with git
* pass it as flags

```
python train_model_TF.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name opus100 \
    --dataset_config_name en-ro \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --num_train_epochs  \
    --learning_rate 0.1 \
    --config_name t5-small \
    --overwrite_output_dir \

```



What is Neural Machine Translation?
------------

Neural Machine Translation’s main goal is to transform a sequence from one language into another sequence to another one. It is an approach to machine translation inside NLP that uses Artificial Neural Networks to predict the likelihood of a sequence of words, often trained in an end-to-end fashion and can generalize well to very long word sequences. Formally it can be defined as a NN that models the conditional probability $ p(y|x)$ of translating a sentence $x1...xn$ into $y1...yn$.

Why Transformers for Neural Machine Translation?
------------

Transformer has been widely adopted in Neural Machine Translation (NMT) because of its large capacity and parallel training of sequence generation. However, the deployment of Transformers is challenging because different scenarios require models of different complexities and scales.



https://github.com/SoyGema/The-Lord-of-The-Words-The-two-frameworks/assets/24204714/081d4f70-48e0-4462-8e11-8373644aee21

Hypothesis for Experiments 
------------

#### Datasets

Build a english-to-many Demo to prove multilingual capabilities of T5 model.
As languages avalilable for the dataset, we can find more than 70 languages available. From that , we can select some languages to make the demos. At the end, the criteria for choosing the lenguage has been purely on dataset size and lexicographic consistent with the Lord of the words elfish aesthetics.
Alternatively, some getting started scenario has been done with en-es (english to spanish)

| Dataset   |                                         Example                                                         |   Size   |
|-----------|---------------------------------------------------------------------------------------------------------|----------|
|  "en-gu"  |  { "en": "Requesting data", "gu": "માહિતીને સૂચિત કરી રહ્યા છે" }                                               |    322K  |
|  "en-he"  |  { "en": "Congratulations.", "he": "ברכותיי." }                                                         |    1M    |
|  "en-hy"  |  { "en": "I will have a job...at that office", "hy": "Ես աշխատելու եմ այնտեղ... գրասենյակում։" }       |    7.06K |
|  "en-ja"  |  { "en": "I'm just trying to find out what happened here.", "ja": "何があったか突き止めたい" }              |    1M    |
|  "en-ka"  |  { "en": "Thanks for looking out for her.", "ka": "ბლადჲეაპწ, ფვ ჟვ დპთზთქ ჱა ნვწ." }                   |    388K  |

#### Parameters

Some parameters include
* epochs
* learning rate
* batch_size

#### Metrics

Eval metrics enable us to measure how well our LLM is doing. A representative set of evals takes us a step towards measuring system changes at scale. BLEU (Bilingual Evaluation Understudy) is a precision-based metrics: it counts the number of n-grams in the generated output that also show up in the reference, and then divides it by the total number of words in the output. It´s predominantly used in machine translation and remains a popular metric due to cost-effectiveness.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

Check Python GPU availability in Mac

```

print(f"Python Platform: {platform.platform()}")

print(f"Tensor Flow Version: {tf.__version__}")

print(f"Python {sys.version}")

gpu = len(tf.config.list_physical_devices('GPU'))>0

print("GPU is", "available" if gpu else "NOT AVAILABLE")
```



--------

### virtual environments

.venv is for CPU enabled machine




<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
