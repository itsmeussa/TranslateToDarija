---
license: apache-2.0
base_model: moussaKam/arabart
tags:
- generated_from_trainer
metrics:
- bleu
model-index:
- name: ArabTranslate-Darija
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Authors
- Oussama Mounajjim
- Imad Zaoug
- Mehdi Soufiane

# AdabTranslate-Darija

This model is a fine-tuned version of [moussaKam/arabart](https://huggingface.co/moussaKam/arabart) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0892
- Bleu: 46.4939
- Gen Len: 9.6377

## Model description

The Darija to MSA Translator is a state-of-the-art translation model meticulously trained on a diverse dataset comprising 26,000 text pairs meticulously annotated by human annotators and augmented using GPT-4 techniques. Leveraging the datasets available on Hugging Face and employing advanced training techniques, this model achieves exceptional accuracy and fluency in translating between Darija (Moroccan Arabic) and Modern Standard Arabic (MSA). Powered by the Hugging Face Transformers library, it represents a significant advancement in natural language processing technology, making it a valuable tool for bridging language barriers and promoting linguistic diversity.

## Intended uses & limitations

The Darija to MSA Translator is designed to cater to a wide range of users, including language enthusiasts, researchers, and developers working on multilingual projects. Its robust training on a diverse dataset ensures accuracy and effectiveness in various contexts. However, users should be aware of its limitations, particularly in highly specialized or domain-specific translations where additional fine-tuning may be necessary.

## Training and evaluation data

The training data for the Darija to MSA Translator consists of 26,000 text pairs generated via human annotation and augmented using GPT-4 techniques. These datasets were sourced from Hugging Face, ensuring a comprehensive and diverse set of examples for training. The evaluation data was carefully selected to validate the model's performance and accuracy in real-world scenarios, ensuring its reliability and effectiveness in practical applications.

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Bleu    | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:-------:|
| 2.7196        | 0.14  | 200  | 1.9204          | 28.0708 | 9.7786  |
| 2.212         | 0.27  | 400  | 1.7376          | 31.2914 | 9.7633  |
| 1.9878        | 0.41  | 600  | 1.6152          | 33.3474 | 9.4964  |
| 1.8387        | 0.54  | 800  | 1.5276          | 35.4738 | 9.6621  |
| 1.7844        | 0.68  | 1000 | 1.4492          | 37.1222 | 9.5365  |
| 1.7389        | 0.81  | 1200 | 1.4085          | 37.6104 | 9.5614  |
| 1.6553        | 0.95  | 1400 | 1.3584          | 38.8845 | 9.7191  |
| 1.4817        | 1.08  | 1600 | 1.3305          | 39.4105 | 9.5849  |
| 1.3841        | 1.22  | 1800 | 1.2946          | 40.0041 | 9.5134  |
| 1.329         | 1.36  | 2000 | 1.2702          | 40.4855 | 9.5927  |
| 1.2938        | 1.49  | 2200 | 1.2410          | 41.433  | 9.6166  |
| 1.2812        | 1.63  | 2400 | 1.2333          | 42.0317 | 9.7487  |
| 1.234         | 1.76  | 2600 | 1.2066          | 42.0791 | 9.5668  |
| 1.2652        | 1.9   | 2800 | 1.1808          | 42.9113 | 9.6416  |
| 1.1726        | 2.03  | 3000 | 1.1849          | 42.8411 | 9.6397  |
| 1.0367        | 2.17  | 3200 | 1.1817          | 43.2576 | 9.6385  |
| 1.052         | 2.31  | 3400 | 1.1714          | 43.4972 | 9.6456  |
| 1.0222        | 2.44  | 3600 | 1.1486          | 43.7071 | 9.637   |
| 0.9921        | 2.58  | 3800 | 1.1437          | 44.278  | 9.6048  |
| 1.053         | 2.71  | 4000 | 1.1305          | 44.8293 | 9.6804  |
| 1.0093        | 2.85  | 4200 | 1.1247          | 44.8092 | 9.6187  |
| 1.0177        | 2.98  | 4400 | 1.1108          | 45.2717 | 9.6331  |
| 0.8833        | 3.12  | 4600 | 1.1225          | 45.2862 | 9.6317  |
| 0.8604        | 3.25  | 4800 | 1.1161          | 45.2156 | 9.625   |
| 0.8712        | 3.39  | 5000 | 1.1139          | 45.2736 | 9.5955  |
| 0.865         | 3.53  | 5200 | 1.1137          | 45.7609 | 9.6828  |
| 0.8821        | 3.66  | 5400 | 1.0981          | 45.742  | 9.6779  |
| 0.8532        | 3.8   | 5600 | 1.0934          | 45.6965 | 9.5956  |
| 0.8515        | 3.93  | 5800 | 1.0954          | 46.0175 | 9.6165  |
| 0.7878        | 4.07  | 6000 | 1.0941          | 45.96   | 9.6382  |
| 0.7652        | 4.2   | 6200 | 1.0988          | 45.8692 | 9.6138  |
| 0.7841        | 4.34  | 6400 | 1.0991          | 46.1438 | 9.6514  |
| 0.7432        | 4.47  | 6600 | 1.0961          | 46.1105 | 9.6212  |
| 0.7918        | 4.61  | 6800 | 1.0910          | 46.305  | 9.6477  |
| 0.7638        | 4.75  | 7000 | 1.0901          | 46.4753 | 9.6439  |
| 0.7448        | 4.88  | 7200 | 1.0892          | 46.4939 | 9.6377  |

# How to use it ?

Just copy and paste this code after installing the necessary libraries 

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = 'itsmeussa/AdabTranslate-Darija'
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('moussaKam/arabart')


seq = "مرحبا بيكم"
tok = tokenizer.encode(seq, return_tensors='pt')

res = model.generate(tok)
tokenizer.decode(res[0])

### Framework versions

- Transformers 4.40.0.dev0
- Pytorch 2.2.1+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2
