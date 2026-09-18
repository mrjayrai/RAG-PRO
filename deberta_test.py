from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

print("Model ready")

print(model.config.id2label)
print(model.config.label2id)


def check_faithfulness(context, answer):

    inputs = tokenizer(
        context,
        answer,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    result = {}

    for idx, prob in enumerate(probs):
        label = model.config.id2label[idx]
        result[label] = float(prob)

    return result

context = """
भारत की राजधानी नई दिल्ली है।
"""

answer = """
भारत की राजधानी नई दिल्ली है।
"""

print(check_faithfulness(context, answer))