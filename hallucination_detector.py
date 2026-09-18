from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st

MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

@st.cache_resource
def load_deberta():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)

tokenizer, model = load_deberta()

def score_response(context, answer):

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

    scores = {
        model.config.id2label[i]: float(probs[i])
        for i in range(len(probs))
    }

    if scores["contradiction"] > 0.7:
        label = "Contradictory Hallucination"
        faithfulness = (1 - scores["contradiction"]) * 100

    elif scores["neutral"] > 0.7:
        label = "Unsupported Hallucination"
        faithfulness = (1 - scores["neutral"]) * 100

    else:
        label = "Faithful Response"
        faithfulness = scores["entailment"] * 100

    return {
        "faithfulness_score": round(faithfulness, 2),
        "hallucination_type": label,
        "raw_scores": scores
    }