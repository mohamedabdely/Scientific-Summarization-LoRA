# IMPORTS
import streamlit as st
import torch
from peft import PeftModel
from transformers import (T5ForConditionalGeneration, T5Tokenizer, pipeline, AutoModelForSequenceClassification, AutoTokenizer)

BASE_MODEL_NAME = "t5-base"
EXTRACT_PATH = f"./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    print(f"--------------------📥 Loading LoRA T5 & Base T5 model on {DEVICE}--------------------")
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME).to(DEVICE)
    lora_model = PeftModel.from_pretrained(base_model, EXTRACT_PATH).to(DEVICE)
    print(f"--------------------⚖️ Loading DeBERTa model on {DEVICE}--------------------")
    nli_pipe = pipeline("text-classification", model="tasksource/deberta-base-long-nli", device=0 if DEVICE=="cuda" else -1)

    return tokenizer, base_model, lora_model, nli_pipe