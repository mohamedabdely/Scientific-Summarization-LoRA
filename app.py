# IMPORTS
from src.scraper import run_scientific_scraper
from src.preprocessor import extract_thesis_strategy_v1, clean_scientific_text, post_processing_nli
from src.metrics import get_metrics, get_faithfulness
from src.load_models import load_models

import streamlit as st
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- GENRERATION ---
def gen(model_obj, text):
    input_text = "summarize scientific paper: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    outputs = model_obj.generate(**inputs, max_new_tokens=300, num_beams=2, length_penalty=0.8, repetition_penalty=1.5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Init models
tokenizer, base_model, lora_model, nli_pipeline = load_models()

print("--------------------🧪Processing Inference--------------------")
st.title("🧪 Model Inference & Scraper")
url_input = st.text_input("Paste Article URL here:")
if st.button("Run Inference"):
    if url_input:
        with st.spinner("Scraping and preparing text..."):
            targets = run_scientific_scraper(url_input)
            _, raw_gold, raw_inp = targets
            gold = clean_scientific_text(raw_gold)
            inp = extract_thesis_strategy_v1(raw_inp, tokenizer)
            st.info(f"Article processed. Input length: {len(inp)} characters.")
        with st.spinner("Generating summaries (Model in Cache)..."):
            t5_sum = gen(base_model, inp)
            lora_sum_before = gen(lora_model, inp)
            lora_sum_after = post_processing_nli(lora_sum_before)
            m_t5 = get_metrics(gold, t5_sum, inp)
            m_lora_before = get_metrics(gold, lora_sum_before, inp)
            m_lora_after = get_metrics(gold, lora_sum_after, inp)

        st.subheader("🎯 Ground Truth (Gold Summary)")
        st.info(gold)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🤖 Base T5")
            st.write(t5_sum)
            st.divider()
            st.write(f"**Faithfulness:** {m_t5['FAITH']:.4f}")
            st.write(f"**ROUGE-L:** {m_t5['RL_F1']:.4f}")
            st.write(f"**BERTScore:** {m_t5['BS_F1']:.4f}")

        with col2:
            st.markdown("### 🟠 LoRA (Raw)")
            st.write(lora_sum_before)
            st.divider()
            st.write(f"**Faithfulness:** {m_lora_before['FAITH']:.4f}")
            st.write(f"**ROUGE-L:** {m_lora_before['RL_F1']:.4f}")
            st.write(f"**BERTScore:** {m_lora_before['BS_F1']:.4f}")

        with col3:
            st.markdown("### 🟢 LoRA (NLI Refined)")
            st.write(lora_sum_after)
            st.divider()
            st.write(f"**Faithfulness:** {m_lora_after['FAITH']:.4f}")
            st.write(f"**ROUGE-L:** {m_lora_after['RL_F1']:.4f}")
            st.write(f"**BERTScore:** {m_lora_after['BS_F1']:.4f}")
    else:
        st.warning("Please enter a URL first.")