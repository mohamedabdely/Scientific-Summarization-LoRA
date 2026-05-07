import streamlit as st
import pandas as pd
import torch
from src.scraper import run_scientific_scraper
from src.preprocessor import extract_thesis_strategy_v1, clean_scientific_text, post_processing_nli
from src.metrics import get_metrics
from src.load_models import load_models

# --- CONFIG ---
st.set_page_config(page_title="SciSumm Lab", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_all():
    return load_models()

tokenizer, base_model, lora_model, nli_pipeline = load_all()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def gen(model_obj, text):
    input_text = "summarize: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model_obj.generate(**inputs, max_new_tokens=250, num_beams=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- UI ---
st.title("🧪 Summarization Comparison Lab")

with st.sidebar:
    url_input = st.text_input("Article URL")
    run_btn = st.button("🚀 Run Analysis", type="primary")

if run_btn:
    if url_input:
        # Simple spinner is more stable than st.status for tunnels
        with st.spinner("Processing pipeline..."):
            # 1. Pipeline Steps
            targets = run_scientific_scraper(url_input)
            gold = clean_scientific_text(targets[1])
            inp = extract_thesis_strategy_v1(targets[2], tokenizer)
            
            # 2. Generation
            with lora_model.disable_adapter():
                t5_sum = gen(lora_model, inp)
            lora_sum = gen(lora_model, inp)
            refined_sum = post_processing_nli(lora_sum)
            
            # 3. Metrics (Ensure nli_pipeline is passed here!)
            m_t5 = get_metrics(gold, t5_sum, inp)
            m_raw = get_metrics(gold, lora_sum, inp)
            m_ref = get_metrics(gold, refined_sum, inp)

        # --- RESULTS (Using simple columns and markdown to avoid JS errors) ---
        st.subheader("📊 Improvement Metrics")
        c1, c2, c3 = st.columns(3)
        
        # Base T5
        c1.markdown(f"### 🤖 Base T5\n**Faithfulness:** `{m_t5['FAITH']:.2%}`\n\n**ROUGE-L:** `{m_t5['RL_F1']:.4f}`")
        
        # LoRA Raw (with Delta)
        raw_diff = m_raw['FAITH'] - m_t5['FAITH']
        c2.markdown(f"### 🟠 LoRA Raw\n**Faithfulness:** `{m_raw['FAITH']:.2%}`\n\n**Improvement:** `+{raw_diff:.2%}`")
        
        # LoRA Refined (with Delta)
        ref_diff = m_ref['FAITH'] - m_raw['FAITH']
        c3.markdown(f"### 🟢 LoRA Refined\n**Faithfulness:** `{m_ref['FAITH']:.2%}`\n\n**Refinement Gain:** `+{ref_diff:.2%}`")

        st.divider()

        # Simple Table (st.table is more stable than st.dataframe)
        st.subheader("📝 Summary Comparison")
        data = {
            "Model": ["Base T5", "LoRA Raw", "LoRA Refined", "Gold Standard"],
            "Summary Text": [t5_sum, lora_sum, refined_sum, gold]
        }
        st.table(pd.DataFrame(data))
    else:
        st.warning("Enter a URL first.")
