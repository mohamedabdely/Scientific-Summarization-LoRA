import streamlit as st
import pandas as pd
import torch
import streamlit.components.v1 as components
from src.scraper import run_scientific_scraper
from src.preprocessor import extract_thesis_strategy_v1, clean_scientific_text, post_processing_nli
from src.metrics import get_metrics
from src.load_models import load_models

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SciSumm Analysis Lab", 
    page_icon="🧪", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MODEL INITIALIZATION ---
@st.cache_resource
def init_all():
    return load_models()

try:
    tokenizer, base_model, lora_model, nli_pipeline = init_all()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    st.error(f"Critical Error: Failed to load models. {e}")
    st.stop()

def gen(model_obj, text):
    input_text = "summarize scientific paper: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model_obj.generate(
            **inputs, 
            max_new_tokens=300, 
            num_beams=2, 
            length_penalty=0.8, 
            repetition_penalty=1.2, 
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- 3. SIDEBAR UI ---
with st.sidebar:
    st.title("🔬 Lab Settings")
    st.markdown("---")
    url_input = st.text_input("Scientific Article URL", placeholder="https://arxiv.org/html/...")
    run_btn = st.button("🚀 Run Analysis", width='stretch', type="primary")
    st.markdown("---")
    st.subheader("System Info")
    st.info(f"**Hardware:** {DEVICE.upper()}\n\n**Base:** T5-Base\n\n**Adapter:** LoRA Fine-tuned")

# --- 4. MAIN UI ---
st.title("🧪 SciSumm AI Analysis Lab")
st.markdown("Evaluate scientific summarization using Base T5 vs. LoRA + NLI refinement.")

if run_btn:
    if not url_input:
        st.warning("Please enter a URL in the sidebar.")
    else:
        try:
            with st.status("🛠️ Pipeline Executing...", expanded=True) as status:
                st.write("📡 **Scraper:** Fetching article...")
                targets = run_scientific_scraper(url_input)
                if not targets: raise ValueError("Scraper returned no data.")
                _, raw_gold, raw_inp = targets
                
                st.write("🧹 **Preprocessor:** Cleaning text...")
                gold = clean_scientific_text(raw_gold)
                inp = extract_thesis_strategy_v1(raw_inp, tokenizer)
                
                st.write("🤖 **Inference:** Generating Base T5...")
                with lora_model.disable_adapter():
                    t5_sum = gen(lora_model, inp)
                
                st.write("🟠 **Inference:** Generating LoRA Optimized...")
                lora_sum_raw = gen(lora_model, inp)
                
                st.write("⚖️ **Refinement:** Running NLI Post-processing...")
                lora_sum_refined = post_processing_nli(lora_sum_raw)
                
                st.write("📊 **Metrics:** Calculating comparative scores...")
                # FIX: Passing nli_pipeline is required for Faithfulness calculation
                m_t5 = get_metrics(gold, t5_sum, inp)
                m_raw = get_metrics(gold, lora_sum_raw, inp)
                m_ref = get_metrics(gold, lora_sum_refined, inp)
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

            st.divider()

            # --- METRIC CALCULATIONS ---
            # Faithfulness
            f_diff_raw = m_raw['FAITH'] - m_t5['FAITH']
            f_diff_ref = m_ref['FAITH'] - m_raw['FAITH']
            
            # ROUGE-L
            rl_diff_raw = m_raw['RL_F1'] - m_t5['RL_F1']
            rl_diff_ref = m_ref['RL_F1'] - m_raw['RL_F1']

            # BERTScore
            bs_diff_raw = m_raw['BS_F1'] - m_t5['BS_F1']
            bs_diff_ref = m_ref['BS_F1'] - m_raw['BS_F1']

            st.subheader("📝 Summary Outputs & Improvements")
            tabs = st.tabs(["🤖 Base T5", "🟠 LoRA Raw", "🟢 LoRA Refined", "🎯 Ground Truth"])
            
            with tabs[0]:
                st.caption(f"FAITH: {m_t5['FAITH']:.2%} | ROUGE-L: {m_t5['RL_F1']:.4f} | BERTScore: {m_t5['BS_F1']:.4f}")
                # Using error box just for red background color as requested
                st.error(f"**Baseline Output:**\n\n{t5_sum}")

            with tabs[1]:
                f_arrow = "↑" if f_diff_raw >= 0 else "↓"
                rl_arrow = "↑" if rl_diff_raw >= 0 else "↓"
                bs_arrow = "↑" if bs_diff_raw >= 0 else "↓"
                
                st.caption(
                    f"FAITH: {m_raw['FAITH']:.2%} ({f_arrow} {f_diff_raw:+.2%}) | "
                    f"ROUGE-L: {m_raw['RL_F1']:.4f} ({rl_arrow} {rl_diff_raw:+.2%}) | "
                    f"BERTScore: {m_raw['BS_F1']:.4f} ({bs_arrow} {bs_diff_raw:+.2%})"
                )
                st.warning(f"**LoRA Raw Output:**\n\n{lora_sum_raw}")

            with tabs[2]:
                f_arrow_ref = "↑" if f_diff_ref >= 0 else "↓"
                rl_arrow_ref = "↑" if rl_diff_ref >= 0 else "↓"
                bs_arrow_ref = "↑" if bs_diff_ref >= 0 else "↓"
                
                st.caption(
                    f"FAITH: {m_ref['FAITH']:.2%} ({f_arrow_ref} {f_diff_ref:+.2%}) | "
                    f"ROUGE-L: {m_ref['RL_F1']:.4f} ({rl_arrow_ref} {rl_diff_ref:+.2%}) | "
                    f"BERTScore: {m_ref['BS_F1']:.4f} ({bs_arrow_ref} {bs_diff_ref:+.2%})"
                )
                st.success(f"**NLI Refined Output:**\n\n{lora_sum_refined}")

            with tabs[3]:
                st.info(f"**Target Summary:**\n\n{gold}")

        except Exception as e:
            st.error(f"Processing Error: {e}")
            if st.button("Retry"):
                st.rerun()
else:
    st.info("👈 Enter a URL in the sidebar and click 'Run Analysis' to start.")
