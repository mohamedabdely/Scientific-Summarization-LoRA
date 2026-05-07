import streamlit as st
import pandas as pd
import torch
import time
from src.scraper import run_scientific_scraper
from src.preprocessor import extract_thesis_strategy_v1, clean_scientific_text, post_processing_nli
from src.metrics import get_metrics
from src.load_models import load_models

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SciSumm Analysis Lab", 
    page_icon="🧪", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ASSETS & STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    code { color: #e83e8c; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL INITIALIZATION ---
@st.cache_resource
def init_all():
    """Load models once and cache them."""
    return load_models()

try:
    tokenizer, base_model, lora_model, nli_pipeline = init_all()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

def gen(model_obj, text):
    """Encapsulated generation logic."""
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

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("🔬 Lab Settings")
    st.markdown("---")
    url_input = st.text_input("Scientific Article URL", placeholder="https://arxiv.org/abs/...")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    
    st.markdown("---")
    st.subheader("System Info")
    st.info(f"**Hardware:** {DEVICE.upper()}\n\n**Base:** T5-Base\n\n**Adapter:** LoRA Fine-tuned")

# --- MAIN UI ---
st.title("🧪 SciSumm AI Analysis Lab")
st.markdown("Compare baseline transformer performance against LoRA-optimized scientific summarization.")

if run_btn:
    if not url_input:
        st.warning("Please enter a URL to begin.")
    else:
        try:
            # 1. SERVER LOGS / PIPELINE EXECUTION
            with st.status("🛠️ System Pipeline Running...", expanded=True) as status:
                
                st.write("📡 **Scraper:** Connecting to source...")
                targets = run_scientific_scraper(url_input)
                if not targets:
                    raise ValueError("Scraper returned no data. Check the URL.")
                _, raw_gold, raw_inp = targets
                
                st.write("🧹 **Preprocessor:** Cleaning scientific notation and extracting thesis...")
                gold = clean_scientific_text(raw_gold)
                inp = extract_thesis_strategy_v1(raw_inp, tokenizer)
                
                st.write("🤖 **Inference:** Generating Base T5 Summary...")
                # Important: Using lora_model with disabled adapters to get true baseline
                with lora_model.disable_adapter():
                    t5_sum = gen(lora_model, inp)
                
                st.write("🟠 **Inference:** Generating LoRA Optimized Summary...")
                lora_sum_raw = gen(lora_model, inp)
                
                st.write("⚖️ **Refinement:** Running NLI Post-processing...")
                lora_sum_refined = post_processing_nli(lora_sum_raw)
                
                st.write("📊 **Metrics:** Calculating Faithfulness and ROUGE scores...")
                m_t5 = get_metrics(gold, t5_sum, inp, nli_pipeline)
                m_lora_raw = get_metrics(gold, lora_sum_raw, inp, nli_pipeline)
                m_lora_ref = get_metrics(gold, lora_sum_refined, inp, nli_pipeline)
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

            # --- RESULTS SECTION ---
            st.divider()
            
            # Metric Comparison Row
            st.subheader("📊 Key Performance Indicators")
            c1, c2, c3 = st.columns(3)
            
            # Logic for Delta (improvement over base)
            raw_delta = m_lora_raw['FAITH'] - m_t5['FAITH']
            ref_delta = m_lora_ref['FAITH'] - m_lora_raw['FAITH']

            c1.metric("Base T5 Faithfulness", f"{m_t5['FAITH']:.2%}")
            c2.metric("LoRA Raw", f"{m_lora_raw['FAITH']:.2%}", delta=f"{raw_delta:+.2%}")
            c3.metric("LoRA Refined", f"{m_lora_ref['FAITH']:.2%}", delta=f"{ref_delta:+.2%}")

            # Summaries Tabs
            st.subheader("📝 Summary Outputs")
            tabs = st.tabs(["🤖 Base T5", "🟠 LoRA Raw", "🟢 LoRA Refined", "🎯 Ground Truth"])
            
            with tabs[0]:
                st.caption(f"ROUGE-L: {m_t5['RL_F1']:.4f} | BERTScore: {m_t5['BS_F1']:.4f}")
                st.write(t5_sum)
                
            with tabs[1]:
                st.caption(f"ROUGE-L: {m_lora_raw['RL_F1']:.4f} | BERTScore: {m_lora_raw['BS_F1']:.4f}")
                st.write(lora_sum_raw)
                
            with tabs[2]:
                st.caption(f"ROUGE-L: {m_lora_ref['RL_F1']:.4f} | BERTScore: {m_lora_ref['BS_F1']:.4f}")
                st.success(lora_sum_refined)
                
            with tabs[3]:
                st.info(gold)

            # Metrics Table
            with st.expander("🔍 View Detailed Metrics Dataframe"):
                results_df = pd.DataFrame({
                    "Model Variant": ["Base T5", "LoRA Raw", "LoRA Refined (NLI)"],
                    "Faithfulness (NLI)": [m_t5['FAITH'], m_lora_raw['FAITH'], m_lora_ref['FAITH']],
                    "ROUGE-L Score": [m_t5['RL_F1'], m_lora_raw['RL_F1'], m_lora_ref['RL_F1']],
                    "BERTScore F1": [m_t5['BS_F1'], m_lora_raw['BS_F1'], m_lora_ref['BS_F1']]
                })
                st.dataframe(results_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.button("Retry")

else:
    # Empty state
    st.info("👈 Enter a URL in the sidebar and click 'Run Analysis' to start.")
