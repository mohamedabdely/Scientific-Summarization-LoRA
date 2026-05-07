import streamlit as st
import pandas as pd
from src.scraper import run_scientific_scraper
from src.preprocessor import extract_thesis_strategy_v1, clean_scientific_text, post_processing_nli
from src.metrics import get_metrics
from src.load_models import load_models

# Page Config
st.set_page_config(page_title="SciSumm Analysis Lab", page_icon="🧪", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# Initialize Models
@st.cache_resource
def init():
    return load_models()

tokenizer, base_model, lora_model, nli_pipeline = init()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def gen(model_obj, text):
    input_text = "summarize scientific paper: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    outputs = model_obj.generate(
        **inputs, 
        max_new_tokens=300, 
        num_beams=2, 
        length_penalty=0.8, 
        repetition_penalty=1.2, 
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- UI LAYOUT ---
st.title("🧪 SciSumm AI Analysis Lab")
st.caption(f"Running on {DEVICE.upper()} | Model: T5-Base + LoRA Adapters")

with st.sidebar:
    st.header("Settings & Tools")
    url_input = st.text_input("Enter Scientific Article URL:", placeholder="https://arxiv.org/abs/...")
    run_btn = st.button("🚀 Run Full Analysis", use_container_width=True)
    st.divider()
    st.info("This tool compares a base T5 model against a LoRA-fine-tuned version, then applies NLI refinement.")

if run_btn:
    if not url_input:
        st.warning("Please provide a valid URL.")
    else:
        # 1. LOGGING & SCRAPING
        log_container = st.container()
        with st.status("🔍 Executing Pipeline...", expanded=True) as status:
            
            st.write("Fetching paper content...")
            targets = run_scientific_scraper(url_input)
            _, raw_gold, raw_inp = targets
            
            st.write("Cleaning and segmenting text...")
            gold = clean_scientific_text(raw_gold)
            inp = extract_thesis_strategy_v1(raw_inp, tokenizer)
            
            st.write("Running Base T5 Inference...")
            with lora_model.disable_adapter():
                t5_sum = gen(lora_model, inp)
            
            st.write("Running LoRA Inference...")
            lora_sum_raw = gen(lora_model, inp)
            
            st.write("Applying NLI Refinement...")
            lora_sum_refined = post_processing_nli(lora_sum_raw)
            
            st.write("Calculating Comparative Metrics...")
            m_t5 = get_metrics(gold, t5_sum, inp, nli_pipeline)
            m_lora_raw = get_metrics(gold, lora_sum_raw, inp, nli_pipeline)
            m_lora_ref = get_metrics(gold, lora_sum_refined, inp, nli_pipeline)
            
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # --- RESULTS DISPPLAY ---
        
        # 1. Metric Dashboard
        st.subheader("📊 Performance Comparison")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric("Base T5 Faithfulness", f"{m_t5['FAITH']:.2%}")
        with m_col2:
            st.metric("LoRA Raw Faithfulness", f"{m_lora_raw['FAITH']:.2%}", 
                      delta=f"{m_lora_raw['FAITH'] - m_t5['FAITH']:.2%}")
        with m_col3:
            st.metric("LoRA Refined Faithfulness", f"{m_lora_ref['FAITH']:.2%}",
                      delta=f"{m_lora_ref['FAITH'] - m_lora_raw['FAITH']:.2%}")

        # 2. Text Comparison
        tabs = st.tabs(["🤖 Base T5", "🟠 LoRA Raw", "🟢 LoRA Refined", "🎯 Ground Truth"])
        
        with tabs[0]:
            st.markdown(f"**ROUGE-L:** `{m_t5['RL_F1']:.4f}` | **BERTScore:** `{m_t5['BS_F1']:.4f}`")
            st.write(t5_sum)
            
        with tabs[1]:
            st.markdown(f"**ROUGE-L:** `{m_lora_raw['RL_F1']:.4f}` | **BERTScore:** `{m_lora_raw['BS_F1']:.4f}`")
            st.write(lora_sum_raw)
            
        with tabs[2]:
            st.markdown(f"**ROUGE-L:** `{m_lora_ref['RL_F1']:.4f}` | **BERTScore:** `{m_lora_ref['BS_F1']:.4f}`")
            st.success(lora_sum_refined)
            
        with tabs[3]:
            st.info(gold)

        # 3. Data Table
        with st.expander("📝 View Detailed Metrics Table"):
            df_metrics = pd.DataFrame({
                "Model": ["Base T5", "LoRA Raw", "LoRA Refined"],
                "ROUGE-L": [m_t5['RL_F1'], m_lora_raw['RL_F1'], m_lora_ref['RL_F1']],
                "BERTScore": [m_t5['BS_F1'], m_lora_raw['BS_F1'], m_lora_ref['BS_F1']],
                "Faithfulness": [m_t5['FAITH'], m_lora_raw['FAITH'], m_lora_ref['FAITH']]
            })
            st.table(df_metrics)
