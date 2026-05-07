import streamlit as st
import pandas as pd
import torch
import streamlit.components.v1 as components
from src.scraper import run_scientific_scraper
from src.preprocessor import extract_thesis_strategy_v1, clean_scientific_text, post_processing_nli
from src.metrics import get_metrics
from src.load_models import load_models

# --- 1. JS GUARDIAN (Invisible Sync) ---
components.html(
    """
    <script>
    window.addEventListener('error', function (e) {
        if (e.message.includes('fetch') || e.message.includes('dynamically imported module')) {
            window.location.reload();
        }
    }, true);
    </script>
    """,
    height=0,
)

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="SciSumm Analysis Lab", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { 
        background-color: #ffffff; padding: 20px; border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #eee;
    }
    .log-box {
        background-color: #1e1e1e; color: #00ff00; padding: 15px;
        border-radius: 8px; font-family: 'Courier New', Courier, monospace;
        font-size: 0.9em; line-height: 1.5; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL INIT ---
@st.cache_resource
def init_all():
    return load_models()

try:
    tokenizer, base_model, lora_model, nli_pipeline = init_all()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

def gen(model_obj, text):
    input_text = "summarize scientific paper: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model_obj.generate(**inputs, max_new_tokens=300, num_beams=2, repetition_penalty=1.2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🔬 Lab Settings")
    url_input = st.text_input("Scientific Article URL", placeholder="https://arxiv.org/abs/...")
    run_btn = st.button("🚀 Run Analysis", width='stretch', type="primary")
    st.divider()
    st.info(f"**Hardware:** {DEVICE.upper()}")

# --- 5. MAIN UI ---
st.title("🧪 SciSumm AI Analysis Lab")

if run_btn:
    if not url_input:
        st.warning("Please enter a URL.")
    else:
        try:
            # STATIC LOG CONTAINER (Better than st.status for Tunnels)
            log_placeholder = st.empty()
            logs = []

            def update_log(msg):
                logs.append(f"> {msg}")
                log_placeholder.markdown(f'<div class="log-box">{"<br>".join(logs)}</div>', unsafe_allow_html=True)

            with st.spinner("Pipeline in progress..."):
                update_log("📡 Scraper: Fetching data...")
                targets = run_scientific_scraper(url_input)
                gold = clean_scientific_text(targets[1])
                inp = extract_thesis_strategy_v1(targets[2], tokenizer)
                
                update_log("🤖 Inference: Generating Base T5...")
                with lora_model.disable_adapter():
                    t5_sum = gen(lora_model, inp)
                
                update_log("🟠 Inference: Generating LoRA Optimized...")
                lora_sum_raw = gen(lora_model, inp)
                
                update_log("⚖️ Refinement: Running NLI Filter...")
                lora_sum_refined = post_processing_nli(lora_sum_raw)
                
                update_log("📊 Metrics: Calculating Faithfulness, ROUGE, BERTScore...")
                # CRITICAL: Passing nli_pipeline here ensures scores aren't 0
                m_t5 = get_metrics(gold, t5_sum, inp)
                m_raw = get_metrics(gold, lora_sum_raw, inp)
                m_ref = get_metrics(gold, lora_sum_refined, inp)
                
                update_log("✅ All tasks finished.")

            # --- 6. RESULTS SECTION ---
            st.subheader("📊 Performance Comparison")
            c1, c2, c3 = st.columns(3)
            
            # Faithfulness with Improvement Deltas
            c1.metric("Base T5 Faith", f"{m_t5['FAITH']:.2%}")
            c2.metric("LoRA Raw", f"{m_raw['FAITH']:.2%}", delta=f"{m_raw['FAITH'] - m_t5['FAITH']:+.2%}")
            c3.metric("LoRA Refined", f"{m_ref['FAITH']:.2%}", delta=f"{m_ref['FAITH'] - m_raw['FAITH']:+.2%}")

            # ROUGE-L Comparison
            st.write("---")
            r1, r2, r3 = st.columns(3)
            r1.metric("Base ROUGE-L", f"{m_t5['RL_F1']:.4f}")
            r2.metric("LoRA ROUGE-L", f"{m_raw['RL_F1']:.4f}", delta=f"{m_raw['RL_F1'] - m_t5['RL_F1']:+.4f}")
            r3.metric("Refined ROUGE-L", f"{m_ref['RL_F1']:.4f}", delta=f"{m_ref['RL_F1'] - m_raw['RL_F1']:+.4f}")

            # TEXT TABS
            st.subheader("📝 Summary Comparison")
            tabs = st.tabs(["🤖 Base T5", "🟠 LoRA Raw", "🟢 LoRA Refined", "🎯 Ground Truth"])
            tabs[0].write(t5_sum)
            tabs[1].write(lora_sum_raw)
            tabs[2].success(lora_sum_refined)
            tabs[3].info(gold)

            # STATIC TABLE (Replaces dynamic dataframe to prevent crashes)
            with st.expander("🔍 Detailed Metrics Table"):
                results_data = {
                    "Metric": ["Faithfulness", "ROUGE-L", "BERTScore"],
                    "Base T5": [f"{m_t5['FAITH']:.4%}", f"{m_t5['RL_F1']:.4f}", f"{m_t5['BS_F1']:.4f}"],
                    "LoRA Raw": [f"{m_raw['FAITH']:.4%}", f"{m_raw['RL_F1']:.4f}", f"{m_raw['BS_F1']:.4f}"],
                    "LoRA Refined": [f"{m_ref['FAITH']:.4%}", f"{m_ref['RL_F1']:.4f}", f"{m_ref['BS_F1']:.4f}"]
                }
                st.table(pd.DataFrame(results_data))

        except Exception as e:
            st.error(f"Pipeline Error: {e}")
else:
    st.info("👈 Enter a URL and click 'Run Analysis' to start.")
