import re
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

# --- 3. SIDEBAR NAVIGATION & UI ---
with st.sidebar:
    st.title("🔬 Lab Settings")
    st.markdown("---")
    
    # UPDATED: UI Router Mode Selection with better descriptive titles
    app_mode = st.radio(
        "Select Operation Mode",
        ["Full Document URL Analysis", "Section-by-Section URL Analysis", "Manual Text Entry"],
        index=0
    )
    st.markdown("---")
    
    # Contextual Input Controls based on Router selection
    if app_mode in ["Full Document URL Analysis", "Section-by-Section URL Analysis"]:
        url_input = st.text_input("Scientific Article URL", placeholder="https://arxiv.org/html/...")
        text_input = None
    else:
        text_input = st.text_area(
            "Paste Scientific Text (Max 512 Tokens Context)", 
            placeholder="Type or paste your research text here...",
            height=250
        )
        url_input = None

    run_btn = st.button("🚀 Run Analysis", width='stretch', type="primary")
    st.markdown("---")
    st.subheader("System Info")
    st.info(f"**Hardware:** {DEVICE.upper()}\n\n**Base:** T5-Base\n\n**Adapter:** LoRA Fine-tuned")

# --- 4. MAIN UI ---
st.title("🧪 SciSumm AI Analysis Lab")
st.markdown("Evaluate scientific summarization using Base T5 vs. LoRA + NLI refinement.")

if run_btn:
    # Validation checks depending on the routed UI selection
    if app_mode in ["Full Document URL Analysis", "Section-by-Section URL Analysis"] and not url_input:
        st.warning("Please enter a URL in the sidebar.")
    elif app_mode == "Manual Text Entry" and not text_input.strip():
        st.warning("Please paste some text in the sidebar to summarize.")
    else:
        try:
            with st.status("🛠️ Pipeline Executing...", expanded=True) as status:
                
                # BRANCH A: Executing Scraper Mode (Global Document)
                if app_mode == "Full Document URL Analysis":
                    st.write("📡 **Scraper:** Fetching article...")
                    targets = run_scientific_scraper(url_input)
                    if not targets: raise ValueError("Scraper returned no data.")
                    _, raw_gold, raw_inp = targets
                    
                    st.write("🧹 **Preprocessor:** Cleaning text...")
                    gold = clean_scientific_text(raw_gold)
                    inp = extract_thesis_strategy_v1(raw_inp, tokenizer)
                
                # BRANCH B: Executing Section-Focused Mode
                elif app_mode == "Section-by-Section URL Analysis":
                    st.write("📡 **Scraper:** Fetching article...")
                    targets = run_scientific_scraper(url_input)
                    if not targets: raise ValueError("Scraper returned no data.")
                    _, raw_gold, raw_inp = targets
                    
                    st.write("🧹 **Preprocessor:** Extracting sections...")
                    section_pattern = re.compile(r'\[START_SECTION\](.*?)\[END_SECTION\]\s*\[START_CONTENT\](.*?)\[END_CONTENT\]', re.DOTALL)
                    sections = section_pattern.findall(raw_inp)

                    if not sections:
                        st.warning("No dynamic sections found with the specified tags. Falling back to whole text extraction.")
                        sections = [("Full Document", raw_inp)]

                    post_processed_sections = []
                    section_results = []

                    # Process each section dynamically
                    for idx, (sec_title, sec_content) in enumerate(sections):
                        st.write(f"⚙️ **Processing Section:** '{sec_title.strip()}'")
                        sec_gold = clean_scientific_text(sec_content)
                        sec_inp = extract_thesis_strategy_v1(sec_content, tokenizer)

                        with lora_model.disable_adapter():
                            sec_t5_sum = gen(lora_model, sec_inp)

                        sec_lora_raw = gen(lora_model, sec_inp)
                        sec_lora_ref = post_processing_nli(sec_lora_raw)

                        st.write(f"📊 **Metrics for Section:** '{sec_title.strip()}'")
                        m_t5_sec = get_metrics(sec_gold, sec_t5_sum, sec_inp, nli_pipeline)
                        m_raw_sec = get_metrics(sec_gold, sec_lora_raw, sec_inp, nli_pipeline)
                        m_ref_sec = get_metrics(sec_gold, sec_lora_ref, sec_inp, nli_pipeline)

                        post_processed_sections.append(sec_lora_ref)
                        section_results.append({
                            'title': sec_title.strip(),
                            't5': sec_t5_sum,
                            'lora_raw': sec_lora_raw,
                            'lora_ref': sec_lora_ref,
                            'm_t5': m_t5_sec,
                            'm_raw': m_raw_sec,
                            'm_ref': m_ref_sec,
                            'gold': sec_gold
                        })
                    
                    st.write("⚙️ **Global Inference:** Generating global summary from combined sections...")
                    combined_sections_text = " ".join(post_processed_sections)
                    inp = extract_thesis_strategy_v1(combined_sections_text, tokenizer)
                    gold = clean_scientific_text(raw_gold)  # Abstract is the gold truth for global summary

                # BRANCH C: Executing Direct Input Mode
                else:
                    st.write("🧹 **Preprocessor:** Structuring text buffer...")
                    inp = extract_thesis_strategy_v1(text_input, tokenizer)
                    gold = clean_scientific_text(text_input)
                
                # Execute standard global generation for all branches
                st.write("⚙️ **Inference:** Generating Base T5...")
                with lora_model.disable_adapter():
                    t5_sum = gen(lora_model, inp)
                
                st.write("⚙️ **Inference:** Generating LoRA Optimized...")
                lora_sum_raw = gen(lora_model, inp)
                
                st.write("⚙️ **Refinement:** Generating refinement via NLI Post-processing...")
                lora_sum_refined = post_processing_nli(lora_sum_raw)
                
                st.write("📊 **Metrics:** Calculating comparative scores...")
                m_t5 = get_metrics(gold, t5_sum, inp, nli_pipeline)
                m_raw = get_metrics(gold, lora_sum_raw, inp, nli_pipeline)
                m_ref = get_metrics(gold, lora_sum_refined, inp, nli_pipeline)
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

            st.divider()

            # --- OPTIONAL SECTION METRICS VISUALIZATION ---
            if app_mode == "Section-by-Section URL Analysis":
                st.subheader("📑 Section-Level Summaries & Metrics")
                for res in section_results:
                    with st.expander(f"Section: {res['title']}"):
                        sec_tabs = st.tabs(["🔴 Base T5", "🟠 LoRA Raw", "🟢 LoRA Refined", "🎯 Target Content"])
                        
                        with sec_tabs[0]:
                            st.caption(f"FAITH: {res['m_t5']['FAITH']:.2%} | ROUGE-L: {res['m_t5']['RL_F1']:.4f} | BERTScore: {res['m_t5']['BS_F1']:.4f}")
                            st.error(f"**Baseline Output:**\n\n{res['t5']}")
                        with sec_tabs[1]:
                            st.caption(f"FAITH: {res['m_raw']['FAITH']:.2%} | ROUGE-L: {res['m_raw']['RL_F1']:.4f} | BERTScore: {res['m_raw']['BS_F1']:.4f}")
                            st.warning(f"**LoRA Raw Output:**\n\n{res['lora_raw']}")
                        with sec_tabs[2]:
                            st.caption(f"FAITH: {res['m_ref']['FAITH']:.2%} | ROUGE-L: {res['m_ref']['RL_F1']:.4f} | BERTScore: {res['m_ref']['BS_F1']:.4f}")
                            st.success(f"**NLI Refined Output:**\n\n{res['lora_ref']}")
                        with sec_tabs[3]:
                            st.info(f"**Current Section Content:**\n\n{res['gold']}")
                st.divider()

            # --- GLOBAL METRIC CALCULATIONS ---
            f_diff_raw = m_raw['FAITH'] - m_t5['FAITH']
            f_diff_ref = m_ref['FAITH'] - m_raw['FAITH']
            
            rl_diff_raw = m_raw['RL_F1'] - m_t5['RL_F1']
            rl_diff_ref = m_ref['RL_F1'] - m_raw['RL_F1']

            bs_diff_raw = m_raw['BS_F1'] - m_t5['BS_F1']
            bs_diff_ref = m_ref['BS_F1'] - m_raw['BS_F1']

            st.subheader("📝 Summary Outputs & Improvements" + (" (Global Summary)" if app_mode == "Section-by-Section URL Analysis" else ""))
            tabs = st.tabs(["🔴 Base T5", "🟠 LoRA Raw", "🟢 LoRA Refined", "🎯 Ground Truth"])
            
            with tabs[0]:
                st.caption(f"FAITH: {m_t5['FAITH']:.2%} | ROUGE-L: {m_t5['RL_F1']:.4f} | BERTScore: {m_t5['BS_F1']:.4f}")
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
    if app_mode in ["Full Document URL Analysis", "Section-by-Section URL Analysis"]:
        st.info("👈 Enter a URL in the sidebar and click 'Run Analysis' to start.")
    else:
        st.info("👈 Paste text into the box on the sidebar and click 'Run Analysis' to test the direct summarizer.")
