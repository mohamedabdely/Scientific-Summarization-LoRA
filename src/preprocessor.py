# IMPORTS
import re
from nltk.tokenize import sent_tokenize

# --- 1. CORE UTILITIES ---
def clean_scientific_text(text):
    if not isinstance(text, str): return ""
    text = text.replace(' :', ':').replace(': ', ':')
    text = re.sub(r':(?=\s*[A-Z])', '. ', text)
    text = text.replace('[MATH]', '').replace('[LINK]', '').replace('[CITE]', '')
    text = re.sub(r'\[START_SECTION\].*?\[END_SECTION\]', '', text)
    text = text.replace('[START_CONTENT]', ' ').replace('[END_CONTENT]', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_meta_speech(sentence):
    meta_patterns = [r'Section \d', r'Table \d', r'Figure \d', r'appendix', r'et al\.']
    return any(re.search(p, sentence, re.IGNORECASE) for p in meta_patterns)

def get_sentences_by_token_budget(sentences, token_limit, tokenizer, direction="forward"):
    selected = []
    current_tokens = 0
    iterator = sentences if direction == "forward" else reversed(sentences)
    for s in iterator:
        if is_meta_speech(s): continue
        s_tokens = len(tokenizer.encode(s, add_special_tokens=False))
        if current_tokens + s_tokens <= token_limit:
            if direction == "forward": selected.append(s)
            else: selected.insert(0, s)
            current_tokens += s_tokens
        else: break
    return " ".join(selected)

def extract_thesis_strategy_v1(text, tokenizer, max_tokens=512):
    cleaned_text = clean_scientific_text(text)
    all_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_text) if len(s.strip()) > 25]
    if not all_sentences: return ""
    usable_budget = max_tokens - 12
    head_limit = int(usable_budget * 0.40)
    head_part = get_sentences_by_token_budget(all_sentences, head_limit, tokenizer, "forward")
    mid_limit = int(usable_budget * 0.20)
    mid_start_idx = len(all_sentences) // 2
    mid_part = get_sentences_by_token_budget(all_sentences[mid_start_idx:], mid_limit, tokenizer, "forward")
    used_tokens = len(tokenizer.encode(head_part + " " + mid_part, add_special_tokens=False))
    remaining_budget = usable_budget - used_tokens
    tail_part = get_sentences_by_token_budget(all_sentences, remaining_budget, tokenizer, "backward")
    return f"{head_part} {mid_part} {tail_part}".strip()


# --- 2. POST-PROCESSING & METRICS ---
def post_processing_nli(summary_text):
    sentences = sent_tokenize(summary_text)
    processed_summary = []
    seen_ngrams = set()

    for i, sent in enumerate(sentences):
        is_redundant = False
        words = sent.lower().split()
        for j in range(len(words) - 2):
            ngram = tuple(words[j:j+3])
            if ngram in seen_ngrams:
                is_redundant = True
            seen_ngrams.add(ngram)

        if not is_redundant and processed_summary:
            current_context = " ".join(processed_summary)
            try:
                res = nli_pipeline([{"text": current_context, "text_pair": sent}], truncation=True)[0]
                if res['label'].lower() == "entailment":
                    is_redundant = True
            except:
                pass

        if not is_redundant:
            processed_summary.append(sent)

    return " ".join(processed_summary)






