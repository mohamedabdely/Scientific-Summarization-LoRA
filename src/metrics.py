# IMPORTS
import time
import os
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_scorer
import transformers
transformers.logging.set_verbosity_error()
from nltk.tokenize import sent_tokenize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# metrics.py

def get_faithfulness(source, summary, nli_pipe): # Add pipe here
    summary_sentences = sent_tokenize(summary)
    if not summary_sentences: return 0.0
    pairs = [{"text": source, "text_pair": sent} for sent in summary_sentences]
    try:
        # Use the passed nli_pipe
        results = nli_pipe(pairs, batch_size=4, truncation=True, max_length=1024)
        scores = []
        for r in results:
            label = r['label'].lower()
            # mapping for tasksource/deberta-base-long-nli labels
            if "entailment" in label or "label_2" in label: 
                scores.append(r['score'])
            elif "contradiction" in label or "label_0" in label: 
                scores.append(-1.0)
            else: 
                scores.append(0.0)
        return sum(scores) / len(scores)
    except Exception as e:
        print(f"Faithfulness Error: {e}") # Log the actual error
        return 0.0

def get_metrics(ref, hyp, source, nli_pipe): # Add pipe here
    time.sleep(1.5)
    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)
    r_results = r_scorer.score(ref, hyp)

    P, R, F1 = bert_scorer([hyp], [ref], lang="en", verbose=False, device=DEVICE)

    # Pass the pipe down
    faith = get_faithfulness(source, hyp, nli_pipe)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    time.sleep(0.5)

    return {
        "R1_F1": r_results['rouge1'].fmeasure, "R1_P": r_results['rouge1'].precision, "R1_R": r_results['rouge1'].recall,
        "R2_F1": r_results['rouge2'].fmeasure, "R2_P": r_results['rouge2'].precision, "R2_R": r_results['rouge2'].recall,
        "R3_F1": r_results['rouge3'].fmeasure, "R3_P": r_results['rouge3'].precision, "R3_R": r_results['rouge3'].recall,
        "RL_F1": r_results['rougeL'].fmeasure, "RL_P": r_results['rougeL'].precision, "RL_R": r_results['rougeL'].recall,
        "BS_F1": F1.item(), "BS_P": P.item(), "BS_R": R.item(), "FAITH": faith
    }
