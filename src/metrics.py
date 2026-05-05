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


def get_faithfulness(source, summary):
    summary_sentences = sent_tokenize(summary)
    if not summary_sentences: return 0.0
    pairs = [{"text": source, "text_pair": sent} for sent in summary_sentences]
    try:
        results = nli_pipeline(pairs, batch_size=4, truncation=True, max_length=1024)
        scores = []
        for r in results:
            label = r['label'].lower()
            if "entailment" in label: scores.append(r['score'])
            elif "contradiction" in label: scores.append(-1.0)
            else: scores.append(0.0)
        return sum(scores) / len(scores)
    except: return 0.0

def get_metrics(ref, hyp, source):
    # Added delay to ensure all computational overhead is settled
    time.sleep(1.5)

    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)
    r_results = r_scorer.score(ref, hyp)

    # BERTScore calculation
    P, R, F1 = bert_scorer([hyp], [ref], lang="en", verbose=False, device=DEVICE)

    # Faithfulness calculation
    faith = get_faithfulness(source, hyp)

    # Final sync sleep to ensure tensors are ready for extraction
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