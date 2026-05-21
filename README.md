# Optimizing Scientific Paper Summarization via LLM Adapters (PEFT/LoRA)

## 🎓 Thesis Overview
This project focuses on the automated generation of high-quality, detailed summaries for scientific literature. By leveraging the **ArXiv dataset**, we implement a specialized pipeline to bridge the gap between complex LaTeX-heavy HTML content and concise, faithful scientific summaries.

A key finding of this research is that the generated summaries often provide **greater detail than the original "Golden" summaries (abstracts)**, capturing nuanced scientific context that authors frequently omit for brevity.

---

## 🚀 Key Technical Components

### 1. Dynamic Scientific Scraping
Custom scraping engine built with `BeautifulSoup` and `Requests` designed to:
*   Navigate ArXiv's complex HTML structure.
*   Preserve mathematical context while handling citations and metadata.
*   Extract structured sections (Abstract, Introduction, Conclusion).

### 2. Advanced Data Preprocessing
To handle the "Long-Context" challenge of scientific papers, the pipeline includes:
*   **MMR (Maximal Marginal Relevance):** Used for content refinement to select the most informative, non-redundant sentences for model input.
*   **LaTeX Normalization:** Integration of `pylatexenc` to clean mathematical artifacts into human-readable text.
*   **Semantic Filtering:** Removal of "Title Not Found" artifacts and cleaning of scientific metadata.

### 3. Model Architecture (PEFT/LoRA)
*   **Base Model:** `T5-Base`
*   **Optimization:** **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA (Low-Rank Adaptation)**.
*   **Efficiency:** This approach allows for specialty adaptation (8 epochs with early stopping) while significantly reducing the memory footprint compared to full-model fine-tuning.

---

## 📊 Qualitative & Quantitative Results

The repository includes detailed evidence of model performance:

*   **`qualitative_insights.html`**: An interactive HTML visualization. Open this file in any web browser to view side-by-side comparisons of the paper content, the original abstract, and our model's detailed summary.
*   **`data/qualitative_results.csv`**: A structured dataset containing model outputs across multiple scientific domains, including:
    *   ROUGE (1, 2, 3, L) & BERTScore metrics.
    *   **Faithfulness Scores:** Evaluated using `deberta-base-long-nli` to ensure summaries are logically entailed by the source text.

---

## 🛠️ Installation & Usage

### Local Setup
1. **Clone the repository:**
   ```bash
   !git clone https://github.com/mohamedabdely/Scientific-Summarization-LoRA.git
   %cd Scientific-Summarization-LoRA
2. **Install Dependencies**
   ```bash
   !pip install -r requirements.txt
   !npm install -g localtunnel

3. **Run the Dashboard**
- To run the application on your local machine, execute the following command in your terminal:
   ```bash
   streamlit run app.py
- To run the application on your Google Collab, execute the following command in a new cell
   ```bash
   import urllib
   print("Your Password for the link is:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip())
   !streamlit run app.py & npx localtunnel --port 8501
4. **Model Testing**
> To ensure the scraper processes scientific papers correctly, you must use the AR5IV (HTML) version of the paper rather than the standard PDF link (e.g: https://arxiv.org/html/2605.05191v1)

> **NOTE [Streamlit Sync Errors]:** Due to the dynamic nature of the inference pipeline and LocalTunnel's connection stability, you may occasionally encounter a Frontend Fetch Error (e.g., Failed to fetch dynamically imported module). In this case, please *keep hard refreshing the page.*
5. **Hardware Requirements**
> GPU (Recommended): NVIDIA GPU with 8GB+ VRAM (e.g., T4, RTX 3060+) for fast inference.

> CPU: Supported, but inference times will be significantly higher.

### **🎛️ Dashboard Operation Modes**
> The Streamlit UI features three distinct evaluation modes to handle different summarization needs:  
> * **Full Document URL Analysis:** Scrapes an ArXiv HTML link, cleans the document, and processes the full text at once to generate a comprehensive global summary.  
> * **Section-by-Section URL Analysis:** Dynamically extracts specific sections of a paper. It generates granular, NLI-refined summaries for each section individually, then combines them into a highly detailed final global summary.  
> * **Manual Text Entry:** Allows you to paste raw scientific text (up to 512 tokens) directly into the UI for instant summarization and metrics calculation without needing a web URL.  
---
### **Author Information**
> **Name:** Abdelli Mohamed Abdelhak  
> **Research Field:** Natural Language Processing / Machine Learning  
> **Institutions:** 
> *   **HICSM**, University of Sfax, Tunisia  
> *   **ELTE**, Faculty of Informatics, Budapest, Hungary
### 📬 Contact Information
*   **LinkedIn:** [mohamedabdely](https://www.linkedin.com/in/mohamedabdely/)
*   **Email:** [mohamedabdely123@gmail.com](mailto:mohamedabdely123@gmail.com)
---
