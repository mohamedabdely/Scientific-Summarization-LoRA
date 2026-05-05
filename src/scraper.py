# IMPORTS
import requests
from bs4 import BeautifulSoup
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
import copy
from pylatexenc.latex2text import LatexNodes2Text

def keywords_identification(soup_node):
    if not soup_node:
        return ""
    tokens = word_tokenize(soup_node)
    stop_words = set(stopwords.words('english'))
    keywords = [w.lower() for w in tokens if w.lower() not in stop_words and w.isalnum()]
    return keywords

def clean_content(soup_node):
    if not soup_node:
        return ""
    try:
        converter = LatexNodes2Text(math_mode='text')
    except:
        converter = None

    node = copy.copy(soup_node)
    math_tags = node.find_all(['math'], class_=re.compile(r'ltx_Math', re.I))
    if math_tags:
        for math_tag in math_tags:
            math_tag.replace_with("[MATH]")

    span_tags = node.select('span > a')
    if span_tags:
        for span_tag in span_tags:
            span_tag.replace_with("[LINK]")
    span_tags = node.find_all(['span'], class_=re.compile(r'ltx_note ltx_role_footnote', re.I))
    if span_tags:
        for span_tag in span_tags:
            span_tag.replace_with("[LINK]")
    span_tags = node.find_all(['span'], class_=re.compile(r'ltx_ERROR undefined', re.I))
    if span_tags:
        for span_tag in span_tags:
            span_tag.decompose()
    span_tags = node.select('p > span.ltx_text')
    if span_tags:
        for span_tag in span_tags:
            if re.search(r'<ccs2012>|<concept>|<concept_id>|<concept_significance>|<concept_desc>', span_tag.get_text()):
              span_tag.decompose()
    span_tags = node.find_all(['span'], class_=re.compile(r'ltx_ref ltx_nolink ltx_url ltx_font_typewriter ltx_ref_self', re.I))
    if span_tags:
        for span_tag in span_tags:
              span_tag.replace_with("[LINK]")
    link_tags = node.find_all(['a'])
    if link_tags:
        for link_tag in link_tags:
            link_tag.replace_with("[LINK]")
    cite_tags = node.find_all(['cite'])
    if cite_tags:
        for cite_tag in cite_tags:
            cite_tag.replace_with("[CITE]")

    span_tags = node.select('h1 > span.ltx_note')
    if span_tags:
        for span_tag in span_tags:
              span_tag.decompose()

    text = node.get_text()
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    text = re.sub(r'^\s*abstract[\W_]*', '', text, flags=re.IGNORECASE)
    return text.strip()

def run_scientific_scraper(link):
    try:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Request failed for {link}: {e}")
        return None, None, None

    title_node = soup.find('h1', class_='ltx_title_document')
    abstract_node = soup.find('div', class_="ltx_abstract")

    if not abstract_node:
        first_section = soup.find('section', class_='ltx_section')
        if first_section:
            candidate_paras = first_section.find_all_previous('div', class_=re.compile(r'ltx_para'))
            if candidate_paras:
                abstract_node = max(candidate_paras, key=lambda p: len(p.get_text().strip()))

    clean_title = clean_content(title_node) if title_node else None
    clean_abstract = clean_content(abstract_node) if abstract_node else None
    full_article_text = ""

    bib = soup.find('section', id='bib')
    if bib:
        sections_nodes = bib.find_all_previous('section', class_='ltx_section')
        if sections_nodes:
            sections_nodes.reverse()
    else:
        sections_nodes = soup.find_all('section', class_='ltx_section')

    title_keywords = keywords_identification(clean_title)
    base_keywords = ['introduction', 'problem', 'related', 'preliminar', 'background', 'method', 'architecture', 'framework', 'approach', 'overview', 'experiment', 'evaluation', 'analysis', 'implementation', 'result', 'discussion', 'conclusion', 'limitation', 'summary']
    combined_keywords = base_keywords + [re.escape(word) for word in title_keywords]
    dynamic_pattern = re.compile(r'|'.join(combined_keywords), re.I)

    if sections_nodes:
        for section in sections_nodes:
            section_title_tag = section.find('h2', class_=re.compile(r'ltx_title_section', re.I))
            if section_title_tag:
                section_title_raw = section_title_tag.get_text().strip()
                if dynamic_pattern.search(section_title_raw):
                    section_title_text = clean_content(section_title_tag)
                    section_title_text = re.sub(r'^\s*(?:[0-9]+|[ivxlm]+|[a-z])[\.\s\)\-\:\;\_\*]*\s+', '', section_title_text, flags=re.IGNORECASE)
                    final_section_title_text = '[START_SECTION]' + section_title_text + '[END_SECTION]'
                    section_title_tag.decompose()
                    for tag in section.find_all(['h1', 'h2', 'h3', 'span', 'table', 'figure']):
                        tag.decompose()
                    for tag in section.find_all('div', class_=re.compile(r'ltx_theorem|ltx_proof', re.I)):
                        tag.decompose()
                    section_content_text = clean_content(section)
                    section_content_text = "[START_CONTENT]" + section_content_text + "[END_CONTENT]"
                    full_article_text += (final_section_title_text + section_content_text)

    return clean_title, clean_abstract, full_article_text




