from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, NarrativeText
import spacy
import yaml

# --- split the PDF into chunks & save it (using unstructured) --- #
elements = partition_pdf(filename=Path(__file__).resolve().parent.parent/ "openai27052023.pdf", strategy="auto")
# build dtm, upsert vectors, etc.
paragraphs = ""
for el in elements:
    if isinstance(el, Title):
        paragraphs += "<TITLE>"
    if isinstance(el, NarrativeText):
        el_as_str = str(el).strip()
        if " " in el_as_str and not el_as_str.startswith("["):
            paragraphs += el_as_str
paragraphs = [p for p in paragraphs.split("<TITLE>") if p]
nlp = spacy.load("en_core_web_sm")  #  use this as a sentencizer
sentences_by_paragraph: list[list[str]] = [
    [sent.text for sent in nlp(p).sents]
    for p in paragraphs
]
trigrams_by_paragraph: list[list[str]] = [
    [f"{sentences[i]} {sentences[i+1]} {sentences[i+2]}" for i in range(len(sentences)-2)]
    for sentences in sentences_by_paragraph
]
# just flatten it out
sentences: list[str] = [
    sent
    for sentences in trigrams_by_paragraph
    for sent in sentences
]
# extract the title
title_element = [el for el in elements if isinstance(el, Title)][0]
title = str(title_element).strip()

with open(Path(__file__).parent / "openai27052023.yaml", "w") as f:
    yaml.dump({"title": title, "sentences": sentences}, f)
