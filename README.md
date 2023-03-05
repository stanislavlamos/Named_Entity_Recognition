# Named Entity Recognition

The project focuses on Named Entity Recognition (NER) task using various approaches in Python programming language.
Implemented algorithms are:
1. training own spaCy model to classify entities
2. using dictionary-based approach
3. detecting entities with fasttext word embeddings and classifying them using cosine similarity

In this project, we used two datasets: conll2003 and OntoNotes5.0.

# Project structure
- [`spacy_job/spacy_job.py`](spacy_job/spacy_job.py) => implementation of training own spaCy model to classify entities
- [`dictionary_job.py`](dictionary_job.py) => dictionary-based approach to tackle NER task
- [`fasttext_job.py`](fasttext_job.py) => using fasttext and cosine similarity to detect entities
- [`final_report.pdf`](final_report.pdf) => report describing implemented algorithms and theory behind it in detail
