# Text Summarizer with WSD

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)
![NLP](https://img.shields.io/badge/NLP-spaCy%20%7C%20T5-orange)

An advanced web-based text summarization system that leverages **Word Sense Disambiguation (WSD)** and pre-trained **T5 models** to generate highly accurate and context-aware summaries. 

## 🧠 Overview

Traditional text summarizers often struggle with words that have multiple meanings depending on their surrounding context. This project solves that ambiguity by integrating Word Sense Disambiguation into the summarization pipeline. 

By utilizing **spaCy** for robust natural language processing and **N-gram models** for candidate word selection, the system accurately identifies and contextualizes repeating words with differing meanings. It evaluates multiple generated summaries against each other using **ROUGE scores** to ensure the user only receives the highest quality output.

## ✨ Features

- **Abstractive Summarization**: Leverages pre-trained T5 models to generate human-like summaries rather than just extracting sentences.
- **Word Sense Disambiguation (WSD)**: Intelligently identifies and processes repeating words with different meanings to preserve the original text's true context.
- **N-gram Candidate Selection**: Enhances token selection and text generation logic.
- **Automated ROUGE Evaluation**: Internally compares different summarization results and outputs the optimal summary based on its ROUGE metric.
- **Interactive Web Interface**: A clean, accessible frontend served via Flask.

## 🛠️ Technologies Used

- **Backend**: Python, Flask
- **NLP Libraries**: spaCy, NLTK (for n-grams)
- **Machine Learning**: Pre-trained T5 Models (HuggingFace Transformers)
- **Evaluation Metrics**: ROUGE Score

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed on your local machine.

