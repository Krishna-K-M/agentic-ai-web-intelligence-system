# 🧠 Agentic AI Web Intelligence & Data Analytics System

## 🚀 Overview
This project is a multi-agent AI system that extracts, evaluates, and analyzes web content to generate structured insights.

It combines web scraping, NLP, LLM-based summarization, and analytics into a unified intelligent pipeline. The system also supports retrieval-based querying (RAG-like) over previously processed data.

---

## 🧠 Key Features
- Multi-agent architecture (Router, Scraper, Processing, Decision, Insight)
- Intelligent web scraping using Selenium + BeautifulSoup
- NLP-based entity extraction using spaCy
- LLM-based summarization using Ollama (tinyllama)
- Content quality scoring using multi-factor evaluation
- RAG-like retrieval system for querying past data
- Interactive Streamlit dashboard with analytics
- CSV-based data storage (Power BI ready)

---

## 🏗️ System Architecture

User Input  
      ↓  
Router Agent (decides task: scrape or search)  
      ↓  
Scraper Agent (Selenium-based web scraping)  
      ↓  
Processing Agent (content extraction & cleaning)  
      ↓  
Decision Agent (content quality evaluation)  
      ↓  
Insight Agent (NLP entity extraction + LLM summarization)  
      ↓  
Storage (CSV / data persistence)  
      ↓  
Dashboard (Streamlit analytics & visualization)  
      ↓  
RAG-like Retrieval (search over past data)

---

## ⚙️ Tech Stack
Python, Streamlit, Selenium, BeautifulSoup, spaCy, Ollama, Pandas, Power BI

---

## ⚙️ Setup & Installation

### Install dependencies
pip install -r requirements.txt

### Install spaCy model
python -m spacy download en_core_web_sm

---

## 🤖 Ollama Setup

Install Ollama from: https://ollama.com

Run:
ollama serve

Pull model:
ollama pull tinyllama

---

## ▶️ Run the Application

streamlit run app.py

---

## 📁 Project Structure

project-root/
|
├── app.py # Main Streamlit application
|
├── requirements.txt # Dependencies
|
├── README.md # Project documentation
|
├── data/
│ └── results.csv # Stored outputs (auto-generated)
|
└── .gitignore

---

## 👨‍💻 Author
Krishna
