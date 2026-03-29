import time
import requests
import pandas as pd
import os
from typing import Dict
from collections import Counter

import streamlit as st
from readability import Document
from bs4 import BeautifulSoup

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# spaCy
import spacy
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Agentic AI System", layout="wide")

# ---------------- HELPER FUNCTIONS ---------------- #

def get_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def fetch_page_html(url: str, wait: float = 2.0, headless: bool = True) -> str:
    driver = get_driver(headless=headless)
    try:
        driver.get(url)
        time.sleep(wait)
        return driver.page_source
    finally:
        driver.quit()


def extract_main_text(html: str) -> Dict[str, str]:
    doc = Document(html)
    title = doc.short_title()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.extract()

    text = soup.get_text(separator=" ", strip=True)

    if len(text) < 200:
        soup = BeautifulSoup(doc.summary(), "html.parser")
        text = soup.get_text(separator=" ", strip=True)

    return {"title": title, "text": text}


def summarize_with_ollama(text: str) -> str:
    prompt = f"Summarize in 5 bullet points:\n{text[:2000]}"
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "tinyllama", "prompt": prompt, "stream": False},
            timeout=60
        )
        return r.json().get("response", "")
    except:
        return ""


def extract_entities(text: str):
    return [{"text": e.text, "label": e.label_} for e in nlp(text).ents]


# ---------------- QUALITY SCORE ---------------- #

def compute_quality_score(text):
    words = text.split()
    word_count = len(words)

    entities = extract_entities(text)
    entity_count = len(entities)
    unique_entities = len(set([e["text"] for e in entities]))

    length_score = min(word_count / 1500, 1)
    density_score = min(entity_count / word_count, 0.05) * 20
    uniqueness_score = unique_entities / (entity_count + 1)

    noise_words = ["Navigation", "Contact", "Menu"]
    noise_count = sum(1 for w in words if w in noise_words)
    cleanliness_score = 1 - min(noise_count / word_count, 0.3)

    score = (
        0.2 * length_score +
        0.3 * density_score +
        0.3 * uniqueness_score +
        0.2 * cleanliness_score
    )

    return round(score, 2)


# ---------------- LOGGER ---------------- #

class AgentLogger:
    def __init__(self):
        self.logs = []

    def log(self, agent, msg):
        entry = f"[{agent}] {msg}"
        print(entry)
        self.logs.append(entry)

    def get_logs(self):
        return self.logs


# ---------------- AGENTS ---------------- #

class RouterAgent:
    def route(self, user_input):
        if user_input.startswith("http"):
            return "scrape"
        else:
            return "search"


class ScraperAgent:
    def run(self, url, logger, headless=True):
        logger.log("ScraperAgent", f"Fetching {url}")
        return fetch_page_html(url, headless=headless)


class ProcessingAgent:
    def run(self, html, logger):
        logger.log("ProcessingAgent", "Extracting content")
        data = extract_main_text(html)
        logger.log("ProcessingAgent", f"Text length={len(data['text'])}")
        return data


class DecisionAgent:
    def evaluate(self, text, logger):
        score = compute_quality_score(text)
        logger.log("DecisionAgent", f"Quality Score={score}")

        if score < 0.3:
            logger.log("DecisionAgent", "Low quality → Reject")
            return "reject", score

        return "continue", score


class InsightAgent:
    def run(self, text, logger):
        logger.log("InsightAgent", "Running NLP + LLM")

        entities = [
            e for e in extract_entities(text)
            if e["label"] in ["PERSON", "ORG", "GPE"]
        ][:50]

        summary = summarize_with_ollama(text)

        if not summary:
            summary = text[:300] + "..."

        return {"summary": summary, "entities": entities}


# ---------------- SIMPLE RAG ---------------- #

def simple_search(query, df):
    if df.empty:
        return "No data available"

    matches = df[df["summary"].str.contains(query, case=False, na=False)]

    if not matches.empty:
        return matches.iloc[0]["summary"]

    return "No relevant information found"


# ---------------- PIPELINE ---------------- #

def run_agent_pipeline(user_input: str, headless=True):
    logger = AgentLogger()

    router = RouterAgent()
    scraper = ScraperAgent()
    processor = ProcessingAgent()
    decision = DecisionAgent()
    insight = InsightAgent()

    task = router.route(user_input)
    logger.log("RouterAgent", f"Task={task}")

    if task == "scrape":
        html = scraper.run(user_input, logger, headless)
        extracted = processor.run(html, logger)

        decision_result, score = decision.evaluate(extracted["text"], logger)

        if decision_result == "reject":
            return None

        insights = insight.run(extracted["text"], logger)

        return {
            "title": extracted["title"],
            "summary": insights["summary"],
            "entities": insights["entities"],
            "logs": logger.get_logs(),
            "quality": score,
            "text": extracted["text"]
        }

    return None


# ---------------- UI ---------------- #

st.title("🧠 Agentic AI Web Intelligence System")

url = st.text_input("Enter URL or Search Query")
headless = st.checkbox("Run in headless mode", True)

os.makedirs("data", exist_ok=True)
file_path = "data/results.csv"

if os.path.exists(file_path):
    df_all = pd.read_csv(file_path)
else:
    df_all = pd.DataFrame()

# ---------------- RUN ---------------- #

if st.button("Run") and url:

    start = time.time()

    if url.startswith("http"):

        result = run_agent_pipeline(url, headless)

        if result:

            summary = result["summary"]
            entities_str = "; ".join([e["text"] for e in result["entities"]])

            new_data = pd.DataFrame([{
                "title": result["title"],
                "summary": summary,
                "quality_score": result["quality"],
                "entities": entities_str,
                "text": result["text"]
            }])

            if not df_all.empty:
                df_all = pd.concat([df_all, new_data], ignore_index=True)
            else:
                df_all = new_data

            df_all.to_csv(file_path, index=False)

            st.success("Saved to CSV")

            # -------- OUTPUT -------- #

            st.subheader("Summary")
            st.write(summary)

            st.subheader("Entities")
            st.table(result["entities"])

            # -------- DASHBOARD -------- #

            end = time.time()  

            st.markdown("---")
            st.subheader("📊 Insights Dashboard")

            col1, col2 = st.columns(2)
            col1.metric("Quality Score", result["quality"])
            col2.metric("Execution Time", f"{round(end-start,2)} sec")

            if result["entities"]:
                labels = [e["label"] for e in result["entities"]]

                st.subheader("Entity Distribution")
                st.bar_chart(Counter(labels))

                noise_words = ["Navigation", "Contact", "Menu"]

                clean_entities = [
                    e["text"] for e in result["entities"]
                    if not any(word in e["text"] for word in noise_words)
                ]

                top_entities = Counter(clean_entities).most_common(10)

                st.subheader("🔝 Top Entities")
                df_entities = pd.DataFrame(top_entities, columns=["Entity", "Count"])
                st.table(df_entities)

            # -------- LOGS -------- #

            st.markdown("---")
            st.subheader("🧾 Agent Logs")

            for log in result["logs"]:
                st.text(log)

    else:
        # -------- RAG SEARCH -------- #
        answer = simple_search(url, df_all)

        st.subheader("Search Result")
        st.write(answer)

        end = time.time()
        st.metric("Execution Time", f"{round(end-start,2)} sec")
