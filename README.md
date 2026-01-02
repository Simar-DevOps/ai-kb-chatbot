# AI Support KB Chatbot (Streamlit)

A lightweight support knowledge-base chatbot that answers questions using **only** the provided documentation and shows **sources** for every answer. Includes **guardrails** (“I don’t know” when docs don’t support), **admin controls**, **feedback logging**, and **basic analytics**.

## Screenshots
![Chat UI](assets/ui-chat.png)
![Sidebar controls](assets/ui-sidebar.png)

## Demo flow (60–90 seconds)
1. Ask: “How do I reset my password?” → answer + sources
2. Ask an out-of-scope question → responds with “I don’t know”
3. Toggle **LLM answering** OFF → retrieval-only excerpts
4. Turn on **Analytics** → top queries + feedback rate

## What it does
- Loads support docs from `data/raw/` (Markdown)
- Retrieves the most relevant chunks (BM25)
- Answers with:
  - **LLM ON:** grounded answer + citations to sources
  - **LLM OFF:** retrieval-only excerpts
- Guardrails: if docs don’t support the question → **no LLM call**
- Feedback: 👍 / 👎 logged locally
- Analytics: logs questions + shows top queries + feedback rate
This repo ships with dummy docs only; local logs/feedback are ignored.

## Repo structure
- `app/app.py` — Streamlit app UI + chat + analytics panel
- `src/` — retrieval, answerer, guardrails, settings, feedback storage
- `data/raw/` — sample support KB docs (dummy)
- `docs/` — product/rollout artifacts (Use Case Brief, PRD-lite, Playbook)

## Quickstart
### 1) Setup environment
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Tested on Python 3.14.2. If you run into install issues, try Python 3.12+.
