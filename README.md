🧠 Agentic PDF Reasoning System

A smart AI assistant that plans, retrieves, and reasons across your PDF documents.

📖 What does this do?

Unlike standard "Chat with PDF" tools that just search for keywords, this system uses an Autonomous Planner. It understands your intent and picks the best strategy to answer you.

You ask for a Fact: It acts like a Librarian (Retrieves specific data).

You ask for a Summary: It acts like an Editor (Reads all documents and synthesizes them).

You ask for Logic: It acts like an Analyst (Retrieves data first, then uses a separate reasoning step to compare or analyze it).

✨ Key Features

1. The "Planner" (Brain)

The system doesn't just guess. It classifies your request into one of three intents: RAG, Summarize, or Reason. This ensures the right tool is used for the job.

2. Agent Chaining

For complex tasks (like "Compare Document A and B"), the system creates a chain:
Planner → Retrieve Data → Reasoning Engine → Final Answer.

3. Precision Highlighting

Most PDF tools fail to highlight text if the PDF formatting is messy. I built a "Fuzzy Anchor" search engine that finds the exact location of the text on the page, draws a red box around it, and jumps the viewer directly to that page.

4. Multi-Document Intelligence

The system can handle multiple PDFs at once. It uses Dynamic Retrieval Budgeting to ensure it reads a fair amount of text from every uploaded document, preventing one long document from overpowering the others.

🛠️ Tech Stack

Frontend: Streamlit (Custom PDF Viewer Integration)

Orchestration: LangGraph (State-based Agent Workflow)

Vector DB: ChromaDB (Local & Persistent)

Embeddings: BAAI/bge-small-en-v1.5 (High-performance open-source model)

LLM: Google Gemini Pro

PDF Parsing: PyMuPDF (Fitz) & PyMuPDF4LLM

🚀 How to Run Locally

Clone the Repository

git clone [https://github.com/YOUR_USERNAME/Agentic-PDF-RAG.git](https://github.com/YOUR_USERNAME/Agentic-PDF-RAG.git)
cd AgenticPDF


Set Up Virtual Environment

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate


Install Dependencies

pip install -r requirements.txt


Add Your API Key
Create a .env file in the root folder and add:

GOOGLE_API_KEY=your_gemini_api_key_here


Launch the App

streamlit run app.py
