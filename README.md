🧠 Agentic PDF Reasoning System

A smart AI assistant that reads, understands, and analyzes your PDFs—just like a human researcher.

📖 What is this?

Most "Chat with PDF" apps are simple: they just look up keywords. This project is different. It is an Autonomous System that thinks before it answers.

Instead of just searching for text, it uses a Planner (The Brain) to decide the best strategy:

Fact Check: Need a specific number? It sends the RAG Agent.

Big Picture: Need a summary? It sends the Summarization Agent.

Complex Logic: Need to compare two documents? It sends the Reasoning Agent to "think" through the data.

✨ Why is this special? (The "Elite" Features)

🧠 1. It Has a Brain (Planner Node)

It doesn't just guess. It analyzes your question first.

You ask: "What is the revenue?" -> It routes to RAG.

You ask: "Compare Q1 and Q2 revenue." -> It routes to Reasoning.

🔗 2. It Can "Think" in Steps (Chaining)

For hard questions, it works like a team:

Step 1: The RAG Agent finds the data.

Step 2: It passes that data to the Reasoning Agent.

Step 3: The Reasoning Agent analyzes it and writes the final answer.

🎯 3. Pinpoint Accuracy (Visual Highlighting)

It doesn't just tell you the answer; it shows you.

The system finds the exact sentence in the PDF.

It draws a Red Box around it.

It jumps the viewer to that specific page.

📚 4. It Reads Tables Correctly

Most PDF tools break when they see a table (they just mash the text together).

My Solution: I used PyMuPDF4LLM to convert PDFs into Markdown.

Result: It understands rows, columns, and headers perfectly.

🛠️ Tech Stack

Component

Technology

Why I chose it

Frontend

Streamlit

Fast, interactive UI with a custom PDF viewer.

The Brain

LangGraph

Allows the AI to loop, think, and change its mind (unlike linear chains).

Memory

ChromaDB

Stores document data locally (Fast & Free).

Search

BAAI/bge-small

A highly accurate, open-source search model.

PDF Reader

PyMuPDF (Fitz)

The only tool precise enough to draw highlight boxes on raw PDF text.

🚀 How to Run It

1. Get the Code

git clone [https://github.com/YOUR_USERNAME/Agentic-PDF-RAG.git](https://github.com/YOUR_USERNAME/Agentic-PDF-RAG.git)
cd AgenticPDF


2. Set Up Environment

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate


3. Install Tools

pip install -r requirements.txt


4. Add Your Key

Create a file named .env and paste your Google Gemini API key:

GOOGLE_API_KEY=your_gemini_api_key_here


5. Launch!

streamlit run app.py


🧠 Why I Built It This Way (Interview Notes)

1. Why Hybrid RAG?

I combined Local Search (on your laptop) with Cloud Reasoning (Gemini).

Benefit: Searching is instant and private. Only the final reasoning step uses the API. This saves money and is much faster.

2. The "Highlighting" Challenge

PDFs are messy. The text you see isn't always the text the computer reads.

My Fix: I wrote a "Fuzzy Anchor" search algorithm. Instead of looking for a perfect paragraph match (which fails often), it looks for unique sentence fragments. This guarantees the red box appears in the right place, even on messy scanned documents.

3. Solving "Retrieval Bias"

When you summarize 3 documents, most AIs just read the first one and ignore the rest.

My Fix: I implemented Dynamic Budgeting. The system counts your files and forces the AI to read an equal number of chunks from every document (e.g., 5 chunks from Doc A, 5 from Doc B).
