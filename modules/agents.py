from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from modules.ingestion import get_vectorstore

# ==========================================
# CONFIGURATION
# ==========================================
# Initialize Gemini Pro (Free & Stable)
# We use temperature=0 for maximum factual accuracy and consistency
llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0)


# ==========================================
# 1. RAG AGENT
# Function: Retrieve facts, deduplicate, and answer specific questions.
# ==========================================
def rag_agent(state):
    """
    Retrieves documents based on the query.
    - If intent is 'rag': Generates an answer.
    - If intent is 'reason': Passes documents to the Reasoning Agent (Chaining).
    """
    question = state["question"]
    intent = state.get("intent", "rag")
    vectorstore = get_vectorstore()
    
    print(f"--- RAG Agent processing: {question} ---")
    
    # 1. RETRIEVAL (Fetch more candidates to allow for filtering)
    # We fetch 10 docs to ensure we have enough after removing duplicates
    results = vectorstore.similarity_search_with_score(question, k=10)
    
    docs = []
    context_parts = []
    seen_content = set() # Track duplicates based on text content
    
    for doc, score in results:
        # Create a "fingerprint" of the first 100 chars to check for duplicates
        content_fingerprint = doc.page_content[:100]
        
        # Skip if we've seen this text before (Deduplication Logic)
        if content_fingerprint in seen_content:
            continue
        
        seen_content.add(content_fingerprint)
        
        # Stop once we have 5 unique, high-quality documents
        if len(docs) >= 5:
            break

        # CRITICAL: Save score to metadata so the UI can display the progress bar
        doc.metadata["score"] = f"{score:.4f}"
        docs.append(doc)
        
        # Format the context for the LLM
        source = doc.metadata.get("source_document", "Unknown")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        
        # Add a clear label so the LLM knows where this text came from
        context_parts.append(f"[Source: {source} | ID: {chunk_id}] {doc.page_content}")
    
    # 2. CHAINING CHECK
    # If the Planner said "Reason", we do NOT answer. We just pass the data.
    if intent == "reason":
        print("--- RAG finished. Passing data to Reasoning Agent. ---")
        return {"documents": docs, "messages": ["Data retrieved by RAG Agent. Handing off to Reasoning Engine..."]}
    
    # 3. ANSWER GENERATION (Standard RAG)
    context_text = "\n\n".join(context_parts)
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the user's question based ONLY on the context below. 
        If you answer, you MUST cite the source document and Chunk ID.
        
        Context:
        {context}
        
        Question: 
        {question}
        """
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": question})
    
    return {"messages": [response], "documents": docs}


# ==========================================
# 2. SUMMARIZATION AGENT
# Function: Smartly select files, fetch chunks, interleave results, and summarize.
# ==========================================
def summarization_agent(state):
    """
    Complex Summarizer: 
    1. Asks LLM which files to focus on.
    2. Dynamically budgets chunks per file.
    3. Fetches chunks WITH scores.
    4. Mixes results (Interleaving) so UI shows A, B, C...
    5. Deduplicates results.
    """
    vectorstore = get_vectorstore()
    question = state["question"]
    all_file_names = state.get("file_names", [])
    
    target_files = []

    # --- PHASE 1: SMART FILE SELECTION ---
    if all_file_names:
        print(f"--- Summarizer identifying target files from: {all_file_names} ---")
        selection_prompt = ChatPromptTemplate.from_template(
            """
            You are a File Selector. 
            User Request: "{question}"
            Available Files: {file_list}
            
            Task: Identify which files the user wants to summarize.
            - If they ask for "all", "everything", "the documents", or don't specify, return the word: ALL
            - If they specify a file (e.g., "summarize the invoice"), return ONLY that filename exactly as it appears in the list.
            - If multiple, return them comma-separated.
            
            Return ONLY the filenames or 'ALL'. No other text.
            """
        )
        chain = selection_prompt | llm | StrOutputParser()
        try:
            response = chain.invoke({"question": question, "file_list": ", ".join(all_file_names)})
            cleaned_response = response.strip().replace("'", "").replace('"', "")
            
            if "ALL" in cleaned_response.upper():
                target_files = all_file_names
                print(f"--- Summarizer Target: ALL FILES ---")
            else:
                # Fuzzy matching to find the correct filename
                suggested_files = [f.strip() for f in cleaned_response.split(",")]
                for f in all_file_names:
                    for suggestion in suggested_files:
                        if suggestion in f or f in suggestion:
                            target_files.append(f)
                            break
                
                # Fallback if matching failed
                if not target_files: 
                    target_files = all_file_names
                
                print(f"--- Summarizer Target: {target_files} ---")
        except:
            target_files = all_file_names
    else:
        target_files = [] # No files uploaded

    # --- PHASE 2: DYNAMIC RETRIEVAL WITH SCORES ---
    # We collect lists of docs per file first: [[DocA_1, DocA_2], [DocB_1, DocB_2]]
    docs_by_file_group = []
    TOTAL_CHUNK_BUDGET = 30
    
    if target_files:
        # Calculate how many chunks we can afford per document
        # e.g., if 3 files, we get 10 chunks each.
        k_per_doc = max(1, TOTAL_CHUNK_BUDGET // len(target_files))
        
        for file in target_files:
            # CRITICAL FIX: Use similarity_search_with_score
            res = vectorstore.similarity_search_with_score(
                question, 
                k=k_per_doc, 
                filter={"source_document": file} # Strict filtering by file
            )
            
            file_group = []
            for doc, score in res:
                doc.metadata["score"] = f"{score:.4f}" # Save Score
                file_group.append(doc)
            
            if file_group:
                docs_by_file_group.append(file_group)
    else:
        # Fallback (Should rarely happen if files are uploaded)
        res = vectorstore.similarity_search_with_score(question, k=20)
        file_group = []
        for doc, score in res:
            doc.metadata["score"] = f"{score:.4f}"
            file_group.append(doc)
        docs_by_file_group.append(file_group)

    # --- PHASE 3: INTERLEAVING & DEDUPLICATION ---
    # Turn [[A1, A2], [B1, B2]] into [A1, B1, A2, B2] so UI shows variety
    all_results = []
    seen_content = set()
    
    if docs_by_file_group:
        max_len = max(len(g) for g in docs_by_file_group)
        for i in range(max_len):
            for group in docs_by_file_group:
                if i < len(group):
                    doc = group[i]
                    # Check Fingerprint (First 100 chars)
                    fingerprint = doc.page_content[:100]
                    if fingerprint not in seen_content:
                        seen_content.add(fingerprint)
                        all_results.append(doc)

    # --- PHASE 4: GENERATE SUMMARY ---
    # We format the context to clearly label which document each chunk comes from
    context_text = "\n\n".join([f"== SOURCE DOC: {d.metadata.get('source_document')} ==\n{d.page_content}" for d in all_results])
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert summarizer. 
        Create a comprehensive summary based on the provided context.
        
        Instructions:
        1. Only summarize the documents provided in the context below.
        2. If the user asked for a specific document, focus ONLY on that.
        3. Explicitly mention the document names in your summary.
        4. Structure with clear headings and bullet points.
        
        Context from documents:
        {context}
        """
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_text})
    
    return {"messages": [response], "documents": all_results}


# ==========================================
# 3. REASONING AGENT
# Function: Multi-step logic using data ALREADY retrieved by RAG.
# ==========================================
def reasoning_agent(state):
    print("--- Reasoning Agent Activated ---")
    question = state["question"]
    # We use the documents passed from the RAG agent (Chaining)
    docs = state["documents"] 
    
    context_text = "\n\n".join([f"[{d.metadata.get('source_document')}] {d.page_content}" for d in docs])

    prompt = ChatPromptTemplate.from_template(
        """You are a specialized Reasoning Agent. 
        The user has asked for a complex analysis (Comparison, Timeline, Aggregation, or Logic).
        
        Reference the context provided by the RAG agent.
        If comparing, list similarities and differences.
        If creating a timeline, order events chronologically.
        
        Context:
        {context}
        
        User Request: 
        {question}
        """
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": question})
    
    return {"messages": [response]} # Note: We don't return 'documents' here because they are already in state