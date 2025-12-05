import streamlit as st
import os
import fitz # PyMuPDF
import re
from dotenv import load_dotenv
load_dotenv()

from modules.ingestion import ingest_pdf
from modules.graph import app_graph
from streamlit_pdf_viewer import pdf_viewer

st.set_page_config(page_title="Agentic PDF System", layout="wide")

# --- ROBUST HIGHLIGHTER (Fixed for Rectangles) ---
def get_highlight_coordinates(pdf_path, text_snippet, page_num):
    """
    Finds text using a flexible 'Bag of Words' approach.
    Uses Standard Rectangles (x0, y0) to prevent AttributeErrors.
    """
    if not text_snippet: return []
    doc = fitz.open(pdf_path)
    # Safety Check
    if page_num < 1 or page_num > len(doc): return []
    page = doc[page_num - 1]
    
    annotations = []
    
    # 1. Clean the snippet
    clean_snippet = re.sub(r'[^\w\s]', '', text_snippet).lower()
    words = clean_snippet.split()
    
    # 2. Search Strategy: Try varying lengths of phrases
    search_terms = []
    if len(words) >= 15: search_terms.append(" ".join(words[:15]))
    if len(words) >= 10: search_terms.append(" ".join(words[:10]))
    if len(words) >= 5:  search_terms.append(" ".join(words[:5]))
    if len(words) >= 3:  search_terms.append(" ".join(words[:3])) # Last resort
    
    # Fallback if text is very short
    if not search_terms:
        search_terms.append(clean_snippet)

    # 3. Execute Search
    for term in search_terms:
        # CRITICAL FIX: Use quads=False to get Rectangles (safer)
        rects = page.search_for(term, quads=False)
        
        if rects:
            for rect in rects:
                annotations.append({
                    "page": page_num,
                    "x": rect.x0, 
                    "y": rect.y0,
                    "width": rect.width, 
                    "height": rect.height,
                    "color": "red",
                    "opacity": 0.5 
                })
            break # Stop if we found a match
            
    return annotations

# --- SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_db_ready" not in st.session_state: st.session_state.vector_db_ready = False
if "file_map" not in st.session_state: st.session_state.file_map = {} 
if "annotations" not in st.session_state: st.session_state.annotations = []
if "current_page" not in st.session_state: st.session_state.current_page = 1
if "show_all_pages" not in st.session_state: st.session_state.show_all_pages = True
if "current_file_name" not in st.session_state: st.session_state.current_file_name = None

# --- UI ---
st.title("🤖 Agentic PDF Reasoning System")
st.caption("Multi-Doc Support | Planner | RAG -> Reason Chain")

with st.sidebar:
    st.header("📂 Multi-Doc Navigator")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.button("🚀 Process All Docs"):
        with st.spinner("Indexing all documents..."):
            if not os.path.exists("temp_pdf"): os.makedirs("temp_pdf")
            st.session_state.file_map = {} 
            for file in uploaded_files:
                temp_path = os.path.join("temp_pdf", file.name)
                with open(temp_path, "wb") as f: f.write(file.getbuffer())
                st.session_state.file_map[file.name] = temp_path
            
            if uploaded_files: st.session_state.current_file_name = uploaded_files[0].name

            ingest_pdf(uploaded_files)
            st.session_state.vector_db_ready = True
            st.success(f"Processed {len(uploaded_files)} Documents!")

    if st.session_state.file_map:
        st.divider()
        st.write("### 📄 Viewer")
        file_list = list(st.session_state.file_map.keys())
        
        # Safe Selectbox
        default_idx = 0
        if st.session_state.current_file_name in file_list:
            default_idx = file_list.index(st.session_state.current_file_name)
            
        selected_file = st.selectbox("Select Document:", file_list, index=default_idx)
        
        if selected_file != st.session_state.current_file_name:
            st.session_state.current_file_name = selected_file
            st.session_state.annotations = [] 
            st.session_state.current_page = 1
            st.rerun()

        current_path = st.session_state.file_map[st.session_state.current_file_name]

        if st.button("👁️ Show Full Document"):
            st.session_state.show_all_pages = True
            st.session_state.annotations = [] 
            st.rerun()
        
        # Viewer Logic
        viewer_params = {
            "input": current_path,
            "height": 600,
            "annotations": st.session_state.annotations or []
        }
        if not st.session_state.show_all_pages:
            viewer_params["pages_to_render"] = [st.session_state.current_page]
            
        pdf_viewer(**viewer_params)
        
        if not st.session_state.show_all_pages:
            st.caption(f"📍 Page {st.session_state.current_page} of {st.session_state.current_file_name}")

# --- CHAT LOOP ---
for msg_index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "documents" in msg and msg["documents"]:
            st.write("---")
            st.write("**📚 Supporting Evidence:**")
            
            # Show up to 10 references
            for i, doc in enumerate(msg["documents"][:10]):
                score = doc.metadata.get("score", "N/A")
                chunk_id = doc.metadata.get("chunk_id", "Unknown")
                source = doc.metadata.get("source_document", "Doc")
                page_num = doc.metadata.get("page_number", 1)
                
                # Show CLEAN text in the UI
                with st.expander(f"Ref {i+1}: {source} (Page {page_num})"):
                    # Score Visualization
                    try:
                        score_float = float(score)
                        # Invert score for bar (lower distance = higher relevance)
                        st.progress(max(0.0, min(1.0, 1.0 - score_float)), text=f"Relevance: {score}")
                    except:
                        st.caption(f"Score: {score}")
                    
                    st.caption(f"**ID:** `{chunk_id}`")
                    
                    # Clean up newlines for display so it looks like a paragraph
                    clean_display_text = doc.page_content.replace("\n", " ")
                    st.text(f"{clean_display_text[:300]}...")
                    
                    unique_key = f"btn_{msg_index}_{chunk_id}_{i}"
                    
                    if st.button(f"🔍 Highlight", key=unique_key):
                        st.session_state.current_file_name = source
                        st.session_state.current_page = page_num
                        st.session_state.show_all_pages = False 
                        
                        target_path = st.session_state.file_map.get(source)
                        if target_path:
                            coords = get_highlight_coordinates(target_path, doc.page_content, page_num)
                            st.session_state.annotations = coords
                            st.rerun()

if prompt := st.chat_input("Ask about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if not st.session_state.vector_db_ready: st.error("Please upload PDFs first."); st.stop()

    with st.chat_message("assistant"):
        with st.status("🧠 Planner working...", expanded=True) as status:
            active_files = list(st.session_state.file_map.keys())
            initial_state = {
                "question": prompt, "messages": [], "documents": [], "intent": "",
                "file_names": active_files
            }
            result = app_graph.invoke(initial_state)
            
            st.write(f"Intent detected: **{result['intent'].upper()}**")
            if result['intent'] == 'reason':
                st.write("🔗 **Chain Activated:** RAG Agent (Fetch) ➡️ Reasoning Agent (Logic)")
            
            status.update(label="Complete", state="complete", expanded=False)

        final_ans = result["messages"][0]
        st.markdown(final_ans)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_ans, 
            "documents": result.get("documents", [])
        })
        st.rerun()