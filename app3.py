import streamlit as st
import ollama
import chromadb
import fitz  
import os
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. INITIALIZATION
# ==========================================
@st.cache_resource
def init_system():
    # Persistent storage for RAG
    client = chromadb.PersistentClient(path="./medical_rag_db")
    collection = client.get_or_create_collection(name="medical_docs")
    # Load embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2') 
    return collection, embedder

collection, embedder = init_system()

# ==========================================
# 2. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Medical RAG Assistant", layout="wide")
st.title("⚕️ FMGE AI Assistant")
st.caption("Made by prends")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 3. SIDEBAR (Ingestion & Management)
# ==========================================
with st.sidebar:
    st.header("📁 Knowledge Management")
    
    # --- Batch Ingestion ---
    uploaded_files = st.file_uploader("Upload Medical PDFs", type=['pdf'], accept_multiple_files=True)
    if st.button("Process & Ingest Batch"):
        if uploaded_files:
            batch_progress = st.progress(0)
            for i, f in enumerate(uploaded_files):
                with st.spinner(f"Ingesting ({i+1}/{len(uploaded_files)}): {f.name}"):
                    # Save temp file
                    tmp = f"temp_{f.name}"
                    with open(tmp, "wb") as wb: wb.write(f.getbuffer())
                    
                    # Read PDF
                    doc = fitz.open(tmp)
                    for p_num, page in enumerate(doc):
                        text = page.get_text().strip()
                        if len(text) > 50:
                            embedding = embedder.encode(text).tolist()
                            collection.add(
                                ids=[f"{f.name}_p{p_num}"],
                                embeddings=[embedding],
                                documents=[text],
                                metadatas=[{"source": f.name, "page": p_num + 1}]
                            )
                    doc.close() # CRITICAL: Releases file lock
                    os.remove(tmp)
                batch_progress.progress((i + 1) / len(uploaded_files))
            st.success("✅ Ingestion Complete!")
            st.rerun()
        else:
            st.warning("Please upload files first.")

    # --- Database Management ---
    st.markdown("---")
    st.header("🗑️ Manage Database")
    db_data = collection.get(include=["metadatas"])
    if db_data['metadatas']:
        # Extract unique file names
        files = sorted(list(set([m['source'] for m in db_data['metadatas'] if 'source' in m])))
        del_file = st.selectbox("Select file to remove:", files)
        if st.button("Confirm Delete"):
            with st.spinner("Deleting..."):
                collection.delete(where={"source": del_file})
                st.rerun()
    else:
        st.info("Database is empty.")

# ==========================================
# 4. CHAT INTERFACE
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching database..."):
            try:
                # Retrieve Context
                query_emb = embedder.encode(prompt).tolist()
                res = collection.query(query_embeddings=[query_emb], n_results=3)
                context = "\n\n".join(res['documents'][0]) if res['documents'][0] else "No context found."
                
                # LLM Call
                sys_prompt = "You are a medical assistant. Answer using ONLY the provided context."
                response = ollama.chat(
                    model='gemma4:12b', # Ensure this matches your installed model
                    messages=[{'role': 'user', 'content': f"{sys_prompt}\n\nCONTEXT:\n{context}\n\nQUESTION: {prompt}"}],
                    options={
                        "temperature": 0.2, 
                        "repeat_penalty": 1.2, 
                        "num_predict": 1000 # Prevents infinite loops
                    }
                )
                answer = response['message']['content']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {str(e)}")