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
    client = chromadb.PersistentClient(path="./medical_rag_db")
    collection = client.get_or_create_collection(name="medical_docs")
    embedder = SentenceTransformer('all-MiniLM-L6-v2') 
    
    # Ensure image directory exists
    if not os.path.exists("page_images"):
        os.makedirs("page_images")
        
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
    
    uploaded_files = st.file_uploader("Upload Medical PDFs", type=['pdf'], accept_multiple_files=True)
    if st.button("Process & Ingest Batch"):
        if uploaded_files:
            batch_progress = st.progress(0)
            for i, f in enumerate(uploaded_files):
                with st.spinner(f"Ingesting ({i+1}/{len(uploaded_files)}): {f.name}"):
                    tmp = f"temp_{f.name}"
                    with open(tmp, "wb") as wb: wb.write(f.getbuffer())
                    
                    doc = fitz.open(tmp)
                    for p_num, page in enumerate(doc):
                        text = page.get_text().strip()
                        
                        # Save Page Image
                        img_path = f"page_images/{f.name}_p{p_num}.png"
                        pix = page.get_pixmap()
                        pix.save(img_path)
                        
                        if len(text) > 50:
                            embedding = embedder.encode(text).tolist()
                            collection.add(
                                ids=[f"{f.name}_p{p_num}"],
                                embeddings=[embedding],
                                documents=[text],
                                metadatas=[{"source": f.name, "page": p_num + 1, "img_path": img_path}]
                            )
                    doc.close()
                    os.remove(tmp)
                batch_progress.progress((i + 1) / len(uploaded_files))
            st.success("✅ Ingestion Complete!")
            st.rerun()

    st.markdown("---")
    st.header("🗑️ Manage Database")
    db_data = collection.get(include=["metadatas"])
    if db_data['metadatas']:
        files = sorted(list(set([m['source'] for m in db_data['metadatas'] if 'source' in m])))
        del_file = st.selectbox("Select file to remove:", files)
        if st.button("Confirm Delete"):
            # Also clean up images if you want (Optional, complex)
            collection.delete(where={"source": del_file})
            st.rerun()

# ==========================================
# 4. CHAT INTERFACE
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "img_path" in msg and os.path.exists(msg["img_path"]):
            st.image(msg["img_path"], caption="Retrieved Page Context")

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching database..."):
            try:
                query_emb = embedder.encode(prompt).tolist()
                res = collection.query(query_embeddings=[query_emb], n_results=1) # Reduced to 1 for precise image
                
                if res['documents'][0]:
                    context = res['documents'][0][0]
                    metadata = res['metadatas'][0][0]
                    img_path = metadata.get("img_path")
                    
                    response = ollama.chat(
                        model='gemma4:12b',
                        messages=[{'role': 'user', 'content': f"Context: {context}\n\nQuestion: {prompt}"}],
                        options={"temperature": 0.2, "num_predict": 1000}
                    )
                    answer = response['message']['content']
                    
                    # Display Answer
                    st.markdown(answer)
                    
                    # Display Image
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=f"Source: {metadata['source']} (Page {metadata['page']})")
                        st.session_state.messages.append({"role": "assistant", "content": answer, "img_path": img_path})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.warning("No relevant context found.")
            except Exception as e:
                st.error(f"Error: {str(e)}")