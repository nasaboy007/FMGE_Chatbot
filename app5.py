import streamlit as st
import ollama
import chromadb
import fitz  
import os
import time
import warnings
import base64
from functools import lru_cache
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. ENVIRONMENT & OPTIMIZATION SETTINGS
# ==========================================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

@lru_cache(maxsize=100)
def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# ==========================================
# 2. INITIALIZATION & HELPERS
# ==========================================
@st.cache_resource
def init_system():
    client = chromadb.PersistentClient(path="./medical_rag_db")
    collection = client.get_or_create_collection(name="medical_docs")
    embedder = SentenceTransformer('all-MiniLM-L6-v2') 
    
    if not os.path.exists("page_images"):
        os.makedirs("page_images")
        
    return collection, embedder

collection, embedder = init_system()

def get_system_status():
    try:
        img_count = len([f for f in os.listdir("page_images") if f.endswith('.png')])
    except FileNotFoundError:
        img_count = 0
    doc_count = collection.count()
    return img_count, doc_count

# ==========================================
# 3. STATE MANAGEMENT
# ==========================================
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 4. SYSTEM PROMPT
# ==========================================
SYS_PROMPT = """You are an elite Medical AI Tutor and Clinical Expert, designed specifically to help medical students prepare for the FMGE (Foreign Medical Graduate Examination).

Your primary objective is to deliver highly accurate, high-yield medical knowledge based STRICTLY on the retrieved context, adhering to NMC/ICMR guidelines where applicable.

RESPONSE STRUCTURE:
Whenever you answer a clinical or conceptual question, divide your response into these two exact sections:
1. 🩺 Technical & Clinical Breakdown: Provide a highly detailed, medically precise explanation. Use exact anatomical, pharmacological, and pathological terminology. Focus on high-yield facts.
2. 💡 Simplified Concept: Break the concept down into plain English. Use simple analogies or logical deductions to help the student internalize the information.

LANGUAGE RULES:
- Always respond in English by default, as the FMGE is an English-medium exam.
- Only respond in another language if the user explicitly requests it.
- Do not automatically switch languages just because the user's prompt contains a few foreign words.

SOURCE MASKING & SAFETY (CRITICAL):
- Act as if you inherently know this information. DO NOT mention documents, files, sources, retrieval systems, context, or knowledge bases.
- NEVER start your response with phrases like "Based on the text," "According to the provided documents," or "Here is the information."
- Do not fabricate medical facts, drug dosages, or guidelines. 
- If the retrieved data does not contain the answer, state clearly and professionally that you cannot verify the answer based on the current curriculum, rather than inventing facts.

CONVERSATION MEMORY:
If the user asks a follow-up question, seamlessly use the conversation history to understand the context and expand on it naturally.
"""

# ==========================================
# 5. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Medical RAG Assistant", layout="wide")
st.title("⚕️ FMGE AI Assistant")
st.caption("Made by prends")
st.caption("AI is not fully accurate and please understand that it can make mistakes")
img_count, doc_count = get_system_status()
if doc_count > 0:
    st.success(f"🟢 **System Ready:** {doc_count} pages and {img_count} images loaded.")
else:
    st.warning("🟠 **Database Empty:** Please upload PDFs to get started.")

# ==========================================
# 6. SIDEBAR (Ingestion & Management)
# ==========================================
with st.sidebar:
    st.header("📁 Knowledge Management")
    
    uploaded_files = st.file_uploader(
        "Upload Medical PDFs", 
        type=['pdf'], 
        accept_multiple_files=True, 
        disabled=st.session_state.is_processing
    )
    
    if st.button("Process & Ingest Batch", disabled=st.session_state.is_processing):
        if uploaded_files:
            batch_progress = st.progress(0)
            for i, f in enumerate(uploaded_files):
                with st.spinner(f"Ingesting ({i+1}/{len(uploaded_files)}): {f.name}"):
                    tmp = f"temp_{f.name}"
                    with open(tmp, "wb") as wb: wb.write(f.getbuffer())
                    
                    doc = fitz.open(tmp)
                    author = doc.metadata.get("author", "Unknown") if doc.metadata else "Unknown"
                    if not author or author.strip() == "":
                        author = "Unknown"

                    for p_num, page in enumerate(doc):
                        text = page.get_text().strip()
                        
                        img_path = f"page_images/{f.name}_p{p_num}.png"
                        pix = page.get_pixmap()
                        pix.save(img_path)
                        
                        if len(text) > 50:
                            embedding = embedder.encode(text).tolist()
                            collection.add(
                                ids=[f"{f.name}_p{p_num}"],
                                embeddings=[embedding],
                                documents=[text],
                                metadatas=[{
                                    "source": f.name, 
                                    "page": p_num + 1, 
                                    "img_path": img_path,
                                    "author": author
                                }]
                            )
                    doc.close()
                    os.remove(tmp)
                batch_progress.progress((i + 1) / len(uploaded_files))
            st.success("✅ Ingestion Complete!")
            st.rerun()

    st.markdown("---")
    st.header("🗑️ Manage Database")
    db_data = collection.get(include=["metadatas"])
    if db_data and db_data.get('metadatas'):
        files = sorted(list(set([m['source'] for m in db_data['metadatas'] if 'source' in m])))
        del_file = st.selectbox("Select file to remove:", files, disabled=st.session_state.is_processing)
        if st.button("Confirm Delete", disabled=st.session_state.is_processing):
            collection.delete(where={"source": del_file})
            st.rerun()

# ==========================================
# 7. CHAT INTERFACE
# ==========================================
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg.get("meta", {})
            source = meta.get("source", "Unknown")
            author = meta.get("author", "Unknown")
            page = meta.get("page", "N/A")
            time_taken = msg.get("time_taken", "N/A")
            
            caption_parts = []
            if source not in ["Unknown", "N/A"]: caption_parts.append(f"**Book:** {source}")
            if author not in ["Unknown", "N/A"]: caption_parts.append(f"**Author:** {author}")
            if page != "N/A": caption_parts.append(f"**Page:** {page}")
            if time_taken != "N/A": caption_parts.append(f"⏱️ {time_taken}s")
                
            if caption_parts:
                st.caption(" | ".join(caption_parts))
            
            export_data = f"QUERY:\n{msg.get('prompt', 'N/A')}\n\nOUTPUT:\n{msg['content']}\n\nTIME TAKEN: {time_taken}s\nSOURCE: {source} (Page {page})"
            st.download_button(label="💾 Export Q&A", data=export_data, file_name=f"QnA_{idx}.txt", key=f"dl_{idx}")
            
            img_path = msg.get("img_path")
            if img_path and os.path.exists(img_path):
                with st.expander("🖼️ View Source Page Image"):
                    st.image(img_path)

if prompt := st.chat_input("Ask a medical question...", disabled=st.session_state.is_processing):
    
    st.session_state.is_processing = True
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical data..."):
            try:
                start_time = time.time()
                
                is_greeting = len(prompt.split()) <= 4 and any(word in prompt.lower() for word in ['hi', 'hello', 'hey', 'how are you'])
                
                if is_greeting:
                    stream = ollama.chat(
                        model='gemma4:E4b', # Tip: Change to 'gemma4:e4b' to completely fix the latency
                        messages=[{'role': 'user', 'content': f"You are a helpful FMGE Medical AI Assistant. The user says: {prompt}. Respond briefly and politely without robotic prefixes."}],
                        options={"temperature": 0.5},
                        keep_alive=-1,
                        stream=True
                    )
                    
                    answer = ""
                    placeholder = st.empty()
                    for chunk in stream:
                        answer += chunk['message']['content']
                        placeholder.markdown(answer + "▌")
                    
                    placeholder.markdown(answer)
                    time_taken = round(time.time() - start_time, 2)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "prompt": prompt,
                        "time_taken": time_taken, 
                        "meta": {"source": "N/A", "author": "N/A", "page": "N/A"}
                    })
                    
                else:
                    query_emb = embedder.encode(prompt).tolist()
                    res = collection.query(query_embeddings=[query_emb], n_results=4)
                    
                    if res['documents'] and res['documents'][0]:
                        context = "\n\n---\n\n".join(res['documents'][0])
                        best_meta = res['metadatas'][0][0] 
                        
                        history = [{'role': 'system', 'content': SYS_PROMPT}]
                        for m in st.session_state.messages[-5:-1]: 
                            history.append({'role': m['role'], 'content': m['content']})
                        
                        history.append({'role': 'user', 'content': f"CONTEXT:\n{context}\n\nQUESTION: {prompt}"})
                        
                        # Execute Fast Streaming Inference
                        stream = ollama.chat(
                            model='gemma4:E4b',
                            messages=history,
                            options={"temperature": 0.2,"top_p":0.95,"top_k":64, "num_predict": 1000},
                            keep_alive=-1,
                            stream=True
                        )
                        
                        answer = ""
                        placeholder = st.empty()
                        for chunk in stream:
                            answer += chunk['message']['content']
                            placeholder.markdown(answer + "▌")
                        
                        placeholder.markdown(answer)
                        time_taken = round(time.time() - start_time, 2)
                        
                        source = best_meta.get('source', 'Unknown')
                        author = best_meta.get('author', 'Unknown')
                        page = best_meta.get('page', 'N/A')
                        
                        caption_parts = []
                        if source not in ["Unknown", "N/A"]: caption_parts.append(f"**Book:** {source}")
                        if author not in ["Unknown", "N/A"]: caption_parts.append(f"**Author:** {author}")
                        if page != 'N/A': caption_parts.append(f"**Page:** {page}")
                        caption_parts.append(f"⏱️ {time_taken}s")
                        
                        st.caption(" | ".join(caption_parts))
                        
                        export_data = f"QUERY:\n{prompt}\n\nOUTPUT:\n{answer}\n\nTIME TAKEN: {time_taken}s\nSOURCE: {source} (Page {page})"
                        st.download_button(label="💾 Export Q&A", data=export_data, file_name=f"QnA_latest.txt", key=f"dl_latest_{time.time()}")
                        
                        if best_meta.get("img_path") and os.path.exists(best_meta["img_path"]):
                            with st.expander("🖼️ View Source Page Image"):
                                st.image(best_meta["img_path"])
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer, 
                            "prompt": prompt,
                            "time_taken": time_taken, 
                            "meta": best_meta,
                            "img_path": best_meta.get("img_path")
                        })
                    else:
                        st.warning("No relevant context found in the database.")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                st.session_state.is_processing = False
                st.rerun()