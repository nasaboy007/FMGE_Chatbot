Disclaimer:
This is an educational tool designed exclusively for medical students preparing for academic examinations.
This software is not a substitute for professional medical advice, diagnosis, or treatment. Never use this AI to make clinical decisions or prescribe medications to real patients. Always adhere to official NMC/ICMR guidelines and consult a licensed healthcare professional.


# ⚕️ FMGE Medical AI Assistant (RAG Pipeline)

A high-performance, local Retrieval-Augmented Generation (RAG) chatbot designed specifically for medical students preparing for the Foreign Medical Graduate Examination (FMGE). 

This tool ingests official medical textbooks (PDFs), processes them locally into a vector database, and uses a locally hosted Large Language Model (LLM) to generate highly accurate, zero-hallucination medical answers based strictly on NMC/ICMR guidelines.

## ✨ Key Features
* **Dual-Format Output:** AI automatically structures responses into two parts: a strict **Technical & Clinical Breakdown** and a **Simplified Concept** for easy memorization.
* **Strict Source Masking:** Configured with advanced prompt engineering to act naturally while preventing hallucinations.
* **Multi-Modal UI:** Automatically extracts and displays relevant PDF textbook pages (with a toggleable image viewer) alongside the text response.
* **Conversational Memory:** AI remembers the last 4 chat interactions for seamless follow-up questions (e.g., "Elaborate on that last point").
* **Export Q&A:** Download specific chat outputs as `.txt` files with exact metadata (Book, Author, Page Number, and Inference Time).
* **Blazing Fast Local Inference:** Optimized for 8GB VRAM GPUs using the `gemma4:e4b` model with real-time text streaming and sub-3-second latency.
* **Ngrok Integration:** Built-in secure tunneling to easily share the running app with friends/study groups on mobile devices.

---

## 💻 Hardware & Software Requirements

### Hardware Requirements
* **GPU:** NVIDIA GPU with at least **8GB VRAM** (e.g., RTX 3060, 4060) is highly recommended for real-time text generation. 
* **Storage:** SSD with at least 10GB of free space (for the ChromaDB persistent database, PDF image caching, and LLM weights).

### Software Prerequisites
1. **Python 3.10+**
2. **Ollama:** You must have [Ollama](https://ollama.com/) installed and running on your machine to serve the local AI models.

---
📂 Project Structure
Plaintext
FMGE_Chatbot/
│
├── app5.py                 # Main Streamlit application script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── medical_rag_db/         # Auto-generated persistent ChromaDB folder
└── page_images/            # Auto-generated cache for PDF page screenshots

---

## 🛠️ Setup & Installation

1. Clone the repository:

git clone https://github.com/nasaboy007/FMGE_Chatbot.git
cd FMGE_Chatbot

2. Create and activate a virtual environment (Recommended):

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

3. Install Python dependencies:
pip install -r requirements.txt

4. Download the necessary LLM via Ollama:
ollama pull gemma4:e4b

5. Run:
streamlit run app5.py
or
streamlit run <path of app5.py>

🌐 How to Share with Friends (Ngrok)
If you want to let your friends test the app on their phones without deploying it to a cloud server:

Create a free account at ngrok.com and copy your AuthToken.

Open app5.py in your code editor.

Locate the Sidebar section (around line 140).

Uncomment the Ngrok lines and paste your token:

Python
public_url = start_ngrok_tunnel(auth_token="YOUR_NGROK_TOKEN_HERE")
st.success(f"🌐 **Public Link:** {public_url}")

Save the file. The Streamlit UI will display a secure public link (e.g., https://a1b2c3d4.ngrok-free.app). Send this link to your friends!

or 

you can directly visit ngrok website and take it from there.
