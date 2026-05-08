import os
import shutil
import tempfile
import streamlit as st
import uuid
from typing import List
from dotenv import load_dotenv

import database

# Load environment variables from .env file
load_dotenv()

PUBLIC_DIR = os.path.join("data", "public")
ADMIN_DIR = os.path.join("data", "admin")
os.makedirs(PUBLIC_DIR, exist_ok=True)
os.makedirs(ADMIN_DIR, exist_ok=True)

database.init_db()

# Langchain Imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ==========================================
# 1. Document Ingestion & File Mgmt
# ==========================================
def save_uploaded_files(uploaded_files, target_dir, visibility='public'):
    """Saves Streamlit uploaded files to the specified directory and database."""
    saved_count = 0
    for uploaded_file in uploaded_files:
        file_path = os.path.join(target_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        database.add_document(uploaded_file.name, visibility=visibility)
        saved_count += 1
    return saved_count

def load_directory_documents(directory_path) -> List[Document]:
    """
    Loads all supported documents from a directory using LangChain document loaders.
    """
    documents = []
    if not os.path.exists(directory_path):
        return documents
        
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        file_extension = os.path.splitext(filename)[1].lower()
            
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_extension == ".docx":
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file_extension == ".txt":
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            
    return documents

# ==========================================
# 2. Chunking
# ==========================================
def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Chunks documents into 300-word segments with a 50-word overlap.
    Uses RecursiveCharacterTextSplitter with a custom length function based on word count.
    """
    # Custom length function to split by word count
    def word_count(text: str) -> int:
        return len(text.split())
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=word_count,
        separators=["\n\n", "\n", " ", ""] # Standard separators
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

# ==========================================
# 3. Embedding & Vector Store
# ==========================================
@st.cache_resource
def get_public_vector_store(_chunks: List[Document]):
    """
    Public vectors mapping
    """
    if not _chunks:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(_chunks, embeddings)
    return vector_store

@st.cache_resource
def get_admin_vector_store(_chunks: List[Document]):
    """
    Admin vectors mapping
    """
    if not _chunks:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(_chunks, embeddings)
    return vector_store

# ==========================================
# 4. Retrieval & Generation
# ==========================================
def build_rag_chain(vector_store, groq_api_key: str):
    """
    Builds the retrieval and generation pipeline.
    Retrieves top 5 chunks and passes them to Llama 3 via Groq.
    Enforces a strict system prompt.
    """
    # Initialize Llama 3 model via Groq API
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="llama-3.1-8b-instant", # Updated model because previous was decommissioned
        temperature=0.0 # Keep temperature 0 for strictly factual responses
    )
    
    # Strict System Prompt but with ChatGPT-style Formatting Instructions
    system_prompt = (
        "You are an expert AI assistant providing detailed, structured, and highly readable answers. "
        "Your task is to answer the user's question ONLY using the provided context. "
        "If the answer is not contained in the context, say 'I cannot answer this based on the provided context' and stop. "
        "Do not make up information or use external knowledge. \n\n"
        "When you construct your answer, please use the following guidelines to mimic a premium ChatGPT-like response:\n"
        "- Provide a comprehensive and detailed explanation based entirely on the context.\n"
        "- Use Markdown formatting (bolding key terms, using bullet points or numbered lists where appropriate).\n"
        "- If the context allows, break down the answer logically into sections like 'Simple explanation', 'Key features', or 'Examples'.\n\n"
        "Context:\n{context}"
    )
    
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=f"{system_prompt}\n\nQuestion: {{input}}\nAnswer:"
    )
    
    # Create chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Retrieve top 5 most relevant chunks using cosine similarity
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# ==========================================
# 5. Validation / Evaluation Metrics
# ==========================================
def evaluate_pipeline_ragas():
    """
    Placeholder function for evaluating the pipeline using RAGAS metrics.
    Specifically targeting a faithfulness score > 0.75.
    
    Future implementation:
    1. Collect a dataset of (question, answer, contexts).
    2. Use ragas metrics (faithfulness, answer_relevancy, context_precision, context_recall).
    3. Assert faithfulness_score > 0.75.
    """
    st.info("RAGAS Evaluation Placeholder: Pipeline tracked specifically for Faithfulness score > 0.75.")
    pass

# ==========================================
# Main Streamlit UI
# ==========================================
def main():
    st.set_page_config(page_title="HalluRAG Phase 1", page_icon="📝", layout="wide")
    
    # Custom CSS for One Health Dashboard feel
    st.markdown("""
        <style>
        /* Main page layout */
        .block-container {
            padding-top: 2rem;
            max-width: 1000px;
            margin: 0 auto;
        }
        
        /* Typography */
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* Primary Buttons styling */
        .stButton > button[kind="primary"] {
            background-color: #6c51ff !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(108, 81, 255, 0.2) !important;
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 12px rgba(108, 81, 255, 0.4) !important;
            transform: translateY(-1px) !important;
            background-color: #5b42d1 !important;
        }
        
        /* Secondary Buttons / History items */
        .stButton > button[kind="secondary"] {
            background-color: transparent !important;
            color: #d1d5db !important;
            border-radius: 8px !important;
            border: 1px solid transparent !important;
            padding: 0.5rem 0.5rem !important;
            font-weight: normal !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button[kind="secondary"] p {
            text-align: left !important;
            width: 100%;
        }
        .stButton > button[kind="secondary"]:hover {
            background-color: #2a3158 !important;
            color: #ffffff !important;
        }
        
        /* Inputs styling */
        .stTextInput > div > div > input {
            border-radius: 8px !important;
            border: 1px solid #2a3158 !important;
            background-color: #121936 !important;
            color: #f5f5fa !important;
        }
        
        /* Expander / Cards */
        [data-testid="stExpander"] {
            background-color: #1a2245 !important;
            border-radius: 12px !important;
            border: 1px solid #2a3158 !important;
            overflow: hidden !important;
        }
        
        /* Chat UI */
        .stChatMessage {
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            background-color: #121936;
            border: 1px solid #1e264a;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .stChatMessage:nth-child(even) {
            background-color: #182042;
        }
        .stChatMessageAvatar {
            background-color: transparent !important;
        }
        .stChatInputContainer {
            padding-bottom: 2rem;
            border-top: none;
        }
        
        /* General dividers */
        hr {
            border-color: #2a3158 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("HalluRAG: Document Q&A Pipeline")
    
    # Initialize session state for chat history and vector store
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        # Load previous history if any (usually empty for a new uuid, but good pattern)
        history = database.get_chat_history(st.session_state.session_id)
        if history:
            st.session_state.messages = history
    if "public_vector_store" not in st.session_state:
        st.session_state.public_vector_store = None
    if "admin_vector_store" not in st.session_state:
        st.session_state.admin_vector_store = None
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
        
    groq_api_key = os.environ.get("GROQ_API_KEY")
    admin_pass = os.environ.get("ADMIN_PASSWORD", "admin123")
    
    # Ensure admin user is seeded in the DB
    database.seed_admin_if_needed(admin_pass)
    
    def rebuild_vector_stores():
        with st.spinner("Reading public documents..."):
            docs = load_directory_documents(PUBLIC_DIR)
            get_public_vector_store.clear()
            if docs:
                st.session_state.public_vector_store = get_public_vector_store(chunk_documents(docs))
            else:
                st.session_state.public_vector_store = None

        with st.spinner("Reading admin documents..."):
            admin_docs = list(docs) if docs else []
            admin_docs.extend(load_directory_documents(ADMIN_DIR))
            get_admin_vector_store.clear()
            if admin_docs:
                st.session_state.admin_vector_store = get_admin_vector_store(chunk_documents(admin_docs))
            else:
                st.session_state.admin_vector_store = None
                
        st.success("Vector Stores rebuilt and ready!")

    # Auto-initialize vector stores on startup
    if st.session_state.public_vector_store is None:
        docs = load_directory_documents(PUBLIC_DIR)
        if docs:
            st.session_state.public_vector_store = get_public_vector_store(chunk_documents(docs))
            
    if st.session_state.admin_vector_store is None and st.session_state.is_admin:
        docs = load_directory_documents(PUBLIC_DIR)
        docs.extend(load_directory_documents(ADMIN_DIR))
        if docs:
            st.session_state.admin_vector_store = get_admin_vector_store(chunk_documents(docs))

    def get_current_vector_store():
        return st.session_state.admin_vector_store if st.session_state.is_admin else st.session_state.public_vector_store

    # Sidebar
    with st.sidebar:
        # --- New Chat ---
        if st.button("📝 New chat", use_container_width=True, type="primary"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
            
        # --- Search Chats ---
        search_term = st.text_input("🔍 Search chats", placeholder="Search chats...", label_visibility="collapsed")
        
        # --- History / Recents ---
        st.subheader("Recents")
        sessions = database.get_all_sessions()
        
        if search_term:
            sessions = [s for s in sessions if s['title'] and search_term.lower() in s['title'].lower()]
            
        if sessions:
            for s in sessions:
                title = s['title'] if s['title'] else "Empty Chat"
                if len(title) > 28:
                    title = title[:25] + "..."
                
                # Highlight active session
                btn_type = "primary" if s['session_id'] == st.session_state.session_id else "secondary"
                
                if st.button(title, key=f"hist_{s['session_id']}", use_container_width=True, type=btn_type):
                    st.session_state.session_id = s['session_id']
                    st.session_state.messages = database.get_chat_history(s['session_id'])
                    st.rerun()
        else:
            st.info("No recent chats found.")
            
        st.divider()

        st.header("⚙️ Configuration")
        
        if not groq_api_key or groq_api_key == "paste_your_actual_api_key_here":
            st.warning("⚠️ Please add your exact Groq API Key to the `.env` file.")
            
        st.divider()
        
        # Accessible Documents for User
        st.header("📚 Available Data")
        st.caption("Currently active documents used by the Bot.")
        approved_files = os.listdir(PUBLIC_DIR) if os.path.exists(PUBLIC_DIR) else []
        if st.session_state.is_admin:
            admin_files = os.listdir(ADMIN_DIR) if os.path.exists(ADMIN_DIR) else []
            approved_files.extend([f"[Admin] {f}" for f in admin_files])
            
        if approved_files:
            for a_file in approved_files:
                st.write(f"🔸 {a_file}")
        else:
            st.info("No data available yet.")

        st.divider()

        # Base Users Panel
        st.header("📁 User Uploads")
        st.caption("Upload public documents to immediately start Q&A.")
        uploaded_files = st.file_uploader(
            "Upload files", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        if st.button("Upload and Process", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one document.")
            else:
                count = save_uploaded_files(uploaded_files, PUBLIC_DIR, visibility='public')
                st.success(f"Successfully processed {count} document(s).")
                rebuild_vector_stores()
                
        st.divider()
        
        # Admin Login / Dashboard
        st.header("🛡️ Admin Area")
        if not st.session_state.is_admin:
            pwd = st.text_input("Admin Password", type="password")
            if st.button("Login"):
                user = database.authenticate_user("admin", pwd)
                if user and user.get("role") == "admin":
                    st.session_state.is_admin = True
                    st.success("Logged in as Admin.")
                    st.rerun()
                else:
                    st.error("Incorrect password")
        else:
            if st.button("Logout"):
                st.session_state.is_admin = False
                st.rerun()
                
            st.subheader("Manage Active Documents")
            # Collect all files easily
            all_files = []
            if os.path.exists(PUBLIC_DIR):
                all_files.extend([(f, PUBLIC_DIR) for f in os.listdir(PUBLIC_DIR)])
            if os.path.exists(ADMIN_DIR):
                all_files.extend([(f, ADMIN_DIR) for f in os.listdir(ADMIN_DIR)])
                
            if all_files:
                for a_file, d_path in all_files:
                    prefix = "[Admin] " if d_path == ADMIN_DIR else "[Public] "
                    with st.expander(f"Review: {prefix}{a_file}"):
                        file_path = os.path.join(d_path, a_file)
                        file_extension = os.path.splitext(a_file)[1].lower()
                        try:
                            if file_extension == ".pdf":
                                loader = PyPDFLoader(file_path)
                                docs = loader.load()
                                content = "\n".join([doc.page_content for doc in docs])
                            elif file_extension == ".docx":
                                loader = Docx2txtLoader(file_path)
                                docs = loader.load()
                                content = "\n".join([doc.page_content for doc in docs])
                            elif file_extension == ".txt":
                                loader = TextLoader(file_path)
                                docs = loader.load()
                                content = "\n".join([doc.page_content for doc in docs])
                            else:
                                content = "Unsupported file type."
                            st.text_area("Content Preview", content[:2000] + ("..." if len(content)>2000 else ""), height=150, key=f"preview_{d_path}_{a_file}")
                        except Exception as e:
                            st.error(f"Error loading preview: {e}")
                        
                        if st.button("🗑️ Delete Document", key=f"del_{d_path}_{a_file}", use_container_width=True):
                            os.remove(file_path)
                            database.remove_document(a_file)
                            st.toast(f"Deleted {a_file}")
                            rebuild_vector_stores()
                            st.rerun()
            else:
                st.info("No active documents.")
                
            st.subheader("Direct Admin Upload")
            admin_files = st.file_uploader("Upload additional files", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="admin_uploader")
            if st.button("Upload & Index", use_container_width=True):
                if admin_files:
                    save_uploaded_files(admin_files, ADMIN_DIR, visibility='admin')
                    rebuild_vector_stores()
                else:
                    st.warning("No files uploaded.")
                    
            st.divider()
            if st.button("🔄 Rebuild Main Index"):
                rebuild_vector_stores()
            
            st.divider()
            if st.button("Run Evaluation (Mock)", use_container_width=True):
                evaluate_pipeline_ragas()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # Define avatars
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Message HalluRAG..."):
        # Display user message in chat message container
        st.chat_message("user", avatar="🧑‍💻").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        database.add_chat_message(st.session_state.session_id, "user", prompt)
        
        # Check prerequisites
        if not groq_api_key or groq_api_key == "paste_your_actual_api_key_here":
            st.error("Please add a valid Groq API Key to your `.env` file first.")
            return
            
        current_vs = get_current_vector_store()
        if current_vs is None:
            st.error("Please process some documents first.")
            return
            
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            rag_chain = build_rag_chain(current_vs, groq_api_key)
            
            try:
                # Streaming Output Initialization
                placeholder = st.empty()
                full_response = ""
                sources = []
                
                # Stream the response chunk by chunk
                for chunk in rag_chain.stream({"input": prompt}):
                    # Collect the text answer dynamically
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        # '▌' acts as our blinking cursor
                        placeholder.markdown(full_response + "▌")
                    # Capture the retrieval context
                    if "context" in chunk:
                        sources = chunk["context"]
                        
                # Ensure the final markdown is shown cleanly without the cursor
                placeholder.markdown(full_response)
                
                # Display sources if requested (tucked beneath the response)
                if sources:
                    with st.expander("View Source Context"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(doc.page_content)
                            st.divider()
                            
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                database.add_chat_message(st.session_state.session_id, "assistant", full_response)
                
            except Exception as e:
                st.error(f"Error during streaming generation: {e}")

if __name__ == "__main__":
    main()
