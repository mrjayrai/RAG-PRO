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

USERS_DIR = os.path.join("data", "users")
ADMIN_DIR = os.path.join("data", "admin")
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(ADMIN_DIR, exist_ok=True)

# Persistent vector store paths
VECTORSTORE_DIR = "vectorstores"
ADMIN_VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, "admin_index")
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

database.init_db()


def get_user_paths(user_id: int):
    document_dir = os.path.join(USERS_DIR, str(user_id), "documents")
    vectorstore_dir = os.path.join(USERS_DIR, str(user_id), "vectorstore")
    os.makedirs(document_dir, exist_ok=True)
    os.makedirs(vectorstore_dir, exist_ok=True)
    return document_dir, vectorstore_dir


def get_document_path(document: dict):
    if document.get("visibility") == "admin":
        return os.path.join(ADMIN_DIR, document["filename"])
    return os.path.join(USERS_DIR, str(document.get("owner_id")), "documents", document["filename"])


def get_all_user_document_dirs():
    all_dirs = []
    if not os.path.exists(USERS_DIR):
        return all_dirs
    for user_folder in os.listdir(USERS_DIR):
        candidate = os.path.join(USERS_DIR, user_folder, "documents")
        if os.path.isdir(candidate):
            all_dirs.append(candidate)
    return all_dirs

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
def save_uploaded_files(uploaded_files, target_dir, owner_id, visibility='public'):
    """Saves Streamlit uploaded files to the specified directory and database."""
    saved_count = 0
    for uploaded_file in uploaded_files:
        file_path = os.path.join(target_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        database.add_document(uploaded_file.name, owner_id, visibility=visibility)
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
def get_vector_store(chunks: List[Document], store_path: str):
    """
    Loads or creates FAISS vector store safely.
    """
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

    # Load existing vector store if present
    if os.path.exists(store_path):
        try:
            return FAISS.load_local(
                store_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"Failed loading vector store: {e}")

    # Create new vector store
    if not chunks:
        return None

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_store.save_local(store_path)

    return vector_store

# ==========================================
# 4. Retrieval & Generation
# ==========================================
def build_rag_chain(vector_store, groq_api_key: str, extra_context: str = None):
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
    )

    if extra_context:
        system_prompt += (
            "Use the following relevant admin memory to inform your response when it is applicable:\n"
            f"{extra_context}\n\n"
        )

    system_prompt += "Context:\n{context}"

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


def build_admin_memory_context():
    messages = database.get_admin_messages()
    if not messages:
        return None
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])

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
    
    # Premium Modern Dark Theme CSS
    st.markdown("""
        <style>
        /* ==========================================
           BASE DESIGN & LAYOUT
           ========================================== */
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-tertiary: #94a3b8;
            --border-color: #1e293b;
            --border-subtle: #334155;
        }
        
        html, body, [data-testid="stAppViewContainer"] {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }
        
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1.5rem !important;
            max-width: 900px !important;
            margin: 0 auto !important;
        }
        
        /* ==========================================
           TYPOGRAPHY
           ========================================== */
        h1 {
            color: var(--text-primary) !important;
            font-size: 1.875rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px !important;
            margin-bottom: 0.5rem !important;
        }
        
        h2 {
            color: var(--text-primary) !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.25px !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
        }
        
        h3 {
            color: var(--text-secondary) !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            letter-spacing: 0px !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        p, span {
            color: var(--text-secondary) !important;
            font-size: 0.95rem !important;
            line-height: 1.6 !important;
        }
        
        .stCaption {
            color: var(--text-tertiary) !important;
            font-size: 0.85rem !important;
            margin-top: 0.25rem !important;
        }
        
        /* ==========================================
           SIDEBAR STYLING
           ========================================== */
        [data-testid="stSidebar"] {
            background-color: var(--bg-secondary) !important;
            border-right: 1px solid var(--border-color) !important;
        }
        
        [data-testid="stSidebarContent"] {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        
        /* Sidebar section spacing */
        [data-testid="stSidebarContent"] > div > div {
            margin-bottom: 0.5rem !important;
        }
        
        /* ==========================================
           BUTTONS
           ========================================== */
        .stButton > button {
            border-radius: 6px !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
            border: none !important;
            transition: all 0.2s ease !important;
            letter-spacing: 0.3px !important;
        }
        
        .stButton > button[kind="primary"] {
            background-color: var(--primary) !important;
            color: white !important;
            padding: 0.5rem 1rem !important;
            box-shadow: 0 2px 4px rgba(99, 102, 241, 0.2) !important;
        }
        
        .stButton > button[kind="primary"]:hover {
            background-color: var(--primary-hover) !important;
            box-shadow: 0 4px 8px rgba(99, 102, 241, 0.3) !important;
            transform: translateY(-1px) !important;
        }
        
        .stButton > button[kind="secondary"] {
            background-color: transparent !important;
            color: var(--text-secondary) !important;
            border: 1px solid var(--border-subtle) !important;
            padding: 0.5rem 0.75rem !important;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-subtle) !important;
        }
        
        .stButton > button[kind="secondary"] p {
            text-align: left !important;
            width: 100% !important;
            margin: 0 !important;
        }
        
        /* ==========================================
           INPUTS & TEXT AREAS
           ========================================== */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 6px !important;
            padding: 0.5rem 0.75rem !important;
            font-size: 0.95rem !important;
            transition: all 0.2s ease !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus {
            border-color: var(--primary) !important;
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
        }
        
        .stChatInputContainer {
            padding: 1rem 0 !important;
            border-top: 1px solid var(--border-color) !important;
        }
        
        /* ==========================================
           CHAT MESSAGES
           ========================================== */
        .stChatMessage {
            background-color: transparent !important;
            border: none !important;
            padding: 1rem 0 !important;
            margin-bottom: 1rem !important;
        }
        
        .stChatMessage [data-testid="stChatMessageContent"] {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            margin-left: 0.5rem !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
        }
        
        .stChatMessage[data-testid="stChatMessage-user"] [data-testid="stChatMessageContent"] {
            background-color: var(--primary) !important;
            border-color: var(--primary) !important;
            color: white !important;
        }
        
        .stChatMessage[data-testid="stChatMessage-user"] [data-testid="stChatMessageContent"] p,
        .stChatMessage[data-testid="stChatMessage-user"] [data-testid="stChatMessageContent"] span {
            color: white !important;
        }
        
        .stChatMessageAvatar {
            width: 32px !important;
            height: 32px !important;
            border-radius: 6px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 1.2rem !important;
            background-color: var(--bg-tertiary) !important;
        }
        
        .stChatMessage[data-testid="stChatMessage-user"] .stChatMessageAvatar {
            background-color: var(--primary) !important;
        }
        
        /* Chat message content markdown */
        .stChatMessage [data-testid="stChatMessageContent"] a {
            color: var(--primary) !important;
            text-decoration: none !important;
            border-bottom: 1px solid var(--primary) !important;
        }
        
        .stChatMessage [data-testid="stChatMessageContent"] a:hover {
            opacity: 0.8 !important;
        }
        
        .stChatMessage [data-testid="stChatMessageContent"] code {
            background-color: var(--bg-primary) !important;
            color: #e879f9 !important;
            padding: 0.2rem 0.4rem !important;
            border-radius: 4px !important;
            font-size: 0.9rem !important;
        }
        
        .stChatMessage [data-testid="stChatMessageContent"] pre {
            background-color: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px !important;
            padding: 1rem !important;
            overflow-x: auto !important;
        }
        
        /* ==========================================
           EXPANDERS & CONTAINERS
           ========================================== */
        [data-testid="stExpander"] {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 6px !important;
        }
        
        [data-testid="stExpanderDetails"] {
            background-color: var(--bg-primary) !important;
            border-top: 1px solid var(--border-color) !important;
        }
        
        /* ==========================================
           FILE UPLOADER
           ========================================== */
        [data-testid="stFileUploadDropzone"] {
            background-color: var(--bg-primary) !important;
            border: 2px dashed var(--border-subtle) !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }
        
        [data-testid="stFileUploadDropzone"]:hover {
            border-color: var(--primary) !important;
            background-color: rgba(99, 102, 241, 0.05) !important;
        }
        
        /* ==========================================
           ALERTS & NOTIFICATIONS
           ========================================== */
        .stAlert {
            border-radius: 6px !important;
            padding: 1rem !important;
            border: 1px solid !important;
        }
        
        .stSuccess {
            background-color: rgba(34, 197, 94, 0.1) !important;
            border-color: rgba(34, 197, 94, 0.3) !important;
        }
        
        .stInfo {
            background-color: rgba(59, 130, 246, 0.1) !important;
            border-color: rgba(59, 130, 246, 0.3) !important;
        }
        
        .stWarning {
            background-color: rgba(245, 158, 11, 0.1) !important;
            border-color: rgba(245, 158, 11, 0.3) !important;
        }
        
        .stError {
            background-color: rgba(239, 68, 68, 0.1) !important;
            border-color: rgba(239, 68, 68, 0.3) !important;
        }
        
        /* ==========================================
           DIVIDERS
           ========================================== */
        hr {
            border: none !important;
            border-top: 1px solid var(--border-color) !important;
            margin: 1rem 0 !important;
        }
        
        /* ==========================================
           RESPONSIVE DESIGN
           ========================================== */
        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            
            h1 {
                font-size: 1.5rem !important;
            }
            
            h2 {
                font-size: 1.1rem !important;
            }
            
            [data-testid="stSidebar"] {
                width: 100% !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("HalluRAG: Document Q&A Pipeline")
    
    # Initialize session state for auth, chat history, and vector stores
    if "user" not in st.session_state:
        st.session_state.user = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "role" not in st.session_state:
        st.session_state.role = "user"
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_vector_store" not in st.session_state:
        st.session_state.user_vector_store = None
    if "admin_vector_store" not in st.session_state:
        st.session_state.admin_vector_store = None
        
    groq_api_key = os.environ.get("GROQ_API_KEY")
    admin_pass = os.environ.get("ADMIN_PASSWORD", "admin123")
    
    # Ensure admin user is seeded in the DB
    database.seed_admin_if_needed(admin_pass)

    if not st.session_state.is_authenticated:
        login_tab, register_tab = st.tabs(["Login", "Register"])
        with login_tab:
            st.markdown("### Login")
            login_username = st.text_input("Username or Email", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", use_container_width=True, type="primary"):
                if not login_username or not login_password:
                    st.error("Please enter both username and password.")
                else:
                    user = database.authenticate_user(login_username, login_password)
                    if user:
                        st.session_state.user = user["username"]
                        st.session_state.user_id = user["id"]
                        st.session_state.role = user["role"]
                        st.session_state.is_authenticated = True
                        st.session_state.is_admin = user["role"] == "admin"
                        st.session_state.messages = []
                        st.session_state.session_id = str(uuid.uuid4())
                        database.create_chat_session(st.session_state.session_id, st.session_state.user_id)
                        st.success("Logged in successfully.")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
        with register_tab:
            st.markdown("### Register")
            register_username = st.text_input("Username or Email", key="register_username")
            register_password = st.text_input("Password", type="password", key="register_password")
            if st.button("Create account", use_container_width=True, type="secondary"):
                if not register_username or not register_password:
                    st.error("Please enter both username and password.")
                else:
                    created = database.create_user(register_username, register_password, role="user")
                    if created:
                        st.success("Account created successfully. Please log in.")
                    else:
                        st.error("Username already exists. Please choose another.")
        return

    if st.session_state.session_id is None:
        st.session_state.session_id = str(uuid.uuid4())
        database.create_chat_session(st.session_state.session_id, st.session_state.user_id)

    if not st.session_state.messages:
        history = database.get_chat_history(
            st.session_state.session_id,
            user_id=st.session_state.user_id,
            is_admin=st.session_state.is_admin
        )
        if history:
            st.session_state.messages = history

    # Auto-initialize vector stores on startup
    user_doc_dir, user_vector_dir = get_user_paths(st.session_state.user_id)

    def rebuild_vector_stores():
        if st.session_state.is_admin:
            if os.path.exists(ADMIN_VECTORSTORE_PATH):
                shutil.rmtree(ADMIN_VECTORSTORE_PATH, ignore_errors=True)

            all_docs = []
            for doc_dir in get_all_user_document_dirs():
                all_docs.extend(load_directory_documents(doc_dir))
            all_docs.extend(load_directory_documents(ADMIN_DIR))

            if all_docs:
                st.session_state.admin_vector_store = get_vector_store(
                    chunk_documents(all_docs),
                    ADMIN_VECTORSTORE_PATH
                )
            else:
                st.session_state.admin_vector_store = None
        else:
            if os.path.exists(user_vector_dir):
                shutil.rmtree(user_vector_dir, ignore_errors=True)

            docs = load_directory_documents(user_doc_dir)
            if docs:
                st.session_state.user_vector_store = get_vector_store(
                    chunk_documents(docs),
                    user_vector_dir
                )
            else:
                st.session_state.user_vector_store = None

    if st.session_state.user_vector_store is None and not st.session_state.is_admin:
        docs = load_directory_documents(user_doc_dir)
        if docs:
            st.session_state.user_vector_store = get_vector_store(
                chunk_documents(docs),
                user_vector_dir
            )

    if st.session_state.admin_vector_store is None and st.session_state.is_admin:
        all_docs = []
        for doc_dir in get_all_user_document_dirs():
            all_docs.extend(load_directory_documents(doc_dir))
        all_docs.extend(load_directory_documents(ADMIN_DIR))
        if all_docs:
            st.session_state.admin_vector_store = get_vector_store(
                chunk_documents(all_docs),
                ADMIN_VECTORSTORE_PATH
            )

    def get_current_vector_store():
        return st.session_state.admin_vector_store if st.session_state.is_admin else st.session_state.user_vector_store

    # Sidebar
    with st.sidebar:
        st.markdown(f"### {st.session_state.user} — {st.session_state.role.capitalize()}")
        if st.button("Logout", use_container_width=True, type="secondary"):
            st.session_state.is_authenticated = False
            st.session_state.user = None
            st.session_state.user_id = None
            st.session_state.role = "user"
            st.session_state.is_admin = False
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.user_vector_store = None
            st.session_state.admin_vector_store = None
            st.rerun()

        st.divider()
        # --- New Chat ---
        st.markdown("### 💬 Chat")
        if st.button("New chat", use_container_width=True, type="primary"):
            st.session_state.session_id = str(uuid.uuid4())
            database.create_chat_session(st.session_state.session_id, st.session_state.user_id)
            st.session_state.messages = []
            st.rerun()
            
        # --- Search Chats ---
        search_term = st.text_input("Search", placeholder="Search chats...", label_visibility="collapsed")
        
        # --- History / Recents ---
        st.markdown("### Recents")
        sessions = database.get_all_sessions(
            user_id=st.session_state.user_id,
            is_admin=st.session_state.is_admin
        )
        
        if search_term:
            sessions = [s for s in sessions if s['title'] and search_term.lower() in s['title'].lower()]
            
        if sessions:
            for s in sessions:
                title = s['title'] if s['title'] else "Empty Chat"
                if len(title) > 28:
                    title = title[:25] + "..."
                
                # Highlight active session
                btn_type = "primary" if s['session_id'] == st.session_state.session_id else "secondary"
                
                col1, col2 = st.columns([5, 1])
                with col1:
                    if st.button(title, key=f"hist_{s['session_id']}", use_container_width=True, type=btn_type):
                        st.session_state.session_id = s['session_id']
                        st.session_state.messages = database.get_chat_history(
                            s['session_id'],
                            user_id=st.session_state.user_id,
                            is_admin=st.session_state.is_admin
                        )
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{s['session_id']}", use_container_width=True):
                        database.delete_chat_session(
                            s['session_id'],
                            user_id=st.session_state.user_id,
                            is_admin=st.session_state.is_admin
                        )
                        if s['session_id'] == st.session_state.session_id:
                            st.session_state.session_id = str(uuid.uuid4())
                            st.session_state.messages = []
                        st.rerun()
        else:
            st.caption("No recent chats yet")
            
        st.divider()

        st.markdown("### ⚙️ Configuration")
        
        if not groq_api_key or groq_api_key == "paste_your_actual_api_key_here":
            st.warning("⚠️ Add your Groq API Key to `.env`", icon="⚠️")
            
        st.divider()

        st.markdown("### 📚 Available Data")
        st.caption("Documents used by the assistant")
        current_documents = database.get_all_documents(
            is_admin=st.session_state.is_admin,
            user_id=st.session_state.user_id
        )
        if current_documents:
            for document in current_documents:
                display_name = document["filename"]
                if st.session_state.is_admin and document["visibility"] == "admin":
                    display_name = f"[Admin] {display_name}"
                elif st.session_state.is_admin:
                    display_name = f"[User {document['owner_id']}] {display_name}"
                st.caption(f"📄 {display_name}")
        else:
            st.caption("_No data available_")

        st.divider()

        st.markdown("### 📁 Upload Documents")
        st.caption("Make documents available to Q&A")
        uploaded_files = st.file_uploader(
            "Upload files", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if st.button("Upload and Process", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one document")
            else:
                count = save_uploaded_files(
                    uploaded_files,
                    get_user_paths(st.session_state.user_id)[0],
                    owner_id=st.session_state.user_id,
                    visibility='private'
                )
                st.success(f"Processed {count} document(s)")
                rebuild_vector_stores()
                st.rerun()

        st.divider()

        if st.session_state.is_admin:
            st.markdown("### 🛡️ Admin Tools")
            st.markdown(f"**Signed in as:** {st.session_state.user}")
            st.caption("Admin can preview, download, and delete any document.")

            all_documents = database.get_all_documents(is_admin=True)
            if all_documents:
                for document in all_documents:
                    file_path = get_document_path(document)
                    label = document["filename"]
                    if document["visibility"] == "admin":
                        label = f"[Admin] {label}"
                    else:
                        label = f"[User {document['owner_id']}] {label}"

                    with st.expander(f"📄 {label}", expanded=False):
                        try:
                            file_extension = os.path.splitext(file_path)[1].lower()
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
                            st.text_area("Preview", content[:2000] + ("..." if len(content) > 2000 else ""), height=120, disabled=True)
                        except Exception as e:
                            st.error(f"Error loading file: {e}")
                        
                        # with open(file_path, "rb") as f:
                        #     st.download_button(
                        #         label="Download file",
                        #         data=f,
                        #         file_name=document["filename"],
                        #         use_container_width=True
                        #     )
                        if os.path.exists(file_path):

                            with open(file_path, "rb") as f:
                                st.download_button(
                                    label="Download file",
                                    data=f,
                                    file_name=document["filename"],
                                    use_container_width=True
                                )

                        else:
                            st.warning("File missing from disk.")

                            # Optional cleanup
                            try:
                                database.remove_document(
                                    document["filename"],
                                    owner_id=document.get("owner_id")
                                )
                            except:
                                pass
                        if st.button("Delete", key=f"del_{document['id']}", use_container_width=True):
                            try:
                                os.remove(file_path)
                            except FileNotFoundError:
                                pass
                            database.remove_document(document["filename"], owner_id=document["owner_id"])
                            rebuild_vector_stores()
                            st.rerun()

            st.divider()
            st.markdown("#### Upload Admin Files")
            admin_files = st.file_uploader("Files", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="admin_uploader", label_visibility="collapsed")
            if st.button("Process Admin Files", use_container_width=True):
                if admin_files:
                    count = save_uploaded_files(admin_files, ADMIN_DIR, owner_id=st.session_state.user_id, visibility='admin')
                    rebuild_vector_stores()
                    st.success(f"Processed {count} admin document(s)")
                else:
                    st.warning("No files to upload")

            st.divider()
            if st.button("🔄 Rebuild Index", use_container_width=True):
                rebuild_vector_stores()

            st.divider()
            if st.button("📊 Run Evaluation", use_container_width=True):
                evaluate_pipeline_ragas()
        else:
            st.markdown("### 🛡️ User Tools")
            st.caption("Upload documents and ask questions using your own document store.")
            user_documents = database.get_all_documents(is_admin=False, user_id=st.session_state.user_id)
            if user_documents:
                for document in user_documents:
                    file_path = get_document_path(document)
                    with st.expander(f"📄 {document['filename']}", expanded=False):
                        try:
                            if not os.path.exists(file_path):
                                st.error("File not found on disk.")
                                continue
                            file_extension = os.path.splitext(file_path)[1].lower()
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
                            st.text_area("Preview", content[:2000] + ("..." if len(content) > 2000 else ""), height=120, disabled=True)
                        except Exception as e:
                            st.error(f"Error loading file: {e}")
                        
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label="Download file",
                                data=f,
                                file_name=document["filename"],
                                use_container_width=True
                            )
                        if st.button("Delete", key=f"del_user_{document['id']}", use_container_width=True):
                            try:
                                os.remove(file_path)
                            except FileNotFoundError:
                                pass
                            database.remove_document(document["filename"], owner_id=st.session_state.user_id)
                            rebuild_vector_stores()
                            st.rerun()
            else:
                st.caption("_No documents uploaded yet._")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Message HalluRAG..."):
        # Display user message in chat message container
        st.chat_message("user", avatar="user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        database.add_chat_message(
            st.session_state.session_id,
            "user",
            prompt,
            st.session_state.user_id
        )
        
        # Check prerequisites
        if not groq_api_key or groq_api_key == "paste_your_actual_api_key_here":
            st.error("Please add a valid Groq API Key to your `.env` file first.")
            return
            
        current_vs = get_current_vector_store()
        if current_vs is None:
            st.error("Please process some documents first.")
            return
            
        extra_context = None
        if st.session_state.is_admin:
            extra_context = build_admin_memory_context()

        # Generate response
        with st.chat_message("assistant", avatar="assistant"):
            rag_chain = build_rag_chain(current_vs, groq_api_key, extra_context=extra_context)
            
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
                    with st.expander("📖 View Source Context"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.markdown(doc.page_content)
                            if i < len(sources) - 1:
                                st.divider()
                            
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                database.add_chat_message(
                    st.session_state.session_id,
                    "assistant",
                    full_response,
                    st.session_state.user_id
                )
                
            except Exception as e:
                st.error(f"Error during streaming generation: {e}")

if __name__ == "__main__":
    main()
