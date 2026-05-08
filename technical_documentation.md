Technical Stack & Database Schemas: HalluRAG

1. Technical Stack Overview

HalluRAG is an AI-powered Document Q&A pipeline, also known as a Retrieval-Augmented Generation (RAG) system. The application is built using Python, and it seamlessly integrates several cutting-edge AI tools to process documents, store vectors, and generate ChatGPT-like conversational responses based on the provided context.

Frontend and App Logic
- **Framework**: [Streamlit](https://streamlit.io/)
  Provides the web application interface, enabling quick UI prototyping, file uploads, chat interactions, and session state management.
- **Language**: Python 3.x

AI & Embeddings Framework
- **Orchestrator**: [LangChain](https://python.langchain.com/) (Versions: `0.2.16`)
  Manages the flow of information between document loaders, text splitters, vector stores, and LLMs. It handles the specific chains like `create_stuff_documents_chain` and `create_retrieval_chain`.
- **Embeddings Model**: `HuggingFaceEmbeddings` 
  Utilizes the `sentence-transformers/all-mpnet-base-v2` model from Hugging Face to convert document chunks into 768-dimensional dense vectors.
- **LLM Provider**: [Groq](https://groq.com/) API via `ChatGroq`
  Powers the conversational interface. The system specifically uses the high-speed Llama 3 model variant (`llama-3.1-8b-instant`).

Databases Storage
- **Vector Database**: [FAISS](https://faiss.ai/) (Facebook AI Similarity Search) (CPU Version)
  Used for fast and highly efficient similarity search of embeddings locally. The index is built from the uploaded and approved documents.
- **Relational Database**: [SQLite3](https://docs.python.org/3/library/sqlite3.html)
  A lightweight, serverless database used to store persistent application metadata such as users, document access logs, and chat histories.

Document Processing Libraries
- **PDFs**: `pypdf` (via `PyPDFLoader`)
- **Word Documents (.docx)**: `python-docx` / `docx2txt` (via `Docx2txtLoader`)
- **Text Files (.txt)**: Standard Python Text Loading (via `TextLoader`)

---

2. Database Schema (SQLite3)

The application utilizes local SQLite (`data/hallurag.db`) for lightweight state and metadata persistence. The database consists of three main tables:

A. `users` Table
Stores user credentials and roles to grant access to the administrative areas of the application. Passwords are comprehensively hashed using the SHA-256 algorithm.

| Column Name | Data Type | Constraints & Defaults | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Unique identifier for each user. |
| `username` | `TEXT` | `UNIQUE NOT NULL` | Login identifier for the user. |
| `password_hash` | `TEXT` | `NOT NULL` | SHA-256 hash of the user's password. |
| `role` | `TEXT` | `NOT NULL DEFAULT 'user'` | Role-based access control (e.g., 'admin', 'user'). |

B. `documents` Table
Acts as a registry of all approved documents currently active for indexing in the FAISS vector database. Used to track when specific files were added.

| Column Name | Data Type | Constraints & Defaults | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Unique identifier for the document entry. |
| `filename` | `TEXT` | `UNIQUE NOT NULL` | Name of the uploaded file natively saved in the workspace. |
| `upload_date` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP` | The exact timestamp when the file was processed. |
| `status` | `TEXT` | `NOT NULL DEFAULT 'active'` | File status indicator (e.g., 'active' / 'deleted'). |

C. `chat_history` Table
Stores conversational logs mapped to specific session IDs so users can maintain a history payload alongside the bot interface logic across reruns. 

| Column Name | Data Type | Constraints & Defaults | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Unique identifier for the specific message. |
| `session_id` | `TEXT` | `NOT NULL` | Session UUID tracking the conversation thread. |
| `role` | `TEXT` | `NOT NULL` | Sender role: 'user' or 'assistant'. |
| `content` | `TEXT` | `NOT NULL` | The message body or conversational text. |
| `timestamp` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP` | Exact time the message was recorded. |

---

3. Workflow & Processing Characteristics

- **Document Ingestion**: Handled under `target_dir` logic (usually `./data/approved`). Approved documents exist in this local directory.
- **Chunking Profile**: Documents are recursively chunked into segments of **300 words**, carrying an overlap context of **50 words** using LangChain's `RecursiveCharacterTextSplitter`.
- **Top Retrieval**: The RAG queries search FAISS by extracting the top **5** most semantically identical documents (Chunks) relative to the prompt (`k=5`).
- **Security Check**: Admin user checks database records upon running. If missing, it will securely insert an admin user using credentials mapped into `.env`.
