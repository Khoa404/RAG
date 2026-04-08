# RAG_API - Retrieval-Augmented Generation Chatbot

A RAG (Retrieval-Augmented Generation) application using LangChain and OpenAI API to build an intelligent chatbot capable of answering questions based on PDF documents.

## ✨ Features

- ✅ Automatically load and process PDF documents
- ✅ Vectorize and index documents using FAISS
- ✅ Intelligent semantic search (Semantic Search)
- ✅ Answer questions based on document context
- ✅ Provide source references (source) for each answer
- ✅ Use OpenAI's GPT-3.5-turbo model

## 📦 Requirements

- OpenAI API key (must have an OpenAI account with credits)
- Python libraries 

## 🚀 Installation

### 1. Clone the project
```bash
git clone <repository-url>
cd RAG_API
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. install manually
```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters python-dotenv faiss-cpu unstructured
```

## ⚙️ Configuration

### 1. Get OpenAI API Key
- Visit [OpenAI API](https://platform.openai.com)
- Sign in with your account
- Go to "API keys" section and create a new key

### 2. Create `.env` file

Create a `.env` file in the project directory and add:

```env
OPENAI_API_KEY=your_api_key_here
```

### 3. Add documents

- Put PDF files in the `papers/` directory
- The program will automatically load all PDFs in this directory


## 🔄 How It Works

### RAG Process:

```
1. LOAD DOCUMENTS
   └─ Load all PDFs from papers/ directory

2. SPLIT DOCUMENTS
   └─ Split documents into chunks of 1200 characters
      with 200 character overlap between chunks

3. EMBEDDINGS (Vectorization)
   └─ Convert each chunk into numerical vectors
      using OpenAI text-embedding-3-small

4. CREATE VECTORSTORE (FAISS)
   └─ Store vectors in FAISS database
      with Cosine similarity metric

5. RETRIEVER
   └─ Find k=5 most similar text chunks
      with score threshold >= 0.2

6. PROMPT TEMPLATE
   └─ Format question + context + rules

7. LLM (GPT-3.5-turbo)
   └─ Generate answer based on context
      (temperature=0 for accuracy, no creativity)

8. OUTPUT PARSER
   └─ Convert result to text
```

## 🔧 Customization

### Change the number of retrieved documents

Modify `"k": 5` to a different number:

```python
search_kwargs={"k": 10, "score_threshold": 0.2}  # Retrieve 10 chunks
```

### Change similarity score threshold

Modify `"score_threshold": 0.2` (from 0 to 1):

```python
search_kwargs={"k": 5, "score_threshold": 0.5}  # Only take strong matches
```

### Change chunk size

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,    # Change size
    chunk_overlap=300,
    ...
)
```

## 📄 License

Free to use for educational and research purposes.
