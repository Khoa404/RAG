# RAG_Local - Local Retrieval-Augmented Generation Chatbot

A local RAG (Retrieval-Augmented Generation) application that runs completely on your machine using LangChain, HuggingFace embeddings, and Orca-Mini-3B GGUF model. No API keys or internet connection required after initial download!

## ✨ Features

- ✅ **100% Local** - Runs entirely on your machine
- ✅ **Free** - No API costs or subscription fees
- ✅ **Privacy** - All data stays on your computer
- ✅ **CPU-Friendly** - Optimized for CPU-only machines
- ✅ Automatically load and process PDF documents
- ✅ Vectorize and index documents using FAISS and HuggingFace embeddings
- ✅ Intelligent semantic search (Semantic Search)
- ✅ Answer questions based on document context using Orca-Mini-3B model

## 📦 Requirements

- Python 3.8+
- 4GB+ RAM (6GB+ recommended)
- CPU (GPU support optional)
- ~3GB disk space for models

**Note:** First run will download embeddings and LLM models (~2GB total)

## 🚀 Installation

### 1. Clone the project
```bash
git clone <repository-url>
cd RAG_Local
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
pip install langchain langchain-community langchain-huggingface langchain-text-splitters python-dotenv faiss-cpu unstructured ctransformers
```

### 4. Download the GGUF model

The Orca-Mini-3B GGUF model should be placed in `models/` folder:

```bash
# Create models directory if it doesn't exist
mkdir models

# Download Orca-Mini-3B (4-bit quantized, ~2.2GB)
# Download from: https://huggingface.co/TheBloke/orca-mini-3B-gguf2/resolve/main/orca-mini-3b-gguf2-q4_0.gguf
# Or use curl/wget:

# Linux/Mac
wget https://huggingface.co/TheBloke/orca-mini-3B-gguf2/resolve/main/orca-mini-3b-gguf2-q4_0.gguf -O models/orca-mini-3b-gguf2-q4_0.gguf

# Windows (using PowerShell)
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/orca-mini-3B-gguf2/resolve/main/orca-mini-3b-gguf2-q4_0.gguf" -OutFile "models/orca-mini-3b-gguf2-q4_0.gguf"
```

## 📁 Setup

### 1. Add documents

- Create or ensure `papers/` directory exists
- Place your PDF files in the `papers/` folder
- The program will automatically load all PDFs

```bash
# Example structure
papers/
  ├── document1.pdf
  ├── document2.pdf
  └── ...
```

### 2. Run the application

```bash
python chatbox_Local.py
```

**First run notes:**
- Takes 2-3 minutes for initial model loading
- Subsequent runs are faster (models cached)
- First query takes longer due to model warm-up

## 🔄 How It Works

### RAG Process:

```
1. LOAD DOCUMENTS
   └─ Load all PDFs from papers/ directory

2. SPLIT DOCUMENTS
   └─ Split documents into chunks of 1200 characters
      with 200 character overlap between chunks

3. EMBEDDINGS (Vectorization)
   └─ Convert each chunk into vectors using
      HuggingFace sentence-transformers/all-MiniLM-L6-v2
      (~22MB, optimized for CPU)

4. CREATE VECTORSTORE (FAISS)
   └─ Store vectors in FAISS database
      with Cosine similarity metric

5. RETRIEVER
   └─ Find k=5 most similar text chunks
      based on semantic similarity

6. PROMPT TEMPLATE
   └─ Format question + context + rules

7. LLM (Orca-Mini-3B)
   └─ Generate answer based on context using
      local GGUF model (3B parameters)
      temperature=0.3 for focused responses

8. OUTPUT PARSER
   └─ Convert result to text
```

### Diagram:

```
Question Input
    ↓
Retriever (find 5 similar chunks)
    ↓
Context + Question → Prompt Template
    ↓
Local LLM (Orca-Mini-3B)
    ↓
Answer Output
```

## 🔧 Customization

### Change the number of retrieved documents

Modify `"k": 5` to a different number in `chatbox_Local.py`:

```python
search_kwargs={"k": 10}  # Retrieve 10 chunks instead of 5
```

### Change embedding model

Replace the model name to use a different HuggingFace model:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Change this
    ...
)
```

**Recommended lightweight models:**
- `sentence-transformers/all-MiniLM-L6-v2` (22MB - fastest)
- `sentence-transformers/paraphrase-MiniLM-L6-v2` (33MB)
- `sentence-transformers/all-mpnet-base-v2` (420MB - most accurate)

### Change chunk size

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,    # Increase for larger chunks
    chunk_overlap=300,  # Adjust overlap
    ...
)
```

### Change LLM parameters

In the `CustomLLM` class:

```python
def __call__(self, text):
    response = self.model(
        text,
        max_new_tokens=200,      # Max answer length
        temperature=0.3,         # Lower = more focused, 0-1 range
        top_p=0.9,              # Nucleus sampling
        top_k=40,               # Top-k sampling
        repetition_penalty=1.1,  # Penalize repeated text
        stream=False
    )
    return response
```

## 📚 Models Used

### Embeddings
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Size:** 22MB
- **Dimension:** 384
- **Speed:** Very fast
- **Accuracy:** Good for semantic search

### LLM
- **Model:** Orca-Mini-3B (Quantized)
- **Format:** GGUF (4-bit quantization)
- **Size:** ~2.2GB
- **Parameters:** 3 Billion
- **Speed:** ~5-15 seconds per response (CPU)
- **Quality:** Good for Q&A and summarization

**Download link:** [Orca-Mini-3B-GGUF2](https://huggingface.co/TheBloke/orca-mini-3B-gguf2)

### Example

Question: What is MINA?

Answer: MINA is a hardware architecture designed for efficient convolutional neural network (CNN) inference on resource-constrained platforms, such as FPGAs. It consists of a PEA, which performs matrix multiplication and activation functions, and a SBA, which routes data to the appropriate PEs based on their output. MINA is optimized for speed, area, and energy efficiency, making it suitable for applications where resource constraints are a concern.

## 📄 License

Free to use for educational and research purposes.


**Happy Question Answering! 🎉**
