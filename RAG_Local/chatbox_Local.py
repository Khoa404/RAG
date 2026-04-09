from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ctransformers import AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

print(" Đang tải documents...")
loader = DirectoryLoader(
    path="./papers",
    glob='**/*.*',
    show_progress=True,
    loader_cls=UnstructuredFileLoader,      
    use_multithreading=True
)

docs = loader.load()

MARKDOWN_SEPARATORS = [
    "\n#{1,6}",
    "```\n",
    "\n\\*\\*\\**\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

print(" Đang chia documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)

splits = text_splitter.split_documents(docs)

# ===== EMBEDDINGS: HuggingFace (tối ưu cho CPU) =====
print(" Đang tải embedding model (nhẹ nhất cho CPU)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Mô hình rất nhẹ: ~22MB
    model_kwargs={
        "device": "cpu",
        "trust_remote_code": True,
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32,  # Giảm batch size cho CPU yếu
    },
)

print(" Đang tạo vectorstore...")
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings,
    distance_strategy=DistanceStrategy.COSINE,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",  # Dùng similarity để luôn lấy được k docs tốt nhất
    search_kwargs={"k": 5}  # Lấy 5 docs phù hợp nhất
)

# ===== LLM: Dùng mô hình Orca-Mini-3B (GGUF) =====
print(" Đang tải LLM model Orca-Mini-3B (lần đầu sẽ mất thời gian)...\n")

# Load GGUF model - Orca-Mini-3B instruction-tuned, rất tốt cho RAG
MODEL_PATH = "./models/orca-mini-3b-gguf2-q4_0.gguf"

llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="mistral",
    gpu_layers=0,  # CPU only
    context_length=3072, 
)

# ===== Custom LLM Wrapper =====
class CustomLLM:
    def __init__(self, model_obj):
        self.model = model_obj
    
    def __call__(self, text):
        # Generate answer with Orca model
        response = self.model(
            text,
            max_new_tokens=200,  # Allow longer responses
            temperature=0.3,  # Lower temp = more focused
            top_p=0.9,  # Nucleus sampling
            top_k=40,
            repetition_penalty=1.1,  # Penalize repetition
            stream=False
        )
        return response

llm = CustomLLM(llm_model)

# ===== Prompt Template =====
template = (
    "You are a friendly and helpful assistant that answers questions based on the following context:\n" \
    "RULES:\n"
    "1) Use only the provided context to answer the question. Do not use any external information or make assumptions.\n"
    "2) If the context does not contain enough information to answer the question, say you don't know.\n"
    "3) Be concise and to the point in your answers.\n"
    "4) Always provide a source for your answer by including the page number from the context where you found the information.\n"
   
    "Dựa vào thông tin sau, hãy trả lời câu hỏi một cách ngắn gọn:\n"
    "Context:\n{context}\n\n" 
    "Question: {question}\n"
    "Answer:"
)

prompt = ChatPromptTemplate.from_template(template)

# ===== RAG Chain =====
def generate_answer(input_dict):
    prompt_text = template.format(context=input_dict["context"], question=input_dict["question"])
    answer = llm(prompt_text)
    # Lấy phần trả lời sau "Answer:"
    if "Answer:" in answer:
        return answer.split("Answer:")[-1].strip()
    return answer

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnablePassthrough.assign(context=lambda x: "\n".join([doc.page_content for doc in x["context"]]))
    | RunnablePassthrough.assign(answer=lambda x: generate_answer(x))
)

# ===== Main Loop =====
print(" Hệ thống sẵn sàng! (chạy local trên CPU)")
print("="*60)

while True:
    try:
        question = input("\n Câu hỏi (gõ 'exit' để thoát): ").strip()
        
        if question.lower() == "exit":
            print(" Tạm biệt!")
            break
        
        if not question:
            print("  Vui lòng nhập câu hỏi!")
            continue
        
        print("\n Đang xử lý...")
        result = rag_chain.invoke(question)
        answer = result.get("answer", "Không thể trả lời")
        
        print(f"\n Trả lời: {answer}\n")
        
    except Exception as e:
        print(f" Lỗi: {e}")
        import traceback
        traceback.print_exc()

