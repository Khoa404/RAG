from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()  # tai cac bien moi truong tu file .env

# === 1. Load documents ===
loader = DirectoryLoader(
    path="./papers",                # duong dan den thu muc chua cac file PDF
    glob='**/*.pdf',                # doc cac file PDF (hoac **/*.*)
    show_progress=True,         # hien thi tien do doc file
    loader_cls=UnstructuredFileLoader,      
    use_multithreading=True     # su dung da luong de doc file nhanh hon
)

docs = loader.load()          # doc cac file va tra ve danh sach cac document

MARKDOWN_SEPARATORS = [
    "\n#{1,6}",  # tieu de markdown (tu h1 den h6)
    "```\n",  # chia theo block
    "\n\\*\\*\\**\n",  # doan van ban moi sau cac ky tu dac biet (nhu dau sao, v.v.)
    "\n---+\n",  # doan van ban moi sau cac ky tu dac biet (nhu dau gach ngang, v.v.)
    "\n___+\n",  # doan van ban moi sau cac ky tu dac biet (nhu dau gach ngang, v.v.)
    "\n\n",  # doan van ban moi
    "\n",    # dong moi
    " ",     # khoang trang
    "",
]

# === 2. Split documents ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,    # kich thuoc moi doan van ban sau khi chia
    chunk_overlap=200,  # so ky tu trung lap giua cac doan van ban (de dam bao tinh lien ket giua cac doan)
    add_start_index=True, # them chi so bat dau vao moi doan van ban (de giu nguyen thu tu cua cac doan van ban sau khi chia
    strip_whitespace=True, # loai bo khoang trang o dau va cuoi moi doan van ban sau khi chia    
    separators=MARKDOWN_SEPARATORS,
)

splits = text_splitter.split_documents(docs)  # chia cac document thanh cac doan van ban nho hon

# ===== 3. EMBEDDINGS (vector hóa) =====
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # su dung mo hinh embedding nho hon de tiet kiem chi phi
)  # khoi tao doi tuong embeddings su dung OpenAI

# ===== 4. tạo VECTORSTORE: FAISS =====
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings,
    distance_strategy=DistanceStrategy.COSINE,  # su dung khoang cach cosine de tinh do tuong dong giua cac vector

)  # tao vectorstore tu cac doan van ban da chia va embeddings

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",  # su dung phuong phap tim kiem theo do tuong dong
    search_kwargs={"k": 5, "score_threshold": 0.2} # search_kwargs la cac tham so cho phuong phap tim kiem, o day la tim 5 doan van ban tuong dong nhat va co diem tuong dong lon hon 0.2 (co the dieu chinh de tim duoc nhieu hoac it doan van ban hon)
    
)  # tao retriever tu vectorstore, tra ve 5 doan van ban tuong dong nhat khi truy van

template = (
    "You are a friendly and helpful assistant that answers questions based on the following context:\n" \
    "RULES:\n"
    "1) Use only the provided context to answer the question. Do not use any external information or make assumptions.\n"
    "2) If the context does not contain enough information to answer the question, say you don't know.\n"
    "3) Be concise and to the point in your answers.\n"
    "4) Always provide a source for your answer by including the page number from the context where you found the information.\n"
    "Context:\n{context}\n\n" 
    "Question: {question}" 
)

prompt = ChatPromptTemplate.from_template(template)

# ===== 5. LLM: Dùng mô hình Orca-Mini-3B (GGUF) =====
llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0   # do nhiet thap de co cau tra loi chinh xac hon, khong sang tao nhieu, va tap trung vao viec su dung thong tin tu context de tra loi cau hoi
    
)  # khoi tao doi tuong LLM su dung mo hinh gpt-3.5-turbo voi do nhiet thap de co cau tra loi chinh xac hon

# ===== 6. RAG Chain =====
rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}       # tao input cho prompt, trong do "context" duoc lay tu retriever va "question" duoc truyen truc tiep vao prompt ma khong can xu ly gi
            |  prompt 
            |  llm
            |  StrOutputParser()
    )

question = input("Question: ")  # nhap cau hoi tu nguoi dung

answer = rag_chain.invoke(question)  # thuc thi chuoi cac buoc trong rag_chain voi cau hoi da nhap, va tra ve cau tra loi

print("Answer:", answer)  # in cau tra loi


# ====================================

# cái này dùng API của OpenAI nên phải nạp tiền mới dùng đc 
# hihi ^-^

# ====================================