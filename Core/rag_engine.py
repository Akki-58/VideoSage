from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from Core.vector_store import build_vector_store, load_vector_store, get_retriever
from chromadb import PersistentClient

from dotenv import load_dotenv
load_dotenv()

CHROMA_DIR = "vector_db"

def load_llm():
    return ChatMistralAI(
        model = "ministral-8b-latest",
        temperature=0
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system",
"""You are an expert meeting assistant. Answer the user's question based ONLY on the meeting transcript context provided below.

You must not use outside knowledge.

If the answer is partially available,
state what is available and clearly indicate
what information is missing.

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}")
        ]
    )

def collection_exists(collection_name: str) -> bool:
    client = PersistentClient(path=CHROMA_DIR)

    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False

def get_rag_chain(transcript: str, video_id: str):
    if collection_exists(collection_name= video_id):
        print(f"Loading existing vector store: {video_id}")
        vector_store = load_vector_store(collection_name= video_id)

    else:
        if transcript is None:
            raise ValueError(
                f"Collection '{video_name}' does not exist and no transcript was provided."
            )

        print(f"Building new vector store: {video_id}")
        vector_store = build_vector_store(transcript, video_id)

    retriever = get_retriever(vector_store)

    llm = load_llm()
    prompt = get_prompt()

    # RAG pipeline
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs) ,
        "question": RunnablePassthrough()
        } 
        | prompt | llm | StrOutputParser()
    )

    return rag_chain

def ask_question(rag_chain, question:str) -> str:
    # print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    # print(f"answer :{answer}")
    return answer