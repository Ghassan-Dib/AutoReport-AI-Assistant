from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import ANTHROPIC_MODEL, EMBEDDINGS_MODEL_NAME, VECTOR_STORE_PATH

load_dotenv()

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create chat history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    # Load the vector store
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    # Create a retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    # Initialize the LLM
    llm = ChatAnthropic(model=ANTHROPIC_MODEL, temperature=0, max_tokens=4096)

    # Contextualize question prompt - helps reformulate follow-up questions
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA system prompt - your financial analyst instructions
    qa_system_prompt = """You are an AI financial analyst. You must answer only using the context provided below.
    If the answer is not explicitly stated, reply: "The data is not explicitly stated in the reports."

    Context:
    {context}

    Give a direct, factual answer based on the numbers and statements in the context."""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create the full RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Wrap with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


def clear_session_history(session_id: str):
    """Clear chat history for a specific session."""
    if session_id in store:
        del store[session_id]
