from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import ANTHROPIC_MODEL, EMBEDDINGS_MODEL_NAME, VECTOR_STORE_PATH
from app.prompts import (
    CONTEXTUALIZE_Q_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    SYNTHESIZE_ANSWER_SYSTEM_PROMPT,
)
from app.utils.retrieval import (
    create_multi_query_retriever,
    create_query_decomposition_retriever,
)

load_dotenv()

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create chat history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_rag_chain(retriever_type: str | None = None) -> RunnableWithMessageHistory:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    # Load the vector store
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    # Initialize the LLM
    llm = ChatAnthropic(model=ANTHROPIC_MODEL, temperature=0, max_tokens=4096)

    # Create the base retriever
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    # Choose retriever type
    if retriever_type == "multi_query":
        retriever = create_multi_query_retriever(llm, base_retriever)
    elif retriever_type == "query_decomposition":
        # Decomposition handles retrieval and initial QA internally
        decomposition_retriever = create_query_decomposition_retriever(
            llm, base_retriever
        )

        # Synthesis chain
        synthesis_prompt = ChatPromptTemplate.from_template(
            SYNTHESIZE_ANSWER_SYSTEM_PROMPT
        )

        synthesis_chain = synthesis_prompt | llm | StrOutputParser()

        def format_decomposition_output(answer: str) -> dict:
            """Wrap string answer in dict format for consistency"""
            return {"answer": answer}

        # Combine decomposition and synthesis
        decomposition_rag_chain = (
            decomposition_retriever
            | synthesis_chain
            | RunnableLambda(format_decomposition_output)
        )

        # Wrap with message history
        conversational_rag_chain = RunnableWithMessageHistory(
            decomposition_rag_chain,
            get_session_history,
            input_messages_key="input",
            output_messages_key="answer",
        )

        return conversational_rag_chain
    else:
        # Use standard retrieval
        retriever = base_retriever

    # Contextualize question prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA system prompt
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
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
