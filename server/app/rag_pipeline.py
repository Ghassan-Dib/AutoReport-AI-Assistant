import dotenv

from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import EMBEDDINGS_MODEL_NAME, ANTHROPIC_MODEL, VECTOR_STORE_PATH

dotenv.load_dotenv()


def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    # Load the vector store with the embedding function
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    # Create a retriever object
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    # Initialize the LLM
    llm = ChatAnthropic(model=ANTHROPIC_MODEL, temperature=0)

    # Create a memory object to store conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    prompt_template = """
        You are an AI financial analyst. You must answer only using the context provided below.
        If the answer is not explicitly stated, reply: "The data is not explicitly stated in the reports."

        Context:
        {context}

        Question:s
        {question}

        Give a direct, factual answer based on the numbers and statements in the context.
    """

    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=prompt_template,
    )

    # Create Conversational RAG Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False,
        # return_source_documents=True,
        # memory_key="answer",
    )

    return chain
