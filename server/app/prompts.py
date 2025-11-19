MULTI_QUERY_TEMPLATE = """You are an AI language model assistant tasked with generating informative queries for a vector search engine.
The user has a question: "{question}"
Your goal is to create three variations of this question that capture different aspects of the user's intent. These variations will help the search engine retrieve relevant documents even if they don't use the exact keywords as the original question.
Provide these alternative questions, each on a new line.
Original question: {question}"""

QUERY_DECOMPOSITION_TEMPLATE = """You are an AI language model assistant that generates multiple sub-questions related to an input question. \n
Your task is to break down the input into three sub-problems / sub-questions that can be answered in isolation. \n
Generate multiple search queries related to: {question} \n
Original question: {question}"""

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

QA_SYSTEM_PROMPT = """You are an AI financial analyst. You must answer only using the context provided below.
If the answer is not explicitly stated, reply: "The data is not explicitly stated in the reports."

Context:
{context}

Give a direct, factual answer based on the numbers and statements in the context."""

SUB_QUESTION_QA_TEMPLATE = """Answer the following question based on this context:
Context:
{context}

Question:
{question}

Provide a concise, factual answer based only on the context provided."""

SYNTHESIZE_ANSWER_SYSTEM_PROMPT = """Here is a set of QA pairs: {context} Use these to synthesize an answer to the question: {question} """
