import logging

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from app.prompts import (
    MULTI_QUERY_TEMPLATE,
    QUERY_DECOMPOSITION_TEMPLATE,
    SUB_QUESTION_QA_TEMPLATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def get_unique_docs(documents: list[list]):
    """Unique union of retrieved docs"""
    flattened_docs = [doc for sublist in documents for doc in sublist]

    unique_docs = []
    seen_content = set()

    for doc in flattened_docs:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_docs.append(doc)

    return unique_docs


def create_multi_query_retriever(llm, base_retriever):
    """Create a multi-query retriever that generates query variations"""
    prompt_perspectives = ChatPromptTemplate.from_template(MULTI_QUERY_TEMPLATE)

    # Generate queries
    generate_queries = (
        prompt_perspectives
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )

    # Create the multi-query retrieval chain
    multi_query_retriever = generate_queries | base_retriever.map() | get_unique_docs

    return multi_query_retriever


def create_query_decomposition_retriever(llm, base_retriever):
    """Create a query decomposition retriever that breaks down queries into sub-questions"""
    prompt_decomposition = ChatPromptTemplate.from_template(
        QUERY_DECOMPOSITION_TEMPLATE
    )

    # Step 1: Generate sub-questions
    generate_sub_questions = (
        prompt_decomposition
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )

    # Step 2: QA chain for sub-questions
    prompt_rag = ChatPromptTemplate.from_template(SUB_QUESTION_QA_TEMPLATE)
    sub_qa_chain = prompt_rag | llm | StrOutputParser()

    # Step 3: Process sub-questions
    def process_decomposition(inputs):
        """Decompose question, answer sub-questions, format for synthesis"""
        question = inputs.get("question") or inputs.get("input") or inputs

        # Generate sub-questions
        sub_questions = generate_sub_questions.invoke({"question": question})

        logging.info("\n=== DECOMPOSED SUB-QUESTIONS ===")
        for i, sq in enumerate(sub_questions, 1):
            logging.info(f"{i}. {sq}")
        logging.info("=" * 60)

        # Answer each sub-question
        sub_answers = []
        for sub_question in sub_questions:
            retrieved_docs = base_retriever.invoke(sub_question)
            answer = sub_qa_chain.invoke(
                {"context": retrieved_docs, "question": sub_question}
            )
            sub_answers.append(answer)

        logging.info("\n=== SUB-QUESTION ANSWERS ===")
        for i, (sq, ans) in enumerate(zip(sub_questions, sub_answers, strict=False), 1):
            logging.info(f"\nQ{i}: {sq}")
            logging.info(f"A{i}: {ans[:200]}..." if len(ans) > 200 else f"A{i}: {ans}")
        logging.info("=" * 60 + "\n")

        # Format Q&A pairs
        formatted_context = ""
        for i, (sq, ans) in enumerate(
            zip(sub_questions, sub_answers, strict=False), start=1
        ):
            formatted_context += f"Question {i}: {sq}\nAnswer {i}: {ans}\n\n"

        return {
            "context": formatted_context.strip(),
            "question": question,
            "sub_questions": sub_questions,
            "sub_answers": sub_answers,
        }

    return RunnableLambda(process_decomposition)
