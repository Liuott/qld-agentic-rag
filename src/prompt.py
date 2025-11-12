from langchain.prompts import ChatPromptTemplate

RAG_FALLBACK = ChatPromptTemplate.from_template(
    """
    You are a careful assistant for Queensland road rules. Answer strictly based on the provided context.
    - Cite the specific clause number and source (Road Rules 2009 / Driving Handout text clauses) when possible.
    - If the answer cannot be supported by the context, say you don't know and ask for a more specific query.
    - You MUST include a source line at the end in the format: "Source: <Act/MUTCD, clause #>".
    - If no clause in the context supports the answer, reply: "I don't know. Please provide a more specific clause or scenario."

    --------------------
    Context:
    {context}
    --------------------
    Question: {question}
    """
)


EVAL_TEMPLATE = ChatPromptTemplate.from_template(
    """
    You are grading a QA for Queensland road rules.
    Role: strict examiner. Only judge based on the given context.


    Question: {question}
    Retrieved Context: {context}
    Candidate Answer: {answer}


    Please output a JSON with fields:
    - correctness: one of ["yes", "no"] 
    - relevance: one of ["high", "medium", "low"] 
    - faithfulness: one of ["yes", "no"] 
    - rationale: short explanation in English
    """
)

GEN_QA_TEMPLATE = ChatPromptTemplate.from_template(
    """
    You are creating exam-style QA pairs for Queensland road rules.
    Use only the snippet below to propose ONE useful user question and its ground-truth short answer.


    Snippet:
    {snippet}


    Output JSON with fields: question, ground_truth.
    """
)