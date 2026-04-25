from operator import itemgetter

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation history and a follow-up question, rewrite the "
            "follow-up as a standalone question that captures all necessary context. "
            "If the question is already standalone, return it unchanged.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert assistant. Answer the user's question using ONLY the "
            "information in the context below. If the answer is not in the context, "
            "say 'I don't have enough information to answer that from the provided document.'\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(retriever, api_key: str, model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=model,
        temperature=0,
        max_tokens=1500,
    )

    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    def condense_question(inputs: dict) -> str:
        if inputs.get("chat_history"):
            return condense_chain.invoke(inputs)
        return inputs["question"]

    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=RunnableLambda(condense_question)
        )
        | RunnablePassthrough.assign(
            source_documents=lambda x: retriever.invoke(x["standalone_question"]),
        )
        | RunnablePassthrough.assign(
            context=lambda x: _format_docs(x["source_documents"]),
        )
        | {
            "answer": QA_PROMPT | llm | StrOutputParser(),
            "source_documents": itemgetter("source_documents"),
        }
    )

    return rag_chain


def make_history(messages: list[dict]) -> list:
    """Convert [{role, content}] display messages to LangChain message objects."""
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history
