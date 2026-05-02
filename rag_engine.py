"""
rag_engine.py — RAG pipeline using LangChain + Google GenAI + Chroma
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DIR = Path(__file__).parent / ".chroma_db"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert HR assistant for ACME Corporation with deep knowledge of all company policies.
Your role is to help employees understand company policies accurately and clearly.

Guidelines:
- Answer ONLY based on the retrieved policy context provided below.
- If the answer is not in the context, say so honestly — do NOT make up policies.
- Be concise, friendly, and professional.
- When citing a specific policy section, mention it (e.g., "According to Section 2.1...").
- Use bullet points or numbered lists for clarity when appropriate.
- If an employee seems confused or the question is complex, offer to clarify further.

Retrieved Policy Context:
{context}
"""

# ── Loader ────────────────────────────────────────────────────────────────────

def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Load all .txt and .pdf files from the docs directory."""
    docs: list[Document] = []

    # Load .txt files
    txt_files = list(docs_dir.glob("*.txt"))
    for txt_file in txt_files:
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs.extend(loader.load())

    # Load .pdf files
    pdf_files = list(docs_dir.glob("*.pdf"))
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())

    return docs


# ── Vector store ──────────────────────────────────────────────────────────────

def build_vectorstore(api_key: str, force_rebuild: bool = False) -> Chroma:
    """Build or load a persisted Chroma vector store."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=api_key)
    
    # Use existing store if it exists and rebuild not forced
    if CHROMA_DIR.exists() and not force_rebuild:
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        # If store is empty for some reason, rebuild
        if vectorstore._collection.count() > 0:
            return vectorstore

    # Build from scratch
    raw_docs = load_documents()
    if not raw_docs:
        raise ValueError(
            f"No documents found in '{DOCS_DIR}'. "
            "Please add .txt or .pdf policy files there."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n━", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


# ── RAG chain ─────────────────────────────────────────────────────────────────

class PolicyRAGEngine:
    """Manages the end-to-end RAG pipeline with conversational history."""

    def __init__(self, api_key: str, force_rebuild: bool = False):
        self.api_key = api_key
        self.vectorstore = build_vectorstore(api_key, force_rebuild=force_rebuild)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={"k": TOP_K, "fetch_k": 20},
        )
        self.llm = ChatOpenAI(model=CHAT_MODEL, openai_api_key=api_key, temperature=0.2, max_tokens=2048)
        self._build_chain()

    def _build_chain(self):
        """Assemble the RAG chain with history-aware retrieval."""

        # Step 1: Condense the follow-up question with chat history into a
        #         standalone question for the retriever.
        condense_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given the conversation history and a follow-up question, "
             "rewrite the follow-up question to be a fully self-contained, "
             "standalone question that captures all necessary context. "
             "Return ONLY the rewritten question, no explanation."),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

        condense_chain = condense_prompt | self.llm | StrOutputParser()

        # Step 2: Full QA chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

        def retrieve_context(inputs: dict) -> dict:
            """Condense question → retrieve docs → format context."""
            history = inputs.get("chat_history", [])
            question = inputs["question"]

            if history:
                standalone_q = condense_chain.invoke(
                    {"chat_history": history, "question": question}
                )
            else:
                standalone_q = question

            docs = self.retriever.invoke(standalone_q)
            context = "\n\n---\n\n".join(
                f"[Source: {doc.metadata.get('source', 'policy')}]\n{doc.page_content}"
                for doc in docs
            )
            return {
                "context": context,
                "question": question,
                "chat_history": history,
                "source_docs": docs,
            }

        self._retrieve_context = retrieve_context

        self._qa_chain = (
            qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def query(
        self,
        question: str,
        chat_history: Optional[list] = None,
    ) -> dict:
        """
        Run a RAG query.

        Returns:
            dict with keys: answer (str), source_docs (list[Document])
        """
        history = chat_history or []

        enriched = self._retrieve_context({
            "question": question,
            "chat_history": history,
        })

        answer = self._qa_chain.invoke({
            "context": enriched["context"],
            "question": enriched["question"],
            "chat_history": enriched["chat_history"],
        })

        return {
            "answer": answer,
            "source_docs": enriched["source_docs"],
        }

    def get_doc_count(self) -> int:
        """Return the number of chunks in the vector store."""
        return self.vectorstore._collection.count()
