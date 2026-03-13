from groq import Groq
from retriever.retriever import get_retriever
from config import GROQ_API_KEY, GROQ_MODEL_NAME


class SimpleRAGChain:
    def __init__(self):
        self.retriever = get_retriever()
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your environment or .env file.")
        self.client = Groq(api_key=GROQ_API_KEY)
        
    def invoke(self, inputs):
        query = inputs["input"]
        
        # Retrieve relevant documents
        try:
            docs = self.retriever.get_relevant_documents(query)
        except Exception:
            docs = self.retriever.invoke(query)
        
        # Combine context into labeled snippets with source/page for better grounding & citations
        context_parts = []
        total_chars = 0
        max_chars = 6000
        for idx, doc in enumerate(docs, start=1):
            text = (doc.page_content or "").strip()
            if not text:
                continue
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("source", "unknown")
            page = meta.get("page")
            label = f"S{idx}"
            header = f"[{label}] Source: {src}"
            if page is not None:
                header += f", Page: {page}"
            snippet = f"{header}\n{text}"
            if total_chars + len(snippet) > max_chars:
                remaining = max_chars - total_chars
                if remaining <= 0:
                    break
                snippet = snippet[:remaining]
            context_parts.append(snippet)
            total_chars += len(snippet)
        context = "\n\n".join(context_parts)
        
        # Create prompt
        system_msg = (
            "You are an expert RAG assistant. Follow these rules strictly:\n"
            "1) Use ONLY the provided context snippets to answer. Do NOT use outside knowledge.\n"
            "2) If the answer is not in the context, reply exactly: I don't know.\n"
            "3) Start with a direct 1–3 sentence answer. Be complete and precise.\n"
            "4) When relevant, compute values (e.g., years of experience) from dates in context. Show the computed value.\n"
            "5) Prefer exact facts: names, titles, numbers, dates, organizations, skills.\n"
            "6) If multiple snippets apply, synthesize them into one coherent answer.\n"
            "7) Cite the most relevant snippet IDs in square brackets, e.g., [S1], [S3].\n"
            "8) Do NOT speculate. If uncertain or conflicting, say I don't know."
        )
        user_msg = (
            f"Context snippets (each labeled [S#] with source/page):\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer (cite snippets like [S1], [S2] when applicable):"
        )

        # Generate answer via Groq Chat Completions
        completion = self.client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            top_p=1.0,
            max_tokens=384,
        )
        answer = completion.choices[0].message.content.strip()
        
        return {
            "answer": answer,
            "context": docs
        }


def get_rag_chain():
    return SimpleRAGChain()
