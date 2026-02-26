# llm_engine.py
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config



@st.cache_resource(show_spinner="Loading LLM...")
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        base_url=Config.LLM_BASE_URL,
        num_ctx=Config.LLM_NUM_CTX,
        num_thread=Config.LLM_NUM_THREAD,
        num_predict=200,           
        repeat_penalty=1.1,
        num_gpu=0,
        top_k=20,
        top_p=0.85,
        num_batch=128,
        keep_alive="10m",
    )



_PROMPT_TEMPLATE = """\
You are a precise document assistant. Answer using ONLY the information in <context>.

<context>
{context}
</context>

<question>{question}</question>

Instructions:
- Answer directly without restating the question.
- Use markdown formatting: **bold** for important names and terms.
- For lists (tiers, stages, types, steps, criteria): list ALL items found in the context, \
each as a bullet point with a full explanation. Do NOT stop early.
- For simple factual questions: answer in 2-4 sentences.
- If the answer is absent from context, say exactly: "This information is not in the provided document."
- Never invent facts. Never use outside knowledge.

Answer:"""



def _estimate_tokens(query: str) -> int:
    query_lower = query.lower()

    
    list_keywords = [
        "all", "list", "every", "each", "tiers", "stages", "types",
        "phases", "steps", "criteria", "describe all", "what are the",
        "enumerate", "summarize", "explain all", "give all", "show all",
        "categories", "levels", "classes", "components", "elements"
    ]

    
    short_keywords = [
        "what is", "who is", "when was", "define", "how many",
        "which", "does it", "is there", "what does"
    ]

    if any(kw in query_lower for kw in list_keywords):
        return 450   

    if any(kw in query_lower for kw in short_keywords):
        return 150   

    return 250       



def generate_streaming_answer(query: str, contexts: list[str]):
    if not contexts:
        yield (
            "⚠️ No relevant information was found in the document for your question. "
            "Try rephrasing or ask about a different topic."
        )
        return

    formatted_context = ""
    for i, ctx in enumerate(contexts, start=1):
        formatted_context += f"--- Document Section {i} ---\n{ctx.strip()}\n\n"

    
    base_llm = get_llm()
    llm = base_llm.copy(update={"num_predict": _estimate_tokens(query)})

    chain = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE) | llm | StrOutputParser()

    try:
        for chunk in chain.stream({"context": formatted_context, "question": query}):
            if chunk:
                yield chunk
    except Exception as e:
        yield (
            f"\n\n⚠️ LLM error: {e}\n\n"
            f"Please ensure Ollama is running (`ollama serve`) and the model is pulled "
            f"(`ollama pull {Config.LLM_MODEL}`)."
        )
