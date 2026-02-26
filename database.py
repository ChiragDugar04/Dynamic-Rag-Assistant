# database.py
import re
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from config import Config


@st.cache_resource(show_spinner="Loading reranker model...")
def get_reranker() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-4-v2", device="cpu")


@st.cache_resource(show_spinner="Connecting to Ollama embeddings...")
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=Config.EMBEDDING_MODEL)



def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def initialize_session_database(documents: list) -> bool:
    embeddings = get_embeddings()
    batch_size = Config.EMBEDDING_BATCH_SIZE

    #  Batched embedding 
    all_texts = [doc.page_content for doc in documents]
    all_metadatas = [doc.metadata for doc in documents]
    all_embeddings: list[list[float]] = []

    total_batches = (len(all_texts) + batch_size - 1) // batch_size
    progress = st.progress(0, text="Embedding chunks...")

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_texts))
        batch_texts = all_texts[start:end]

        batch_vectors = embeddings.embed_documents(batch_texts)
        all_embeddings.extend(batch_vectors)

        pct = int((batch_idx + 1) / total_batches * 100)
        progress.progress(pct, text=f"Embedding chunks... {pct}%")

    progress.empty()

    vector_db = Chroma(
        collection_name=Config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=Config.CHROMA_PATH,
    )
    vector_db._collection.add(
        ids=[str(i) for i in range(len(all_texts))],
        embeddings=all_embeddings,
        documents=all_texts,
        metadatas=all_metadatas,
    )

   
    tokenized_corpus = [_tokenize(t) for t in all_texts]
    bm25_index = BM25Okapi(tokenized_corpus)

    st.session_state.vector_db = vector_db
    st.session_state.bm25_index = bm25_index
    st.session_state.raw_documents = documents

    return True


def hybrid_retrieval(query: str, k: int = Config.RETRIEVAL_K) -> list:
    if "vector_db" not in st.session_state:
        embeddings = get_embeddings()
        st.session_state.vector_db = Chroma(
            persist_directory=Config.CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=Config.COLLECTION_NAME,
        )

    vector_db: Chroma = st.session_state.vector_db
    bm25: BM25Okapi = st.session_state.bm25_index
    docs: list = st.session_state.raw_documents

    # Dense retrieval
    vector_results = vector_db.similarity_search(query, k=k)

    # Sparse retrieval
    tokenized_query = _tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = bm25_scores.argsort()[-k:][::-1]
    keyword_results = [docs[i] for i in top_bm25_idx]

    # Deduplicate by content hash
    seen: dict[str, object] = {}
    for doc in (vector_results + keyword_results):
        key = doc.page_content[:120]  
        if key not in seen:
            seen[key] = doc

    return list(seen.values())



def rerank_and_get_parents(query: str, initial_results: list, top_n: int = Config.RERANK_TOP_N) -> list[str]:
    if not initial_results:
        return []

    reranker = get_reranker()

    pairs = [[query, doc.page_content] for doc in initial_results]
    scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)

    ranked = sorted(zip(scores, initial_results), key=lambda x: x[0], reverse=True)

    final_contexts: list[str] = []
    seen_parents: set = set()
    total_chars = 0

    for score, doc in ranked:
        raw_parent = doc.metadata.get("parent_text", doc.page_content)
        parent_text = extract_relevant_sentences(raw_parent, doc.page_content, n_sentences=4)
        parent_id = doc.metadata.get("parent_id", hash(parent_text[:80]))

        if parent_id in seen_parents:
            continue

        
        if total_chars + len(parent_text) > Config.MAX_CONTEXT_CHARS:
            remaining = Config.MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:   
                parent_text = parent_text[:remaining] + "…"
            else:
                break

        seen_parents.add(parent_id)
        final_contexts.append(parent_text)
        total_chars += len(parent_text)

        if len(final_contexts) >= top_n:
            break

    return final_contexts

def extract_relevant_sentences(parent_text: str, child_text: str, n_sentences: int = 4) -> str:
    import re
    
    sentences = re.split(r'(?<=[.!?])\s+', parent_text.strip())
    
    if len(sentences) <= n_sentences:
        return parent_text 
    
    child_words = set(_tokenize(child_text))
    
    scored = []
    for i, sentence in enumerate(sentences):
        sentence_words = set(_tokenize(sentence))
        overlap = len(child_words & sentence_words)
        scored.append((overlap, i, sentence))
    
    
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:n_sentences]
    top_by_position = sorted(top, key=lambda x: x[1])
    
    return " ".join(s for _, _, s in top_by_position)
