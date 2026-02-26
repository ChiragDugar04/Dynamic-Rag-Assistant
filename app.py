# app.py
import os
import time
import shutil
import streamlit as st

from config import Config
from utils import process_document_advanced, save_temp_file
from database import initialize_session_database, hybrid_retrieval, rerank_and_get_parents
from llm_engine import generate_streaming_answer

st.set_page_config(
    page_title="Dynamic RAG Assistant",
    page_icon="🛡️",
    layout="wide",
)


def _init_state():
    defaults = {
        "database_ready": False,
        "messages": [],
        "source_file_name": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


def cleanup_session():
    """Wipes session state and cached files, then reruns."""
    
    if "vector_db" in st.session_state:
        try:
            st.session_state.vector_db._client.reset()  
            del st.session_state.vector_db
        except Exception:
            pass

    keys = ["vector_db", "bm25_index", "raw_documents",
            "database_ready", "messages", "source_file_name"]
    for k in keys:
        st.session_state.pop(k, None)

    for folder in ["temp_uploads", Config.CHROMA_PATH]:
        if os.path.exists(folder):
            for attempt in range(3):
                try:
                    shutil.rmtree(folder)
                    break
                except PermissionError:
                    if attempt < 2:
                        time.sleep(1)  
                    else:
                        st.error(
                            f"Could not delete `{folder}` — file still locked by ChromaDB.\n\n"
                            "Please stop the app, manually delete the folder, and restart."
                        )
                        return  

    st.rerun()


with st.sidebar:
    st.header(" Document Session")
    st.caption(f"LLM: `{Config.LLM_MODEL}` · Embed: `{Config.EMBEDDING_MODEL}`")
    st.caption("Pipeline: PyMuPDF4LLM → Hybrid Search → BGE Reranker → qwen2.5")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type="pdf",
        help="Medical, legal, financial, or technical documents work best.",
    )

    if st.button(" Clear Session & Reset DB", use_container_width=True):
        cleanup_session()

    st.divider()
    st.markdown("**Tips for best results:**")
    st.markdown("- Ask specific questions\n- Reference section names if known\n- One topic per question")



st.title("🛡️ Dynamic RAG Assistant")



if not st.session_state.database_ready:
    if uploaded_file is None:
        st.info(
            " Welcome! Upload a PDF in the sidebar to start a private, local Q&A session.\n\n"
            
        )
        st.stop()

    
    with st.status(" Running ingestion pipeline...", expanded=True) as status:

        
        t0 = time.perf_counter()
        st.write(" Saving uploaded file...")
        temp_path = save_temp_file(uploaded_file)
        st.write(f"   ✅ Saved in {time.perf_counter() - t0:.1f}s")

        
        t1 = time.perf_counter()
        st.write(" Extracting & chunking document...")
        final_docs = process_document_advanced(temp_path)
        extract_time = time.perf_counter() - t1

        if not final_docs:
            status.update(label="❌ Ingestion failed", state="error")
            st.error(
                "Could not extract text from the PDF. "
                "This can happen with scanned/image-only PDFs. "
                "Try running OCR on the file first."
            )
            st.stop()

        st.write(f"   ✅ {len(final_docs)} chunks created in {extract_time:.1f}s")

        
        t2 = time.perf_counter()
        st.write(" Embedding chunks & building search indexes...")
        success = initialize_session_database(final_docs)
        embed_time = time.perf_counter() - t2

        if not success:
            status.update(label="❌ Indexing failed", state="error")
            st.stop()

        st.write(f"   ✅ Indexed in {embed_time:.1f}s")
        total_time = time.perf_counter() - t0
        st.write(f"⏱️ **Total ingestion time: {total_time:.1f}s**")

        st.session_state.database_ready = True
        st.session_state.source_file_name = uploaded_file.name
        st.write(" Warming up models...")
        from database import get_reranker
        from llm_engine import get_llm
        get_reranker()  
        get_llm()      
        st.write("   ✅ Models warm and ready!")
        status.update(label="✅ Ready to chat!", state="complete", expanded=False)

    st.rerun()



source_name = st.session_state.get("source_file_name", "your document")
st.success(f"✅ Active session: **{source_name}**")

with st.expander("🛠️ Debug: Retrieval Inspector"):
    test_query = st.text_input("Enter a query to inspect retrieval results:")
    if test_query:
        with st.spinner("Retrieving..."):
            hits = hybrid_retrieval(test_query)
            contexts = rerank_and_get_parents(test_query, hits)

        if hits:
            col1, col2 = st.columns(2)
            with col1:
                st.caption("**Top hybrid candidate (before reranking)**")
                st.text(hits[0].page_content[:400])
            with col2:
                st.caption("**Top reranked parent section**")
                st.text(contexts[0][:400] if contexts else "No context found")
        else:
            st.warning("No results returned. Check that the database is populated.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input(f"Ask a question about {source_name}...")

if prompt:
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching document..."):
            t_ret = time.perf_counter()
            initial_hits = hybrid_retrieval(prompt)
            final_contexts = rerank_and_get_parents(prompt, initial_hits)
            ret_time = time.perf_counter() - t_ret

        if not final_contexts:
            response = "I couldn't find relevant information in the document for that question."
            st.markdown(response)
        else:
            
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            status_placeholder.status("🤖 Generating answer...", state="running")

            t_llm = time.perf_counter()
            full_response = ""
            first_token_received = False

            for chunk in generate_streaming_answer(prompt, final_contexts):
                if not first_token_received:
                    
                    status_placeholder.empty()
                    first_token_received = True
                full_response += chunk
                
                response_placeholder.markdown(full_response + "▌")

            
            response_placeholder.markdown(full_response)
            llm_time = time.perf_counter() - t_llm
            st.caption(f" Retrieval: {ret_time:.1f}s ·  LLM: {llm_time:.1f}s ·  {len(final_contexts)} section(s) used")
            response = full_response

        st.session_state.messages.append({"role": "assistant", "content": response})
