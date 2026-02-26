# utils.py
import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from config import Config


def pdf_to_markdown(file_path: str) -> str:
    try:
        import pymupdf4llm  
        md_text = pymupdf4llm.to_markdown(file_path)
        return md_text
    except ImportError:
        st.warning("pymupdf4llm not found — falling back to basic PyMuPDF extraction.")
        return _pymupdf_fallback(file_path)
    except Exception as e:
        st.error(f"PDF parsing error: {e}")
        raise


def _pymupdf_fallback(file_path: str) -> str:
    import fitz  
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(pages)



HEADERS_TO_SPLIT = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

def _build_children_for_parent(parent_index: int, parent_doc, source_name: str):
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHILD_CHUNK_SIZE,
        chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_text = parent_doc.page_content
    section_meta = parent_doc.metadata  

    child_texts = child_splitter.split_text(parent_text)
    children = []

    for child_text in child_texts:
        if not child_text.strip():
            continue
        doc = Document(
            page_content=child_text,
            metadata={
                "parent_id": parent_index,
                "parent_text": parent_text,          
                "source": source_name,
                **section_meta,                       
            },
        )
        children.append(doc)

    return children


def process_document_advanced(file_path: str):
    try:
        source_name = os.path.basename(file_path)
        st.write(" Extracting text from PDF...")
        full_markdown = pdf_to_markdown(file_path)

        if not full_markdown or len(full_markdown.strip()) < 50:
            st.error("Extracted markdown is empty or too short. The PDF may be scanned/image-only.")
            return None

        
        st.write(" Splitting into semantic sections...")
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT,
            strip_headers=False,   
        )
        parent_docs = markdown_splitter.split_text(full_markdown)

        if not parent_docs:
            st.error("No sections found after markdown splitting.")
            return None

        st.write(f"   Found {len(parent_docs)} parent sections.")
        
        st.write("  Building child chunks in parallel...")
        all_children: list[Document] = []

        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = {
                executor.submit(_build_children_for_parent, i, parent, source_name): i
                for i, parent in enumerate(parent_docs)
            }
            for future in as_completed(futures):
                try:
                    all_children.extend(future.result())
                except Exception as exc:
                    st.warning(f"Section {futures[future]} skipped due to error: {exc}")

        if not all_children:
            st.error("No child chunks were created.")
            return None

        st.write(f"   Total child chunks ready: {len(all_children)}")
        return all_children

    except Exception as e:
        st.error(f"Ingestion pipeline error: {e}")
        print(f"[utils] Error detail: {e}")
        return None



def save_temp_file(uploaded_file) -> str:
    """Saves Streamlit upload to disk and returns the path."""
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = os.path.join("temp_uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
