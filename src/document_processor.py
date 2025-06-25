"""Document processing module for PDF handling."""

import os
import logging
from typing import List, Dict, Any, Optional
from io import BytesIO
import PyPDF2
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from langdetect import detect
import streamlit as st

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handle PDF document processing and text extraction."""
    
    def __init__(self, max_file_size_mb: int = 50, max_files_count: int = 100):
        self.max_file_size_mb = max_file_size_mb
        self.max_files_count = max_files_count
        self.node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=20
        )
    
    def validate_files(self, uploaded_files: List[Any]) -> bool:
        """Validate uploaded files against size and count limits."""
        if len(uploaded_files) > self.max_files_count:
            st.error(f"Слишком много файлов: {len(uploaded_files)} > {self.max_files_count}")
            return False
        
        for file in uploaded_files:
            file_size_mb = file.size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                st.error(f"Файл {file.name} превышает максимальный размер: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB")
                return False
        
        return True
    
    def extract_pdf_metadata(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract metadata from PDF content."""
        metadata = {
            "title": filename,
            "author": "Unknown",
            "creation_date": None,
            "filename": filename
        }
        
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            if pdf_reader.metadata:
                metadata.update({
                    "title": pdf_reader.metadata.get("/Title", filename),
                    "author": pdf_reader.metadata.get("/Author", "Unknown"),
                    "creation_date": pdf_reader.metadata.get("/CreationDate", None)
                })
        except Exception as e:
            logger.warning(f"Could not extract metadata from {filename}: {e}")
        
        return metadata
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content."""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """Detect language of the text."""
        try:
            return detect(text[:1000])  # Use first 1000 chars for detection
        except:
            return "en"  # Default to English
    
    def process_uploaded_files(self, uploaded_files: List[Any]) -> List[Document]:
        """Process uploaded files and return LlamaIndex documents."""
        if not self.validate_files(uploaded_files):
            return []
        
        documents = []
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Read file content
                pdf_content = uploaded_file.read()
                
                # Extract text
                text = self.extract_text_from_pdf(pdf_content)
                if not text.strip():
                    st.warning(f"Не удалось извлечь текст из {uploaded_file.name}")
                    continue
                
                # Extract metadata
                metadata = self.extract_pdf_metadata(pdf_content, uploaded_file.name)
                metadata["language"] = self.detect_language(text)
                metadata["file_size"] = uploaded_file.size
                
                # Create LlamaIndex document
                document = Document(
                    text=text,
                    metadata=metadata
                )
                documents.append(document)
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            except Exception as e:
                st.error(f"Ошибка обработки файла {uploaded_file.name}: {e}")
                logger.error(f"Error processing {uploaded_file.name}: {e}")
        
        progress_bar.empty()
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Any]:
        """Split documents into chunks using LlamaIndex node parser."""
        nodes = []
        for document in documents:
            doc_nodes = self.node_parser.get_nodes_from_documents([document])
            nodes.extend(doc_nodes)
        return nodes