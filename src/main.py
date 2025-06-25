"""Main Streamlit application for RAG QA System."""

import streamlit as st
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from config import get_settings
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from query_engine import QueryEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'settings' not in st.session_state:
        try:
            st.session_state.settings = get_settings()
        except Exception as e:
            st.error(f"Ошибка конфигурации: {e}")
            st.stop()
    
    if 'vector_store_manager' not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager(
            api_key=st.session_state.settings.pinecone_api_key,
            environment=st.session_state.settings.pinecone_environment,
            index_name=st.session_state.settings.pinecone_index_name
        )
    
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor(
            max_file_size_mb=st.session_state.settings.max_file_size_mb,
            max_files_count=st.session_state.settings.max_files_count
        )
    
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = QueryEngine(st.session_state.vector_store_manager)
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    if 'index_stats' not in st.session_state:
        st.session_state.index_stats = {}

def render_sidebar():
    """Render sidebar with filters and settings."""
    st.sidebar.header("⚙️ Настройки поиска")
    
    # Search parameters
    similarity_top_k = st.sidebar.slider(
        "Количество результатов",
        min_value=1,
        max_value=10,
        value=st.session_state.settings.similarity_top_k,
        help="Максимальное количество документов для поиска"
    )
    
    similarity_threshold = st.sidebar.slider(
        "Порог релевантности",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.settings.similarity_threshold,
        step=0.1,
        help="Минимальная релевантность для отображения результатов"
    )
    
    # Metadata filters
    st.sidebar.header("🔍 Фильтры")
    
    author_filter = st.sidebar.text_input(
        "Автор документа",
        placeholder="Введите имя автора..."
    )
    
    # Store filters in session state
    st.session_state.search_params = {
        "similarity_top_k": similarity_top_k,
        "similarity_threshold": similarity_threshold,
        "metadata_filters": {}
    }
    
    if author_filter:
        st.session_state.search_params["metadata_filters"]["author"] = author_filter
    
    # Index statistics
    if st.session_state.index_stats:
        st.sidebar.header("📊 Статистика индекса")
        stats = st.session_state.index_stats
        st.sidebar.metric("Всего векторов", stats.get("total_vectors", 0))
        st.sidebar.metric("Размерность", stats.get("dimension", 0))
        if stats.get("index_fullness"):
            st.sidebar.metric("Заполненность", f"{stats['index_fullness']:.1%}")
    
    # Query statistics
    query_stats = st.session_state.query_engine.get_statistics()
    if query_stats["total_queries"] > 0:
        st.sidebar.header("📈 Статистика запросов")
        st.sidebar.metric("Всего запросов", query_stats["total_queries"])
        st.sidebar.metric("Средняя уверенность", f"{query_stats['avg_confidence']:.1f}%")
        
        if query_stats["language_distribution"]:
            st.sidebar.write("**Языки запросов:**")
            for lang, count in query_stats["language_distribution"].items():
                st.sidebar.write(f"- {lang}: {count}")

def render_document_upload():
    """Render document upload section."""
    st.header("📄 Загрузка документов")
    
    uploaded_files = st.file_uploader(
        "Выберите PDF файлы",
        type=['pdf'],
        accept_multiple_files=True,
        help=f"Максимум {st.session_state.settings.max_files_count} файлов, до {st.session_state.settings.max_file_size_mb}MB каждый"
    )
    
    if uploaded_files:
        st.write(f"**Выбрано файлов:** {len(uploaded_files)}")
        
        # Show file details
        for file in uploaded_files:
            file_size_mb = file.size / (1024 * 1024)
            st.write(f"- {file.name} ({file_size_mb:.1f}MB)")
        
        if st.button("🚀 Обработать документы", type="primary"):
            with st.spinner("Обработка документов..."):
                # Process documents
                documents = st.session_state.document_processor.process_uploaded_files(uploaded_files)
                
                if documents:
                    st.success(f"Успешно обработано {len(documents)} документов")
                    
                    # Initialize or create index
                    with st.spinner("Создание векторного индекса..."):
                        if not st.session_state.vector_store_manager.create_index():
                            st.error("Ошибка создания индекса Pinecone")
                            return
                        
                        if not st.session_state.vector_store_manager.create_vector_index(documents):
                            st.error("Ошибка создания векторного индекса")
                            return
                    
                    st.session_state.documents_loaded = True
                    st.session_state.index_stats = st.session_state.vector_store_manager.get_index_stats()
                    st.success("Документы успешно проиндексированы!")
                    st.rerun()
                else:
                    st.error("Не удалось обработать документы")

def render_query_interface():
    """Render query interface."""
    st.header("🤖 Поиск и вопросы")
    
    if not st.session_state.documents_loaded:
        # Try to load existing index
        if st.session_state.vector_store_manager.load_existing_index():
            st.session_state.documents_loaded = True
            st.session_state.index_stats = st.session_state.vector_store_manager.get_index_stats()
        else:
            st.warning("Сначала загрузите документы для индексации")
            return
    
    # Query input
    query = st.text_area(
        "Задайте вопрос по документам:",
        placeholder="Например: Что говорится о машинном обучении в документах?",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_button = st.button("🔍 Найти ответ", type="primary", disabled=not query.strip())
    
    with col2:
        if st.button("🗑️ Очистить историю"):
            st.session_state.query_engine.clear_history()
            st.success("История очищена")
            st.rerun()
    
    if search_button and query.strip():
        with st.spinner("Поиск ответа..."):
            result = st.session_state.query_engine.process_query(
                query=query,
                **st.session_state.search_params
            )
        
        if result["success"]:
            # Display answer
            st.subheader("💡 Ответ")
            st.write(result["answer"])
            
            # Display confidence
            confidence = result["confidence"]
            if confidence >= 80:
                confidence_color = "green"
                confidence_label = "Высокая"
            elif confidence >= 60:
                confidence_color = "orange"
                confidence_label = "Средняя"
            else:
                confidence_color = "red"
                confidence_label = "Низкая"
            
            st.metric(
                "Уверенность в ответе",
                f"{confidence:.1f}%",
                delta=f"{confidence_label}"
            )
            
            # Display sources
            if result["sources"]:
                st.subheader("📚 Источники")
                for source in result["sources"]:
                    with st.expander(f"📄 {source['filename']} (релевантность: {source['score']:.2f})"):
                        st.write(f"**Заголовок:** {source['title']}")
                        st.write(f"**Автор:** {source['author']}")
                        st.write(f"**Фрагмент текста:**")
                        st.write(source['text_snippet'])
        else:
            st.error(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")

def render_query_history():
    """Render query history."""
    st.header("📋 История запросов")
    
    history = st.session_state.query_engine.get_query_history()
    
    if not history:
        st.info("История запросов пуста")
        return
    
    for i, item in enumerate(reversed(history[-5:])):  # Show last 5
        with st.expander(f"❓ {item['query'][:50]}... ({item['timestamp'][:19]})"):
            st.write(f"**Вопрос:** {item['query']}")
            st.write(f"**Ответ:** {item['answer']}")
            st.write(f"**Уверенность:** {item['confidence']:.1f}%")
            st.write(f"**Язык:** {item['language']}")
            st.write(f"**Источников:** {len(item['sources'])}")

def main():
    """Main application function."""
    st.title("📚 RAG QA System")
    st.markdown("Система вопросов и ответов для исследовательских команд")
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📄 Документы", "🔍 Поиск", "📋 История"])
    
    with tab1:
        render_document_upload()
    
    with tab2:
        render_query_interface()
    
    with tab3:
        render_query_history()

if __name__ == "__main__":
    main()