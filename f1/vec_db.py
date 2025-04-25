import time
import re
import logging
from pymilvus import connections, utility, db
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class vecDB():
    """
    Main class for vector database operations using Milvus.
    Handles connection, setup, and document ingestion with semantic chunking.
    """
    def __init__(self, host="localhost", port="19530", db_name="db_sciQA_m2"):
        """
        Initialize the vector database with connection parameters.
        
        Args:
            host: Milvus server hostname (default: "localhost")
            port: Milvus server port (default: "19530")
            db_name: Database name to use (default: "db_sciQA_m2")
        """
        self.host = host
        self.port = port
        self.db_name = db_name

    def setup_milvus(self):
        """
        Connect to Milvus server and set up the database.
        Ensures proper connection and database creation if needed.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Setting up Milvus...")
        host = self.host
        port = self.port
        db_name = self.db_name
        try:
            connections.disconnect("default")
            connections.connect(alias="default", host=host, port=port, timeout=10)
            _ = utility.list_collections()
            
            if db_name not in db.list_database():
                db.create_database(db_name)
            
            db.using_database(db_name)
            logger.info(f"Connected to Milvus and using database: {db_name}")
            logger.info(f"Current collections: {utility.list_collections()}")
            return True
            
        except Exception as e:
            logger.exception(f"Milvus connection error: {e}")
            return False
        
    def checkForCollection(self, colName):
        """
        Check if a collection exists and drop it if it does.
        Note: This actually drops the collection rather than just checking,
        which seems contrary to the method name.
        
        Args:
            colName: Name of the collection to check
        """
        logger.info('Checking collections')
        if colName in utility.list_collections():
            utility.drop_collection(colName)
            logger.info("Using existing collection")
    
    def setup_enhanced_collection(self, embed_dim=768):
        """
        Create or get the RAG document collection with metadata fields.
        Sets up indexes for both vector and scalar fields.
        
        Args:
            embed_dim: Dimension of the embedding vectors (default: 768)
            
        Returns:
            Collection object for the RAG documents
            
        Raises:
            Exception: If collection setup fails
        """
        logger.info("Setting up collection...")
        try:
            if "rag_docs" in utility.list_collections():
                collection = Collection("rag_docs")
                if not collection.has_index():
                    self.create_vector_index(collection)
                    self.create_scalar_indexes(collection)
                return collection
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embed_dim),
                FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="page_num", dtype=DataType.INT64),
                FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="chunk_position", dtype=DataType.INT64),
                FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=1024)
            ]
            
            schema = CollectionSchema(fields, description="Enhanced RAG collection with metadata")
            collection = Collection("rag_docs", schema)
            
            self.create_vector_index(collection)
            
            self.create_scalar_indexes(collection)
            
            return collection
        except Exception as e:
            logger.exception(f"Error setting up collection: {e}")
            raise e
    
    def create_vector_index(self, collection):
        """
        Create an HNSW index on the embedding field for efficient vector search.
        HNSW offers a good balance of speed and recall for vector similarity search.
        
        Args:
            collection: Milvus collection object
        """
        try:
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }
            collection.create_index("embedding", index_params)
            logger.info("Created HNSW index on embedding field")
        except Exception as e:
            logger.exception(f"Error creating vector index: {e}")


    def create_scalar_indexes(self, collection):
        """
        Create indexes on scalar fields for filtering during search.
        Enables efficient metadata-based filtering alongside vector search.
        
        Args:
            collection: Milvus collection object
        """
        try:
            collection.create_index(
                "page_num", 
                {"index_type": "STL_SORT"},
                index_name="idx_page_num"
            )
            
            collection.create_index(
                "section",
                {"index_type": "INVERTED"},
                index_name="idx_section"
            )
            
            logger.info("Created scalar field indexes for metadata filtering")
        except Exception as e:
            logger.exception(f"Error creating scalar indexes: {e}")
    
    def clear_collection(self, collection):
        """
        Delete all entities from the collection.
        Useful for resetting the collection before reingesting documents.
        
        Args:
            collection: Milvus collection object
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            collection.load()
            expr = "id >= 0"
            collection.delete(expr)
            collection.flush()
            logger.info("Cleared all entities from the collection")
            return True
        except Exception as e:
            logger.exception(f"Error clearing collection: {e}")
            return False
        


    def ingest_document_improved(self, text, metadata, headers, collection):
        """
        Process and ingest document text with metadata into the vector database.
        Chunks the text semantically, extracts keywords, and computes embeddings.
        
        Args:
            text: Document text to ingest
            metadata: Dictionary with document metadata (title, etc.)
            headers: List of section headers with their text
            collection: Milvus collection to insert into
            
        Returns:
            dict: Statistics about the ingestion or None if failed
        """
        logger.info("Ingesting the given text into the database...")
        start_time = time.time()
        
        try:
            chunks = self.chunk_text_semantic(text)
            logger.info(f"Created {len(chunks)} semantic chunks")
            
            if not chunks:
                logger.error("Error: No chunks created")
                return None
            
            embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
            processed_data = []
            for i, chunk in enumerate(chunks):
                if not chunk or len(chunk) < 10:
                    continue
                    
                keywords = self.extract_keywords_improved(chunk)
                
                page_num = 1
                section = ""
                
                page_marker_match = re.search(r'--- Page (\d+) ---', chunk)
                if page_marker_match:
                    page_num = int(page_marker_match.group(1))
                
                for header in headers:
                    if header["text"] in chunk:
                        section = header["text"]
                        break
                
                processed_data.append({
                    "text": chunk,
                    "page_num": page_num,
                    "section": section or "unknown",
                    "position": i,
                    "doc_title": metadata.get("title", ""),
                    "keywords": keywords
                })
            
            batch_size = 100
            for i in range(0, len(processed_data), batch_size):
                batch = processed_data[i:i+batch_size]
                
                embeddings = [embedder.encode(item["text"]).tolist() for item in batch]
                texts = [item["text"] for item in batch]
                page_nums = [item["page_num"] for item in batch]
                sections = [item["section"] for item in batch]
                positions = [item["position"] for item in batch]
                doc_titles = [item["doc_title"] for item in batch]
                keywords_list = [item["keywords"] for item in batch]
                
                insert_data = [
                    embeddings,
                    texts,
                    page_nums,
                    sections,
                    doc_titles,
                    positions,
                    keywords_list
                ]
                
                collection.insert(insert_data)
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(processed_data)-1)//batch_size + 1}")
            
            collection.flush()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully ingested document with {len(processed_data)} chunks in {elapsed_time:.2f} seconds")
            
            return {
                "total_chunks": len(processed_data),
                "avg_chunk_length": sum(len(item["text"]) for item in processed_data) / max(1, len(processed_data)),
                "title": metadata.get("title", "Unknown"),
                "pages": metadata.get("total_pages", 0),
                "processing_time": elapsed_time
            }
        except Exception as e:
            print(f"Error in document ingestion: {e}")
            return None


    def chunk_text_semantic(self, text, chunk_size=150, overlap=30):
        """
        Split text into semantic chunks with controlled size and overlap.
        Tries to preserve paragraph and sentence boundaries when possible.
        
        Args:
            text: Input text to chunk
            chunk_size: Target size in words for each chunk (default: 150)
            overlap: Number of words to overlap between chunks (default: 30)
            
        Returns:
            list: List of text chunks
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            if len(paragraph.split()) > chunk_size * 2:
                sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
                for sentence in sentences:
                    words = sentence.split()
                    if current_size + len(words) <= chunk_size:
                        current_chunk.append(sentence)
                        current_size += len(words)
                    else:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        
                        if len(current_chunk) > 0 and len(words) + overlap < chunk_size:
                            overlap_text = current_chunk[-min(overlap, len(current_chunk)):]
                            current_chunk = overlap_text + [sentence]
                            current_size = sum(len(t.split()) for t in current_chunk)
                        else:
                            current_chunk = [sentence]
                            current_size = len(words)
            else:
                words_in_para = len(paragraph.split())
                
                if current_size + words_in_para <= chunk_size:
                    current_chunk.append(paragraph)
                    current_size += words_in_para
                else:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    
                    current_chunk = [paragraph]
                    current_size = words_in_para
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        validated_chunks = []
        for chunk in chunks:
            if len(chunk) > 8000:
                words = chunk.split()
                for i in range(0, len(words), chunk_size):
                    sub_chunk = ' '.join(words[i:i+chunk_size])
                    if len(sub_chunk) <= 8000:
                        validated_chunks.append(sub_chunk)
            else:
                validated_chunks.append(chunk)
        
        return validated_chunks
    
    def extract_keywords_improved(self, text, top_n=10, use_phrases=True, use_context=None):
        """
        Extract key terms from text using enhanced methods beyond basic TF-IDF
        
        Args:
            text: Input text to extract keywords from
            top_n: Number of top keywords to return
            use_phrases: Whether to extract multi-word phrases
            use_context: Optional corpus of related texts for better IDF calculation
            
        Returns:
            String of comma-separated keywords
        """
        logger.info("Extracting Keywords...")
        try:
            if not text or not text.strip():
                return ""
                
            processed_text = text.lower()
            processed_text = re.sub(r'[^\w\s]', ' ', processed_text)
            
            ngram_range = (1, 3) if use_phrases else (1, 1)
            
            vectorizer = TfidfVectorizer(
                max_df=0.85,
                min_df=1,
                stop_words=stopwords.words('english'),
                ngram_range=ngram_range,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            )
            
            if use_context and isinstance(use_context, list) and len(use_context) > 0:
                corpus = use_context + [processed_text]
                tfidf_matrix = vectorizer.fit_transform(corpus)
                doc_index = len(corpus) - 1
            else:
                corpus = [processed_text, "This is a dummy document to enable TF-IDF"]
                tfidf_matrix = vectorizer.fit_transform(corpus)
                doc_index = 0
                
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix[doc_index].toarray()[0]
            
            tfidf_scores = [(term, scores[i]) for i, term in enumerate(feature_names)]
            tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
            
            filtered_scores = []
            for term, score in tfidf_scores:
                if score <= 0:
                    continue
                    
                if len(term) < 3 and not term.isupper():
                    continue
                    
                if ' ' in term:
                    words = term.split()
                    if all(word in stopwords.words('english') for word in words):
                        continue
                        
                    if any(term in other_term for other_term, _ in filtered_scores):
                        continue
                
                filtered_scores.append((term, score))
                
                if len(filtered_scores) >= top_n:
                    break
            
            top_terms = [term for term, _ in filtered_scores[:top_n]]
            logger.info("Completed Extracting Keywords.")
            return ", ".join(top_terms)
            
        except Exception as e:
            logger.exception(f"Error extracting keywords: {e}")
            return ""