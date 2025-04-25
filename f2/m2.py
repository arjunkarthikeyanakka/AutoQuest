from pymilvus import connections, utility, db
import fitz  
from sentence_transformers import SentenceTransformer, CrossEncoder 
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
import numpy as np
from rank_bm25 import BM25Okapi 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer 
import re
import time
import os

def download_nltk_resources():
    """Download required NLTK resources if they aren't already present"""
    resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

def setup_milvus(host="localhost", port="19530", db_name="db_sciQA_m2"):
    """
    Set up connection to Milvus vector database
    
    Args:
        host: Milvus server host
        port: Milvus server port
        db_name: Name of the database to use
        
    Returns:
        bool: Connection status
    """
    try:
        connections.disconnect("default")
        connections.connect(alias="default", host=host, port=port, timeout=10)
        _ = utility.list_collections()
        
        if db_name not in db.list_database():
            db.create_database(db_name)
        
        db.using_database(db_name)
        print(f"Connected to Milvus and using database: {db_name}")
        print(f"Current collections: {utility.list_collections()}")
        return True
        
    except Exception as e:
        print(f"Milvus connection error: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF with layout preservation
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text with page markers
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            page_text = page.get_text("text")
            
            full_text += f"\n\n--- Page {page_num+1} ---\n\n"
            full_text += page_text
        
        return full_text
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text_semantic(text, chunk_size=150, overlap=30):
    """
    Split text into semantic chunks with paragraph and section awareness
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of chunks in words
        overlap: Number of words to overlap between chunks
        
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

def get_pdf_metadata(pdf_path):
    """
    Extract metadata from PDF file
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        dict: PDF metadata (title, author, etc.)
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "total_pages": doc.page_count
        }
        
        if not metadata["title"]:
            metadata["title"] = os.path.basename(pdf_path).replace(".pdf", "")
            
        return metadata
    except Exception as e:
        print(f"Error extracting PDF metadata: {e}")
        return {
            "title": os.path.basename(pdf_path).replace(".pdf", ""),
            "author": "",
            "subject": "",
            "keywords": "",
            "total_pages": 0
        }

def extract_section_headers(pdf_path):
    """
    Extract section headers from PDF for better context
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        list: List of dictionaries containing identified headers
    """
    try:
        doc = fitz.open(pdf_path)
        headers = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            for span in line["spans"]:
                                is_bold = span.get("font", "").lower().find("bold") >= 0
                                font_size = span.get("size", 0)
                                text = span.get("text", "").strip()
                                
                                if (is_bold or font_size > 12) and text and len(text) < 100:
                                    headers.append({
                                        "text": text,
                                        "page": page_num + 1,
                                        "font_size": font_size,
                                        "is_bold": is_bold
                                    })
        
        return headers
    except Exception as e:
        print(f"Error extracting section headers: {e}")
        return []

def extract_structured_text(pdf_path):
    """
    Extract text with structural information like sections
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        list: Structured content with section assignments
    """
    try:
        doc = fitz.open(pdf_path)
        structured_content = []
        
        current_page = 1
        current_section = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            for span in line["spans"]:
                                is_bold = span.get("font", "").lower().find("bold") >= 0
                                font_size = span.get("size", 0)
                                text = span.get("text", "").strip()
                                
                                if (is_bold or font_size > 12) and text and len(text) < 100:
                                    current_section = text
                                
                                if text:
                                    structured_content.append({
                                        "text": text,
                                        "page": page_num + 1,
                                        "section": current_section,
                                        "is_bold": is_bold,
                                        "font_size": font_size
                                    })
        
        return structured_content
    except Exception as e:
        print(f"Error extracting structured text: {e}")
        return []

def extract_keywords(text, top_n=10):
    """
    Extract key terms from text using TF-IDF
    
    Args:
        text: Input text
        top_n: Number of top keywords to extract
        
    Returns:
        str: Comma-separated keywords
    """
    try:
        if not text or not text.strip():
            return ""
            
        vectorizer = TfidfVectorizer(
            max_df=0.85,
            min_df=1,
            stop_words=stopwords.words('english'),
            use_idf=True
        )
        
        documents = [text, "This is a dummy document to enable TF-IDF"]
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        feature_names = vectorizer.get_feature_names_out()
        
        scores = tfidf_matrix[0].toarray()[0]
        
        tfidf_scores = [(term, scores[i]) for i, term in enumerate(feature_names)]
        tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        
        top_terms = [term for term, score in tfidf_scores[:top_n] if score > 0]
        
        return ", ".join(top_terms)
        
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return ""

def setup_enhanced_collection(embed_dim=768):
    """
    Create a Milvus collection with enhanced metadata fields
    
    Args:
        embed_dim: Dimension of embedding vectors
        
    Returns:
        Collection: Milvus collection for RAG system
    """
    try:
        if "rag_docs" in utility.list_collections():
            collection = Collection("rag_docs")
            if not collection.has_index():
                create_vector_index(collection)
                create_scalar_indexes(collection)
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
        
        create_vector_index(collection)
        
        create_scalar_indexes(collection)
        
        return collection
    except Exception as e:
        print(f"Error setting up collection: {e}")
        raise e

def create_vector_index(collection):
    """
    Create a vector index for fast similarity search
    
    Args:
        collection: Milvus collection
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
        print("Created HNSW index on embedding field")
    except Exception as e:
        print(f"Error creating vector index: {e}")

def create_scalar_indexes(collection):
    """
    Create scalar field indexes for faster filtering
    
    Args:
        collection: Milvus collection
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
        
        print("Created scalar field indexes for metadata filtering")
    except Exception as e:
        print(f"Error creating scalar indexes: {e}")

def ingest_document_improved(pdf_path, collection):
    """
    Enhanced document ingestion with semantic chunking and metadata
    
    Args:
        pdf_path: Path to the PDF file
        collection: Milvus collection
        
    Returns:
        dict: Ingestion statistics
    """
    start_time = time.time()
    print(f"Starting ingestion for: {pdf_path}")
    
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print("Error: Could not extract text from PDF")
            return None
            
        metadata = get_pdf_metadata(pdf_path)
        headers = extract_section_headers(pdf_path)
        
        chunks = chunk_text_semantic(text)
        print(f"Created {len(chunks)} semantic chunks")
        
        if not chunks:
            print("Error: No chunks created")
            return None
        
        embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        processed_data = []
        for i, chunk in enumerate(chunks):
            if not chunk or len(chunk) < 10:
                continue
                
            keywords = extract_keywords(chunk)
            
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
            print(f"Inserted batch {i//batch_size + 1}/{(len(processed_data)-1)//batch_size + 1}")
        
        collection.flush()
        
        elapsed_time = time.time() - start_time
        print(f"Successfully ingested document with {len(processed_data)} chunks in {elapsed_time:.2f} seconds")
        
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

def preprocess_query(query):
    """
    Advanced query preprocessing with synonym expansion
    
    Args:
        query: User question/query
        
    Returns:
        str: Expanded and preprocessed query
    """
    try:
        query = " ".join(query.split())
        
        abbreviations = {
            "w/": "with",
            "w/o": "without",
            "etc.": "etcetera",
            "e.g.": "for example",
            "i.e.": "that is",
            "vs.": "versus",
            "fig.": "figure",
            "eq.": "equation",
            "ref.": "reference",
        }
        
        for abbr, expanded in abbreviations.items():
            query = query.replace(abbr, expanded)
        
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(query.lower())
        key_terms = [word for word in tokens if word.isalnum() and word not in stop_words]
        
        expanded_terms = set(key_terms)
        max_synonyms_per_term = 2
        
        for term in key_terms:
            synsets = wordnet.synsets(term)
            count = 0
            for synset in synsets:
                for lemma in synset.lemmas():
                    if count >= max_synonyms_per_term:
                        break
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != term and synonym not in expanded_terms:
                        expanded_terms.add(synonym)
                        count += 1
        
        is_question = "?" in query
        question_starters = ["what", "how", "why", "when", "where", "who", "which", "whose", "whom", "is", "are", "can", "do", "does"]
        
        reformulated_query = query
        
        if is_question and not any(query.lower().startswith(starter) for starter in question_starters):
            reformulated_query = "explain " + query
        
        if "definition" in query.lower() or "mean" in query.lower():
            reformulated_query += " definition meaning concept"
        
        if "example" in query.lower():
            reformulated_query += " example illustration instance"
        
        expanded_query = reformulated_query + " " + " ".join(expanded_terms)
        
        return expanded_query
    except Exception as e:
        print(f"Error preprocessing query: {e}")
        return query

def hybrid_retrieve_chunks(collection, query, embedder, k=5, similarity_threshold=0.65):
    """
    Performs hybrid retrieval using both vector similarity and BM25 lexical search
    
    Args:
        collection: Milvus collection
        query: User query
        embedder: Sentence transformer model
        k: Number of results to retrieve
        similarity_threshold: Minimum similarity score threshold
        
    Returns:
        list: Retrieved chunks with scores
    """
    try:
        start_time = time.time()
        
        processed_query = preprocess_query(query)
        
        q_embedding = embedder.encode(processed_query).tolist()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 20, "ef": 64}
        }
        
        collection.load(_progress_bar=True)
        
        search_k = k * 5
        
        print("Collection fields:", collection.schema.fields)
        
        vector_results = collection.search(
            data=[q_embedding],
            anns_field="embedding",
            param=search_params,
            limit=search_k,
            output_fields=["chunk_text", "page_num", "section", "doc_title", "keywords"]
        )
        
        print(f"Found {len(vector_results)} result groups")
        if len(vector_results) > 0:
            print(f"First group has {len(vector_results[0])} hits")
            if len(vector_results[0]) > 0:
                first_hit = vector_results[0][0]
                print(f"First hit type: {type(first_hit)}")
                print(f"First hit details: {dir(first_hit)}")
                print(f"First hit entity: {dir(first_hit.entity)}")
                print(f"First hit entity dict: {vars(first_hit.entity)}")
        
        candidate_chunks = []
        
        if len(vector_results) > 0 and len(vector_results[0]) > 0:
            for hit in vector_results[0]:
                try:
                    entity = hit.entity
                    
                    chunk_text = ""
                    try:
                        chunk_text = entity.chunk_text
                    except:
                        try:
                            chunk_text = getattr(entity, "chunk_text", "")
                        except:
                            print("Unable to get chunk_text from entity")
                    
                    if not chunk_text:
                        continue
                    
                    similarity = 1.0 - hit.distance
                    
                    metadata = {}
                    for field in ["page_num", "section", "doc_title", "keywords"]:
                        try:
                            metadata[field] = getattr(entity, field, None)
                        except:
                            metadata[field] = None
                    
                    candidate_chunks.append({
                        "text": chunk_text,
                        "vec_score": similarity,
                        "combined_score": 0,
                        "metadata": metadata
                    })
                except Exception as e:
                    print(f"Error processing hit: {e}")
                    continue
        
        candidate_chunks = [c for c in candidate_chunks if c["vec_score"] >= similarity_threshold]
        
        if candidate_chunks:
            chunk_texts = [chunk["text"] for chunk in candidate_chunks]
            tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunk_texts]
            
            bm25 = BM25Okapi(tokenized_chunks)
            
            tokenized_query = word_tokenize(processed_query.lower())
            bm25_scores = bm25.get_scores(tokenized_query)
            
            if max(bm25_scores) > 0:
                bm25_scores = [score / max(bm25_scores) for score in bm25_scores]
            
            for i, chunk in enumerate(candidate_chunks):
                chunk["bm25_score"] = bm25_scores[i]
                chunk["combined_score"] = 0.7 * chunk["vec_score"] + 0.3 * bm25_scores[i]
            
            candidate_chunks.sort(key=lambda x: x["combined_score"], reverse=True)
        
        top_chunks = candidate_chunks[:k]
        
        if top_chunks:
            avg_vec_score = sum(chunk["vec_score"] for chunk in top_chunks) / len(top_chunks)
            avg_combined = sum(chunk["combined_score"] for chunk in top_chunks) / len(top_chunks)
            elapsed = time.time() - start_time
            print(f"Retrieved {len(top_chunks)} chunks in {elapsed:.2f}s with:")
            print(f"  - Avg vector similarity: {avg_vec_score:.4f}")
            print(f"  - Avg combined score: {avg_combined:.4f}")
        else:
            print("No suitable chunks found after filtering")
            
        return top_chunks
        
    except Exception as e:
        print(f"Error in hybrid retrieval: {e}")
        import traceback
        traceback.print_exc()
        return []

def contextual_rerank(query, candidate_chunks, top_k=5):
    """
    Rerank candidate chunks using a cross-encoder model for better context relevance
    
    Args:
        query: Original query
        candidate_chunks: Chunks from initial retrieval
        top_k: Number of top chunks to return
        
    Returns:
        list: Reranked chunks
    """
    try:
        start_time = time.time()
        
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        pairs = [(query, chunk["text"]) for chunk in candidate_chunks]
        
        scores = cross_encoder.predict(pairs)
        
        for i, score in enumerate(scores):
            candidate_chunks[i]["rerank_score"] = score
        
        ranked_chunks = sorted(candidate_chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        top_chunks = ranked_chunks[:top_k]
        
        elapsed = time.time() - start_time
        print(f"Reranking completed in {elapsed:.2f}s")
        
        return top_chunks
    except Exception as e:
        print(f"Error in reranking: {e}")
        return candidate_chunks[:top_k]

def retrieve_with_reranking(collection, query, embedder, k=5):
    """
    Main retrieval function using two-stage approach:
    1. Get candidate chunks with hybrid search
    2. Rerank candidates with cross-encoder
    
    Args:
        collection: Milvus collection
        query: User query
        embedder: Sentence transformer model
        k: Number of results to return
        
    Returns:
        list: Final chunks for answer generation
    """
    try:
        candidate_count = min(k * 3, 20)
        candidate_chunks = hybrid_retrieve_chunks(collection, query, embedder, k=candidate_count)
        
        if not candidate_chunks:
            print("No candidate chunks found")
            return []
        
        reranked_chunks = contextual_rerank(query, candidate_chunks, top_k=k)
        
        final_chunks = []
        for i, chunk in enumerate(reranked_chunks):
            metadata = chunk.get("metadata", {})
            formatted_chunk = {
                "id": i + 1,
                "text": chunk["text"],
                "score": chunk.get("rerank_score", chunk.get("combined_score", 0)),
                "page": metadata.get("page_num", 0),
                "section": metadata.get("section", ""),
                "document": metadata.get("doc_title", "")
            }
            final_chunks.append(formatted_chunk)
        
        return final_chunks
    except Exception as e:
        print(f"Error in retrieval with reranking: {e}")
        return []

def generate_answer(question, context_chunks, model_name="google/flan-t5-large"):
    """
    Generate answer from retrieved context using a language model
    
    Args:
        question: User question
        context_chunks: Retrieved context chunks
        model_name: Name of the language model to use
        
    Returns:
        str: Generated answer
    """
    try:
        if not context_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        context_text = ""
        for chunk in context_chunks:
            chunk_text = chunk["text"]
            page = chunk.get("page", "")
            section = chunk.get("section", "")
            
            if section:
                context_text += f"[Section: {section}] "
            if page:
                context_text += f"[Page: {page}] "
                
            context_text += chunk_text + "\n\n"
        
        prompt = f"""Answer the following question based ONLY on the provided context. 
If the context doesn't contain enough information to give a complete answer, say so.
If you're unsure about details, acknowledge the uncertainty.

Context:
{context_text}

Question: {question}

Answer:"""
        
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids
        
        output = model.generate(
            input_ids, 
            max_length=256,
            min_length=20,
            temperature=0.7,
            num_beams=4,
            no_repeat_ngram_size=3
        )
        
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        
        if "don't know" in answer.lower() or "no answer" in answer.lower() or "not enough information" in answer.lower():
            return "Based on the available information, I cannot provide a complete answer to your question. The document doesn't contain sufficient details on this specific topic."
        
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I encountered an error while generating the answer. Please try again with a different question."

def rag_query(question, collection, embedder=None):
    """
    Complete RAG pipeline function
    
    Args:
        question: User question
        collection: Milvus collection
        embedder: Optional pre-initialized embedder
        
    Returns:
        str: Generated answer with sources
    """
    start_time = time.time()
    
    try:
        if embedder is None:
            embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        print(f"Processing question: {question}")
        
        context_chunks = retrieve_with_reranking(collection, question, embedder)
        
        if not context_chunks:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or ask something else about the document."
        
        answer = generate_answer(question, context_chunks)
        
        elapsed = time.time() - start_time
        
        response = f"{answer}\n\n"
        response += f"--- Answer generated in {elapsed:.2f} seconds ---\n\n"
        response += "Sources:\n"
        
        for i, chunk in enumerate(context_chunks):
            page = chunk.get("page", "")
            section = chunk.get("section", "")
            doc = chunk.get("document", "")
            
            source_info = f"[{i+1}] "
            if doc:
                source_info += f"{doc} "
            if section:
                source_info += f"Section: {section}, "
            if page:
                source_info += f"Page: {page}"
                
            response += source_info + "\n"
        
        return response
    except Exception as e:
        print(f"Error in RAG query: {e}")
        return "I encountered an error while processing your question. Please try again."

def clear_collection(collection):
    """
    Delete all entities from the collection
    
    Args:
        collection: Milvus collection
        
    Returns:
        bool: Operation success status
    """
    try:
        collection.load()
        expr = "id >= 0"
        collection.delete(expr)
        collection.flush()
        print("Cleared all entities from the collection")
        return True
    except Exception as e:
        print(f"Error clearing collection: {e}")
        return False

if __name__ == "__main__":
    download_nltk_resources()
    
    if not setup_milvus():
        print("Failed to connect to Milvus")
        exit(1)
    
    if "rag_docs" in utility.list_collections():
        utility.drop_collection("rag_docs")
        print("Using existing collection")
    
    embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    collection = setup_enhanced_collection(embed_dim=768)
    
    clear_collection(collection)
    
    pdf_path = r"path to a pdf file..."
    stats = ingest_document_improved(pdf_path, collection) 
    if stats:
        print(f"Ingestion stats: {stats}")
    
    question = "Name some characters in Mahabharatam"
    
    answer = rag_query(question, collection, embedder)
    
    print("\nQuestion:", question)
    print("\nAnswer:", answer)