import logging
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class M2:
    def __init__(self):
        logger.info("Starting M2")
    

    def rag_query(self, question, collection, embedder=None):
        """Complete RAG pipeline function"""
        logger.info('Staring the Query Process...')
        start_time = time.time()
        
        try:
            if embedder is None:
                embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
            logger.info(f"Processing question: {question}")
            
            context_chunks = self.retrieve_with_reranking(collection, question, embedder)
            
            if not context_chunks:
                return "I couldn't find relevant information to answer your question. Please try rephrasing or ask something else about the document."
            
            answer = self.generate_answer(question, context_chunks)
            
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
            logger.exception(f"Error in RAG query: {e}")
            return "I encountered an error while processing your question. Please try again."
        

    def contextual_rerank(self, query, candidate_chunks, top_k=5):
        """
        Rerank candidate chunks using a cross-encoder model
        """
        logger.info(f"Reranking candidate chunks using a cross-encoder model started")
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
            logger.info(f"Reranking completed in {elapsed:.2f}s")
            
            return top_chunks
        except Exception as e:
            print(f"Error in reranking: {e}")
            return candidate_chunks[:top_k]
        

    def retrieve_with_reranking(self, collection, query, embedder, k=5):
        """
        Main retrieval function using two-stage approach:
        1. Get candidate chunks with hybrid search
        2. Rerank candidates with cross-encoder
        """
        logger.info(f"Retrieving with reranking started")
        try:
            candidate_count = min(k * 3, 20)
            candidate_chunks = self.hybrid_retrieve_chunks(collection, query, embedder, k=candidate_count)
            
            if not candidate_chunks:
                logger.warning("No candidate chunks found")
                return []
            
            reranked_chunks = self.contextual_rerank(query, candidate_chunks, top_k=k)
            
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
            logger.exception(f"Error in retrieval with reranking: {e}")
            return []

    def generate_answer(self, question, context_chunks, model_name="google/flan-t5-large"):
        """
        Enhanced answer generation with better model and context formatting
        """
        logger.info("Answer Generation Started")
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
            
            prompt = f"""Answer the following question ONLY based on the provided context. 
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
            logger.exception(f"Error generating answer: {e}")
            return "I encountered an error while generating the answer. Please try again with a different question."



    def hybrid_retrieve_chunks(self, collection, query, embedder, k=5, similarity_threshold=0.3):
        """
        Performs hybrid retrieval using both vector similarity and BM25 lexical search
        """
        logger.info("Working on hybrid retrieval using both vector similarity and BM25 lexical search.")
        try:
            start_time = time.time()
            
            processed_query = self.preprocess_query(query)
            
            q_embedding = embedder.encode(processed_query).tolist()
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 20, "ef": 64}
            }
            
            collection.load(_progress_bar=True)
            
            search_k = k * 5 
            
            logger.info(f"Collection fields: {collection.schema.fields}")
            
            vector_results = collection.search(
                data=[q_embedding],
                anns_field="embedding",
                param=search_params,
                limit=search_k,
                output_fields=["chunk_text", "page_num", "section", "doc_title", "keywords"]
            )
            
            logger.info(f"Found {len(vector_results)} result groups")
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
                                logger.exception("Unable to get chunk_text from entity")
                        
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
                        logger.exception(f"Error processing hit: {e}")
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
            logger.exception(f"Error in hybrid retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []
        

    def preprocess_query(self, query):
        """
        Advanced query preprocessing with synonym expansion
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
            logger.warning(f"Error preprocessing query: {e}")
            return query
