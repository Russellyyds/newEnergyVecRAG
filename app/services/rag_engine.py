from typing import List, Dict, Any, Optional, Union
import asyncio
import logging
from datetime import datetime

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from app.config.settings import settings
from app.models.models import RetrievalStrategy, DocumentType, RetrievedChunk

logger = logging.getLogger(__name__)


class RagEngine:
    def __init__(self):
        """Initialize the RAG Engine."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=0
        )

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY
        )

        # Initialize vector database
        self.vector_db = Chroma(
            persist_directory=settings.VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name="new_energy_knowledge_base"
        )

        # Initialize retrieval strategies
        self.retrieval_strategies = {
            RetrievalStrategy.SEMANTIC: self._semantic_retrieval,
            RetrievalStrategy.KEYWORD: self._keyword_retrieval,
            RetrievalStrategy.HYBRID: self._hybrid_retrieval,
            RetrievalStrategy.METADATA: self._metadata_retrieval
        }

        # Default configuration
        self.config = {
            "retrieval_strategies": [RetrievalStrategy.HYBRID],
            "reranking_enabled": True,
            "vector_similarity_threshold": 0.7,
            "max_chunks_per_strategy": 5
        }

        # Setup document reranker
        self.document_compressor = LLMChainExtractor.from_llm(self.llm)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.document_compressor,
            base_retriever=self.vector_db.as_retriever()
        )

        logger.info("RAG Engine initialized")

    async def configure(self,
                        retrieval_strategies: List[RetrievalStrategy],
                        reranking_enabled: bool,
                        vector_similarity_threshold: float,
                        max_chunks_per_strategy: int) -> bool:
        """
        Configure the RAG engine.

        Args:
            retrieval_strategies: List of retrieval strategies to use
            reranking_enabled: Whether to enable reranking
            vector_similarity_threshold: Threshold for vector similarity
            max_chunks_per_strategy: Maximum number of chunks per strategy

        Returns:
            bool: Success status
        """
        try:
            self.config = {
                "retrieval_strategies": retrieval_strategies,
                "reranking_enabled": reranking_enabled,
                "vector_similarity_threshold": vector_similarity_threshold,
                "max_chunks_per_strategy": max_chunks_per_strategy
            }

            logger.info(f"RAG Engine configured with: {self.config}")
            return True

        except Exception as e:
            logger.error(f"Error configuring RAG Engine: {str(e)}")
            return False

    async def retrieve(self,
                       query: str,
                       document_types: Optional[List[DocumentType]] = None,
                       tags: Optional[List[str]] = None) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks using configured strategies.

        Args:
            query: User query
            document_types: Optional list of document types to include
            tags: Optional list of tags to filter by

        Returns:
            List of retrieved chunks with scores
        """
        try:
            # Prepare filter
            filter_dict = {}

            if document_types:
                filter_dict["document_type"] = {"$in": [dt.value for dt in document_types]}

            if tags:
                filter_dict["tags"] = {"$in": tags}

            # Apply all retrieval strategies
            all_chunks = []
            for strategy in self.config["retrieval_strategies"]:
                strategy_func = self.retrieval_strategies[strategy]
                chunks = await strategy_func(
                    query=query,
                    filter_dict=filter_dict,
                    top_k=self.config["max_chunks_per_strategy"]
                )
                all_chunks.extend(chunks)

            # Rerank results if enabled
            if self.config["reranking_enabled"] and all_chunks:
                all_chunks = await self._rerank_chunks(query, all_chunks)

            # Remove duplicates and sort by score
            unique_chunks = self._remove_duplicates(all_chunks)
            sorted_chunks = sorted(unique_chunks, key=lambda x: x.score, reverse=True)

            return sorted_chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []

    async def _semantic_retrieval(self,
                                  query: str,
                                  filter_dict: Dict[str, Any],
                                  top_k: int) -> List[RetrievedChunk]:
        """
        Perform semantic retrieval based on vector similarity.

        Args:
            query: User query
            filter_dict: Filters to apply
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with scores
        """
        try:
            # Convert query to embedding
            docs_with_score = self.vector_db.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=filter_dict if filter_dict else None
            )

            # Format results
            results = []
            for doc, score in docs_with_score:
                # Convert score to a 0-1 range (Chroma returns distance, not similarity)
                normalized_score = 1.0 - min(1.0, score)

                # Apply threshold filter
                if normalized_score >= self.config["vector_similarity_threshold"]:
                    retrieved_chunk = RetrievedChunk(
                        chunk=doc,  # Need to convert doc to DocumentChunk in real implementation
                        score=normalized_score,
                        strategy=RetrievalStrategy.SEMANTIC
                    )
                    results.append(retrieved_chunk)

            return results

        except Exception as e:
            logger.error(f"Error in semantic retrieval: {str(e)}")
            return []

    async def _keyword_retrieval(self,
                                 query: str,
                                 filter_dict: Dict[str, Any],
                                 top_k: int) -> List[RetrievedChunk]:
        """
        Perform keyword-based retrieval using BM25.

        Args:
            query: User query
            filter_dict: Filters to apply
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with scores
        """
        try:
            # In a real implementation, you'd use a proper BM25 implementation
            # For simplicity, we'll use a simulated approach here

            # Get all documents (filtered by metadata if needed)
            all_docs = self.vector_db.get(
                filter=filter_dict if filter_dict else None
            )

            # Create a temporary BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = top_k

            # Retrieve documents
            docs = bm25_retriever.get_relevant_documents(query)

            # Format results
            results = []
            for i, doc in enumerate(docs):
                # Simulate a score (decreasing by position)
                score = 1.0 - (i * 0.1)

                retrieved_chunk = RetrievedChunk(
                    chunk=doc,  # Need to convert doc to DocumentChunk in real implementation
                    score=max(0.0, score),
                    strategy=RetrievalStrategy.KEYWORD
                )
                results.append(retrieved_chunk)

            return results

        except Exception as e:
            logger.error(f"Error in keyword retrieval: {str(e)}")
            return []

    async def _hybrid_retrieval(self,
                                query: str,
                                filter_dict: Dict[str, Any],
                                top_k: int) -> List[RetrievedChunk]:
        """
        Perform hybrid retrieval combining semantic and keyword approaches.

        Args:
            query: User query
            filter_dict: Filters to apply
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with scores
        """
        try:
            # Get semantic results
            semantic_results = await self._semantic_retrieval(
                query=query,
                filter_dict=filter_dict,
                top_k=top_k
            )

            # Get keyword results
            keyword_results = await self._keyword_retrieval(
                query=query,
                filter_dict=filter_dict,
                top_k=top_k
            )

            # Combine results
            all_results = semantic_results + keyword_results

            # Remove duplicates and sort by score
            unique_results = self._remove_duplicates(all_results)
            sorted_results = sorted(unique_results, key=lambda x: x.score, reverse=True)

            # Return top_k results
            return sorted_results[:top_k]

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []

    async def _metadata_retrieval(self,
                                  query: str,
                                  filter_dict: Dict[str, Any],
                                  top_k: int) -> List[RetrievedChunk]:
        """
        Perform metadata-based retrieval.

        Args:
            query: User query
            filter_dict: Filters to apply
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with scores
        """
        try:
            # This is a simplified implementation
            # In a real system, you would analyze the query to extract
            # metadata filters (e.g., date ranges, authors, etc.)

            # For now, we'll just use the existing filter and do a basic retrieval
            enhanced_filter = filter_dict.copy() if filter_dict else {}

            # Add any query-based metadata filters here
            # For example, if the query contains "2023 data", add a year filter

            # Get documents using metadata filter
            docs = self.vector_db.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=enhanced_filter if enhanced_filter else None
            )

            # Format results
            results = []
            for doc, score in docs:
                # Convert score to a 0-1 range
                normalized_score = 1.0 - min(1.0, score)

                retrieved_chunk = RetrievedChunk(
                    chunk=doc,  # Need to convert doc to DocumentChunk in real implementation
                    score=normalized_score,
                    strategy=RetrievalStrategy.METADATA
                )
                results.append(retrieved_chunk)

            return results

        except Exception as e:
            logger.error(f"Error in metadata retrieval: {str(e)}")
            return []

    async def _rerank_chunks(self,
                             query: str,
                             chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Rerank chunks using a cross-encoder or LLM-based approach.

        Args:
            query: User query
            chunks: List of retrieved chunks

        Returns:
            Reranked chunks
        """
        try:
            # In a real implementation, you would use a cross-encoder
            # or an LLM-based approach for reranking

            # For simplicity, we'll use the compression retriever as a reranker

            # Convert RetrievedChunk to Document objects
            docs = [chunk.chunk for chunk in chunks]

            # Use the compression retriever to rerank
            reranked_docs = self.compression_retriever.get_relevant_documents(query)

            # Map back to RetrievedChunk with updated scores
            reranked_chunks = []
            for i, doc in enumerate(reranked_docs):
                # Find the original chunk
                original_chunk = next((c for c in chunks if c.chunk.chunk_id == doc.metadata.get("chunk_id")), None)

                if original_chunk:
                    # Create a new chunk with updated score
                    # Score decreases by position (simulating reranking)
                    new_score = 1.0 - (i * 0.05)

                    reranked_chunk = RetrievedChunk(
                        chunk=original_chunk.chunk,
                        score=max(0.0, new_score),
                        strategy=original_chunk.strategy
                    )
                    reranked_chunks.append(reranked_chunk)

            return reranked_chunks

        except Exception as e:
            logger.error(f"Error reranking chunks: {str(e)}")
            return chunks  # Return original chunks if reranking fails

    def _remove_duplicates(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Remove duplicate chunks by chunk_id.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Deduplicated chunks
        """
        seen_chunk_ids = set()
        unique_chunks = []

        for chunk in chunks:
            chunk_id = chunk.chunk.metadata.get("chunk_id")
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                unique_chunks.append(chunk)

        return unique_chunks