import os
from typing import List, Dict, Any, Optional, Union
import uuid
import asyncio
from datetime import datetime
import logging

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.config.settings import settings
from app.models.models import DocumentChunk, ChunkMetadata, DocumentType, RetrievalStrategy

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    def __init__(self):
        """Initialize the Knowledge Base Service with a vector database."""
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY
        )

        # Initialize Chroma DB
        self.vector_db = Chroma(
            persist_directory=settings.VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name="new_energy_knowledge_base"
        )

        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        logger.info("Knowledge Base Service initialized")

    async def add_document(self,
                           doc_id: str,
                           file_path: str,
                           document_type: str,
                           tags: Optional[str] = None) -> bool:
        """
        Add a document to the knowledge base.

        Args:
            doc_id: Unique identifier for the document
            file_path: Path to the document file
            document_type: Type of document (text, image)
            tags: Optional tags for the document

        Returns:
            bool: Success status
        """
        try:
            # Convert tags string to list if provided
            tag_list = tags.split(',') if tags else []

            # Process document based on type
            if document_type == DocumentType.TEXT or document_type == DocumentType.PDF:
                return await self._add_text_document(doc_id, file_path, document_type, tag_list)
            elif document_type == DocumentType.IMAGE:
                return await self._add_image_document(doc_id, file_path, tag_list)
            else:
                logger.error(f"Unsupported document type: {document_type}")
                return False

        except Exception as e:
            logger.error(f"Error adding document to knowledge base: {str(e)}")
            return False

    async def _add_text_document(self,
                                 doc_id: str,
                                 file_path: str,
                                 document_type: str,
                                 tags: List[str]) -> bool:
        """
        Add a text document to the knowledge base.

        Args:
            doc_id: Unique identifier for the document
            file_path: Path to the document file
            document_type: Type of document (text, pdf)
            tags: Tags for the document

        Returns:
            bool: Success status
        """
        try:
            # Read the document
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Split text into chunks
            chunks = self.text_splitter.split_text(text)

            # Create Document objects for each chunk
            documents = []
            for i, chunk_text in enumerate(chunks):
                # Create metadata for the chunk
                metadata = {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": doc_id,
                    "document_type": document_type,
                    "chunk_index": i,
                    "file_path": file_path,
                    "tags": tags,
                    "created_at": datetime.now().isoformat()
                }

                # Create Document object
                doc = Document(
                    page_content=chunk_text,
                    metadata=metadata
                )

                documents.append(doc)

            # Add documents to vector database
            self.vector_db.add_documents(documents)

            # Persist changes
            self.vector_db.persist()

            logger.info(f"Added text document {doc_id} with {len(chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error adding text document: {str(e)}")
            return False

    async def _add_image_document(self,
                                  doc_id: str,
                                  file_path: str,
                                  tags: List[str]) -> bool:
        """
        Add an image document to the knowledge base.

        Args:
            doc_id: Unique identifier for the document
            file_path: Path to the image file
            tags: Tags for the document

        Returns:
            bool: Success status
        """
        try:
            # For images, we'll use a multimodal model to extract text and generate embeddings
            # This is simulated here - in a real application, you would use a model like CLIP

            # Simulate image embedding and text extraction
            # In a real system, you would:
            # 1. Extract text using OCR if applicable
            # 2. Generate image embeddings using a multimodal model
            # 3. Store both in your vector database

            # Create metadata for the image
            metadata = {
                "chunk_id": str(uuid.uuid4()),
                "document_id": doc_id,
                "document_type": DocumentType.IMAGE,
                "chunk_index": 0,
                "file_path": file_path,
                "tags": tags,
                "created_at": datetime.now().isoformat()
            }

            # For demonstration purposes, we'll create a placeholder document
            # In a real application, you'd extract actual text and features
            doc = Document(
                page_content=f"Image content from {os.path.basename(file_path)}",
                metadata=metadata
            )

            # Add document to vector database
            self.vector_db.add_documents([doc])

            # Persist changes
            self.vector_db.persist()

            logger.info(f"Added image document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding image document: {str(e)}")
            return False

    async def query(self,
                    query: str,
                    document_type: Optional[str] = None,
                    tags: Optional[str] = None,
                    top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the knowledge base.

        Args:
            query: Query string
            document_type: Optional filter by document type
            tags: Optional filter by tags
            top_k: Number of results to return

        Returns:
            List of matching documents
        """
        try:
            # Prepare filter based on document type and tags
            filter_dict = {}

            if document_type:
                filter_dict["document_type"] = document_type

            if tags:
                tag_list = tags.split(',')
                # This is a simplification - in a real system, you would need
                # to handle tag matching differently depending on your vector DB
                filter_dict["tags"] = {"$in": tag_list}

            # Perform vector similarity search
            docs = self.vector_db.similarity_search(
                query=query,
                k=top_k,
                filter=filter_dict if filter_dict else None
            )

            # Format results
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    # In a real system, you would include the similarity score
                    "score": 0.0  # Placeholder
                })

            return results

        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the knowledge base.

        Args:
            document_id: ID of the document to delete

        Returns:
            bool: Success status
        """
        try:
            # Delete documents with matching document_id
            self.vector_db.delete(
                filter={"document_id": document_id}
            )

            # Persist changes
            self.vector_db.persist()

            logger.info(f"Deleted document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    async def update_vector_settings(self,
                                     similarity_threshold: float) -> bool:
        """
        Update vector database settings.

        Args:
            similarity_threshold: Threshold for similarity matching

        Returns:
            bool: Success status
        """
        try:
            # In a real application, you would update your vector DB settings here
            # For Chroma, this might involve recreating the collection or
            # updating parameters

            logger.info(f"Updated vector settings with threshold {similarity_threshold}")
            return True

        except Exception as e:
            logger.error(f"Error updating vector settings: {str(e)}")
            return False