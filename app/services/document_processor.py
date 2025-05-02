import os
from typing import Optional, List, Dict, Any
import asyncio
import uuid
import logging
from datetime import datetime

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config.settings import settings
from app.models.models import DocumentType

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        """Initialize the Document Processor."""
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        # Map file extensions to document types
        self.extension_to_type = {
            '.txt': DocumentType.TEXT,
            '.pdf': DocumentType.PDF,
            '.csv': DocumentType.TEXT,
            '.xlsx': DocumentType.TEXT,
            '.xls': DocumentType.TEXT,
            '.docx': DocumentType.TEXT,
            '.json': DocumentType.TEXT,
            '.md': DocumentType.TEXT
        }

        # Map document types to loaders
        self.document_loaders = {
            DocumentType.TEXT: TextLoader,
            DocumentType.PDF: PyPDFLoader,
            '.csv': CSVLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader
        }

        logger.info("Document Processor initialized")

    async def process_document(self,
                               file_path: str,
                               tags: Optional[str] = None) -> str:
        """
        Process a document file.

        Args:
            file_path: Path to the document file
            tags: Optional tags for the document

        Returns:
            Document ID
        """
        try:
            # Generate a document ID
            doc_id = str(uuid.uuid4())

            # Get file extension
            _, file_extension = os.path.splitext(file_path)
            file_extension = file_extension.lower()

            # Determine document type
            doc_type = self.extension_to_type.get(file_extension, DocumentType.TEXT)

            # Process document based on type
            if doc_type == DocumentType.PDF:
                await self._process_pdf(doc_id, file_path, tags)
            elif file_extension == '.csv':
                await self._process_csv(doc_id, file_path, tags)
            elif file_extension in ['.xlsx', '.xls']:
                await self._process_excel(doc_id, file_path, tags)
            else:
                await self._process_text(doc_id, file_path, tags)

            logger.info(f"Processed document {doc_id} of type {doc_type}")
            return doc_id

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    async def _process_text(self,
                            doc_id: str,
                            file_path: str,
                            tags: Optional[str] = None) -> None:
        """
        Process a text document.

        Args:
            doc_id: Document ID
            file_path: Path to the document file
            tags: Optional tags for the document
        """
        try:
            # Load document
            loader = TextLoader(file_path)
            documents = loader.load()

            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            # Process chunks (in a real implementation, you might store these)
            for i, chunk in enumerate(chunks):
                # Add metadata
                chunk.metadata.update({
                    "chunk_id": f"{doc_id}-chunk-{i}",
                    "document_id": doc_id,
                    "document_type": DocumentType.TEXT,
                    "chunk_index": i,
                    "tags": tags,
                    "source": file_path,
                    "created_at": datetime.now().isoformat()
                })

            # In a real implementation, you would save these chunks
            # or pass them to a database

            logger.info(f"Processed text document {doc_id} with {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error processing text document: {str(e)}")
            raise

    async def _process_pdf(self,
                           doc_id: str,
                           file_path: str,
                           tags: Optional[str] = None) -> None:
        """
        Process a PDF document.

        Args:
            doc_id: Document ID
            file_path: Path to the document file
            tags: Optional tags for the document
        """
        try:
            # Load document
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            # Process chunks
            for i, chunk in enumerate(chunks):
                # Add metadata
                chunk.metadata.update({
                    "chunk_id": f"{doc_id}-chunk-{i}",
                    "document_id": doc_id,
                    "document_type": DocumentType.PDF,
                    "chunk_index": i,
                    "page_number": chunk.metadata.get("page", 0),
                    "tags": tags,
                    "source": file_path,
                    "created_at": datetime.now().isoformat()
                })

            logger.info(f"Processed PDF document {doc_id} with {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error processing PDF document: {str(e)}")
            raise

    async def _process_csv(self,
                           doc_id: str,
                           file_path: str,
                           tags: Optional[str] = None) -> None:
        """
        Process a CSV document.

        Args:
            doc_id: Document ID
            file_path: Path to the document file
            tags: Optional tags for the document
        """
        try:
            # Load document
            loader = CSVLoader(file_path)
            documents = loader.load()

            # Split into chunks (rows or groups of rows)
            chunks = self.text_splitter.split_documents(documents)

            # Process chunks
            for i, chunk in enumerate(chunks):
                # Add metadata
                chunk.metadata.update({
                    "chunk_id": f"{doc_id}-chunk-{i}",
                    "document_id": doc_id,
                    "document_type": DocumentType.TEXT,
                    "chunk_index": i,
                    "tags": tags,
                    "source": file_path,
                    "created_at": datetime.now().isoformat()
                })

            logger.info(f"Processed CSV document {doc_id} with {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error processing CSV document: {str(e)}")
            raise

    async def _process_excel(self,
                             doc_id: str,
                             file_path: str,
                             tags: Optional[str] = None) -> None:
        """
        Process an Excel document.

        Args:
            doc_id: Document ID
            file_path: Path to the document file
            tags: Optional tags for the document
        """
        try:
            # Load document
            loader = UnstructuredExcelLoader(file_path)
            documents = loader.load()

            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            # Process chunks
            for i, chunk in enumerate(chunks):
                # Add metadata
                chunk.metadata.update({
                    "chunk_id": f"{doc_id}-chunk-{i}",
                    "document_id": doc_id,
                    "document_type": DocumentType.TEXT,
                    "chunk_index": i,
                    "tags": tags,
                    "source": file_path,
                    "created_at": datetime.now().isoformat()
                })

            logger.info(f"Processed Excel document {doc_id} with {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error processing Excel document: {str(e)}")
            raise

    async def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a document.

        Args:
            file_path: Path to the document file

        Returns:
            Document metadata
        """
        try:
            # Get file extension
            _, file_extension = os.path.splitext(file_path)
            file_extension = file_extension.lower()

            # Determine document type
            doc_type = self.extension_to_type.get(file_extension, DocumentType.TEXT)

            # Basic metadata
            metadata = {
                "filename": os.path.basename(file_path),
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "file_extension": file_extension,
                "document_type": doc_type,
                "created_at": datetime.now().isoformat()
            }

            # Document-specific metadata extraction would go here
            # For example, extracting author, title, creation date from PDF

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {"error": str(e)}