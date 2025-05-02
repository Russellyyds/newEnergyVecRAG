from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from datetime import datetime


class DocumentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"


class RetrievalStrategy(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    METADATA = "metadata"


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    stored_filename: str
    document_type: str
    tags: Optional[str] = None
    upload_timestamp: str


class RagConfig(BaseModel):
    retrieval_strategies: List[RetrievalStrategy] = Field(
        default=[RetrievalStrategy.HYBRID],
        description="Strategies to use for retrieval"
    )
    reranking_enabled: bool = Field(
        default=True,
        description="Whether to rerank retrieved documents"
    )
    vector_similarity_threshold: float = Field(
        default=0.7,
        description="Threshold for vector similarity"
    )
    max_chunks_per_strategy: int = Field(
        default=5,
        description="Maximum number of chunks to retrieve per strategy"
    )


class GenerationConfig(BaseModel):
    max_tokens: int = Field(
        default=1000,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for generation"
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include sources in the generated content"
    )
    format_preferences: Dict[str, Any] = Field(
        default={},
        description="Preferences for output formatting"
    )
    content_type: List[str] = Field(
        default=["text"],
        description="Types of content to generate (text, image)"
    )
    domain_specific_settings: Dict[str, Any] = Field(
        default={},
        description="Domain-specific settings for new energy content"
    )


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query")
    document_types: Optional[List[DocumentType]] = Field(
        default=None,
        description="Types of documents to include in search"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Tags to filter documents by"
    )
    generation_config: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="Configuration for content generation"
    )


class QueryResponse(BaseModel):
    query: str
    text_content: Optional[str] = None
    image_content: Optional[List[Dict[str, str]]] = None
    references: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None


class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    document_type: DocumentType
    chunk_index: int
    page_number: Optional[int] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class DocumentChunk(BaseModel):
    chunk_id: str
    text: Optional[str] = None
    image_url: Optional[str] = None
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float
    strategy: RetrievalStrategy


class ContentGenerationResult(BaseModel):
    text_content: Optional[str] = None
    image_content: Optional[List[Dict[str, str]]] = None
    references: List[Dict[str, Any]]
    metrics: Dict[str, Any]