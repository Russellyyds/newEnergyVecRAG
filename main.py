# main.py - Main FastAPI Application
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, List
import uvicorn
import os
import shutil
from datetime import datetime
import uuid

# Import project modules
from app.config import settings
from app.models.models import (
    QueryRequest,
    QueryResponse,
    DocumentUploadResponse,
    RagConfig,
    GenerationConfig
)
from app.services.knowledge_base import KnowledgeBaseService
from app.services.document_processor import DocumentProcessor
from app.services.image_processor import ImageProcessor
from app.services.rag_engine import RagEngine
from app.services.content_generator import ContentGenerator
from app.services.export_service import ExportService
from app.utils.logger import setup_logger

# Setup logger
logger = setup_logger()

# Initialize FastAPI app
app = FastAPI(
    title="New Energy Intelligent Agent",
    description="An intelligent agent for new energy data processing and content generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
knowledge_base_service = KnowledgeBaseService()
document_processor = DocumentProcessor()
image_processor = ImageProcessor()
rag_engine = RagEngine()
content_generator = ContentGenerator()
export_service = ExportService()

# Mount static directory for serving files
os.makedirs("app/static/uploads", exist_ok=True)
os.makedirs("app/static/exports", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Document upload endpoint
@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
        file: UploadFile = File(...),
        document_type: str = Form(...),
        tags: Optional[str] = Form(None)
):
    try:
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = f"app/static/uploads/{unique_filename}"

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process document based on type
        if document_type == "text":
            doc_id = await document_processor.process_document(file_path, tags)
        elif document_type == "image":
            doc_id = await image_processor.process_image(file_path, tags)
        else:
            raise HTTPException(status_code=400, detail="Unsupported document type")

        # Add to knowledge base
        success = await knowledge_base_service.add_document(doc_id, file_path, document_type, tags)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to add document to knowledge base")

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=file.filename,
            stored_filename=unique_filename,
            document_type=document_type,
            tags=tags,
            upload_timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Knowledge base test endpoint
@app.post("/knowledge-base/test")
async def test_knowledge_base(
        query: str = Form(...),
        document_type: Optional[str] = Form(None),
        tags: Optional[str] = Form(None),
        top_k: int = Form(5)
):
    try:
        results = await knowledge_base_service.query(query, document_type, tags, top_k)
        return {"query": query, "results": results}
    except Exception as e:
        logger.error(f"Error querying knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# RAG configuration endpoint
@app.post("/rag/configure")
async def configure_rag(rag_config: RagConfig):
    try:
        success = await rag_engine.configure(
            retrieval_strategies=rag_config.retrieval_strategies,
            reranking_enabled=rag_config.reranking_enabled,
            vector_similarity_threshold=rag_config.vector_similarity_threshold,
            max_chunks_per_strategy=rag_config.max_chunks_per_strategy
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to configure RAG engine")

        return {"status": "success", "config": rag_config.dict()}
    except Exception as e:
        logger.error(f"Error configuring RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Content generation endpoint
@app.post("/generate", response_model=QueryResponse)
async def generate_content(request: QueryRequest):
    try:
        # Retrieve relevant information using RAG
        rag_results = await rag_engine.retrieve(
            query=request.query,
            document_types=request.document_types,
            tags=request.tags
        )

        # Generate content
        content = await content_generator.generate(
            query=request.query,
            rag_results=rag_results,
            generation_config=request.generation_config
        )

        return QueryResponse(
            query=request.query,
            text_content=content.get("text_content"),
            image_content=content.get("image_content"),
            references=content.get("references"),
            metrics=content.get("metrics")
        )
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Export endpoint
@app.post("/export")
async def export_document(
        query_response_id: str = Form(...),
        export_format: str = Form(...),
        template_id: Optional[str] = Form(None)
):
    try:
        # Get query response
        # In a real application, you would retrieve this from a database
        # Here we're just simulating it
        query_response = {"id": query_response_id, "content": "Sample content"}

        # Export document
        export_path = await export_service.export(
            content=query_response,
            export_format=export_format,
            template_id=template_id
        )

        return FileResponse(export_path)
    except Exception as e:
        logger.error(f"Error exporting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)