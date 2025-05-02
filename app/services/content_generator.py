from typing import List, Dict, Any, Optional
import asyncio
import logging
import uuid
from datetime import datetime

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from app.config.settings import settings
from app.models.models import RetrievedChunk, GenerationConfig, ContentGenerationResult

logger = logging.getLogger(__name__)


class ContentGenerator:
    def __init__(self):
        """Initialize the Content Generator."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=0.7
        )

        # Initialize output parser for metrics
        response_schemas = [
            ResponseSchema(name="accuracy", description="Accuracy score from 0-100"),
            ResponseSchema(name="relevance", description="Relevance score from 0-100"),
            ResponseSchema(name="completeness", description="Completeness score from 0-100"),
            ResponseSchema(name="technical_terms", description="List of technical terms used"),
            ResponseSchema(name="time_sensitivity", description="Assessment of time sensitivity")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        logger.info("Content Generator initialized")

    async def generate(self,
                       query: str,
                       rag_results: List[RetrievedChunk],
                       generation_config: Optional[GenerationConfig] = None) -> Dict[str, Any]:
        """
        Generate content based on RAG results.

        Args:
            query: User query
            rag_results: Retrieval results
            generation_config: Configuration for generation

        Returns:
            Generated content with metrics
        """
        try:
            if not generation_config:
                generation_config = GenerationConfig()

            # Extract text from retrieval results
            context_texts = []
            references = []

            for i, chunk in enumerate(rag_results):
                # Extract text content
                content = chunk.chunk.text or "No text content available"

                # Add to context
                context_texts.append(f"[Document {i + 1}]: {content}")

                # Add to references
                references.append({
                    "document_id": chunk.chunk.metadata.document_id,
                    "chunk_id": chunk.chunk.metadata.chunk_id,
                    "score": chunk.score,
                    "strategy": chunk.strategy,
                    "source": chunk.chunk.metadata.source or "Unknown"
                })

            # Combine context texts
            context = "\n\n".join(context_texts)

            # Generate text content
            text_content = await self._generate_text(
                query=query,
                context=context,
                generation_config=generation_config
            )

            # Generate image content if needed
            image_content = None
            if "image" in generation_config.content_type:
                image_content = await self._generate_image_content(
                    query=query,
                    context=context,
                    generation_config=generation_config
                )

            # Calculate metrics
            metrics = await self._calculate_metrics(
                query=query,
                generated_text=text_content,
                context=context,
                generation_config=generation_config
            )

            # Create result
            result = {
                "text_content": text_content,
                "image_content": image_content,
                "references": references,
                "metrics": metrics
            }

            return result

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return {
                "text_content": f"Error generating content: {str(e)}",
                "image_content": None,
                "references": [],
                "metrics": {}
            }

    async def _generate_text(self,
                             query: str,
                             context: str,
                             generation_config: GenerationConfig) -> str:
        """
        Generate text content based on query and context.

        Args:
            query: User query
            context: Retrieved context
            generation_config: Configuration for generation

        Returns:
            Generated text
        """
        try:
            # Define system message
            system_template = """
            You are an expert in new energy technologies and data analysis. Your task is to generate
            high-quality content based on the provided context and user query.

            Follow these guidelines:
            1. Be factual and accurate, basing your response on the provided context
            2. Use technical terms correctly and consistently
            3. Maintain a professional tone
            4. Be concise but comprehensive
            5. Focus on addressing the user's query directly
            6. Cite sources when making specific claims

            When using domain-specific settings:
            - Ensure technical terminology is accurate
            - Include relevant data points when available
            - Consider the time sensitivity of information
            - Highlight innovative approaches in the field of new energy

            The provided context contains information retrieved from a knowledge base.
            Use this information to inform your response.
            """

            # Apply domain-specific settings if provided
            domain_settings = generation_config.domain_specific_settings
            if domain_settings:
                for key, value in domain_settings.items():
                    system_template += f"\n- For {key}: {value}"

            # Define human message
            human_template = """
            Query: {query}

            Context:
            {context}

            Please generate a response addressing the query based on the provided context.
            """

            # Create messages
            messages = [
                SystemMessage(content=system_template),
                HumanMessage(content=human_template.format(query=query, context=context))
            ]

            # Generate text
            response = self.llm.predict_messages(
                messages,
                max_tokens=generation_config.max_tokens,
                temperature=generation_config.temperature
            )

            return response.content

        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error generating text: {str(e)}"

    async def _generate_image_content(self,
                                      query: str,
                                      context: str,
                                      generation_config: GenerationConfig) -> List[Dict[str, str]]:
        """
        Generate image content or image descriptions.

        Args:
            query: User query
            context: Retrieved context
            generation_config: Configuration for generation

        Returns:
            List of image descriptions or placeholders
        """
        try:
            # In a real implementation, you would use an image generation model
            # like DALL-E or Stable Diffusion

            # For now, we'll just generate image descriptions

            # Define prompt for image description
            prompt = f"""
            Based on the following query and context, generate a brief description of an image that would be
            relevant and informative.

            Query: {query}

            Context excerpt: {context[:500]}...

            Generate a concise image description for visualization.
            """

            # Generate description
            messages = [
                SystemMessage(
                    content="You are an expert in creating concise visual descriptions for technical content."),
                HumanMessage(content=prompt)
            ]

            response = self.llm.predict_messages(messages)

            # Create a placeholder image content
            image_content = [{
                "id": str(uuid.uuid4()),
                "description": response.content,
                "url": None,  # Would be filled with actual image URL in real implementation
                "created_at": datetime.now().isoformat()
            }]

            return image_content

        except Exception as e:
            logger.error(f"Error generating image content: {str(e)}")
            return []

    async def _calculate_metrics(self,
                                 query: str,
                                 generated_text: str,
                                 context: str,
                                 generation_config: GenerationConfig) -> Dict[str, Any]:
        """
        Calculate quality metrics for the generated content.

        Args:
            query: User query
            generated_text: Generated text content
            context: Retrieved context
            generation_config: Configuration for generation

        Returns:
            Dictionary of metrics
        """
        try:
            # Define prompt for metrics calculation
            prompt = f"""
            Evaluate the following generated text based on the user query and provided context.

            Query: {query}

            Generated text:
            {generated_text}

            Context excerpt:
            {context[:1000]}...

            Evaluate the content on the following metrics:
            1. Accuracy (0-100): How factually accurate is the content compared to the context?
            2. Relevance (0-100): How relevant is the content to the user's query?
            3. Completeness (0-100): How completely does the content address the user's query?
            4. Technical terms: List the technical terms used in the content.
            5. Time sensitivity: Assessment of how time-sensitive the information is.

            {self.output_parser.get_format_instructions()}
            """

            # Generate metrics
            messages = [
                SystemMessage(content="You are an expert evaluator of content quality for technical documents."),
                HumanMessage(content=prompt)
            ]

            response = self.llm.predict_messages(messages)

            # Parse metrics
            metrics = self.output_parser.parse(response.content)

            # Add timestamp
            metrics["evaluation_timestamp"] = datetime.now().isoformat()

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                "accuracy": 0,
                "relevance": 0,
                "completeness": 0,
                "technical_terms": [],
                "time_sensitivity": "Error in evaluation",
                "evaluation_timestamp": datetime.now().isoformat()
            }

    async def update_generation_settings(self,
                                         model_name: str,
                                         temperature: float) -> bool:
        """
        Update the generation settings.

        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for generation

        Returns:
            bool: Success status
        """
        try:
            # Update LLM
            self.llm = ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model_name=model_name,
                temperature=temperature
            )

            logger.info(f"Updated generation settings: model={model_name}, temperature={temperature}")
            return True

        except Exception as e:
            logger.error(f"Error updating generation settings: {str(e)}")
            return False