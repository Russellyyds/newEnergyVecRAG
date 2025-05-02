import os
from typing import Optional, Dict, Any, List
import asyncio
import uuid
import logging
from datetime import datetime
import json

from app.config import settings

logger = logging.getLogger(__name__)


class ExportService:
    def __init__(self):
        """Initialize the Export Service."""
        self.export_formats = ["pdf", "html", "markdown", "docx", "csv"]
        self.templates_path = "app/static/templates"
        self.exports_path = "app/static/exports"

        # Ensure export directory exists
        os.makedirs(self.exports_path, exist_ok=True)

        logger.info("Export Service initialized")

    async def export(self,
                     content: Dict[str, Any],
                     export_format: str,
                     template_id: Optional[str] = None) -> str:
        """
        Export content to the specified format.

        Args:
            content: Content to export
            export_format: Format to export to
            template_id: Optional template ID to use

        Returns:
            Path to the exported file
        """
        try:
            # Check if format is supported
            export_format = export_format.lower()
            if export_format not in self.export_formats:
                raise ValueError(f"Unsupported export format: {export_format}")

            # Generate a unique filename
            filename = f"{uuid.uuid4()}.{export_format}"
            export_path = os.path.join(self.exports_path, filename)

            # Export based on format
            if export_format == "pdf":
                await self._export_pdf(content, export_path, template_id)
            elif export_format == "html":
                await self._export_html(content, export_path, template_id)
            elif export_format == "markdown":
                await self._export_markdown(content, export_path)
            elif export_format == "docx":
                await self._export_docx(content, export_path, template_id)
            elif export_format == "csv":
                await self._export_csv(content, export_path)

            logger.info(f"Exported content to {export_path}")
            return export_path

        except Exception as e:
            logger.error(f"Error exporting content: {str(e)}")
            raise

    async def _export_pdf(self,
                          content: Dict[str, Any],
                          export_path: str,
                          template_id: Optional[str] = None) -> None:
        """
        Export content to PDF.

        Args:
            content: Content to export
            export_path: Path to export to
            template_id: Optional template ID to use
        """
        try:
            # In a real implementation, you would:
            # 1. Use a library like ReportLab, WeasyPrint, or wkhtmltopdf
            # 2. Apply the template if provided
            # 3. Generate the PDF

            # For now, we'll create a simple placeholder file
            with open(export_path, 'w') as f:
                f.write(f"Simulated PDF export\n\n")
                f.write(f"Content: {json.dumps(content, indent=2)}\n")
                f.write(f"Template: {template_id or 'None'}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")

            logger.info(f"Exported PDF to {export_path}")

        except Exception as e:
            logger.error(f"Error exporting PDF: {str(e)}")
            raise

    async def _export_html(self,
                           content: Dict[str, Any],
                           export_path: str,
                           template_id: Optional[str] = None) -> None:
        """
        Export content to HTML.

        Args:
            content: Content to export
            export_path: Path to export to
            template_id: Optional template ID to use
        """
        try:
            # Get template content if provided
            template_html = ""
            if template_id:
                template_path = os.path.join(self.templates_path, f"{template_id}.html")
                if os.path.exists(template_path):
                    with open(template_path, 'r') as f:
                        template_html = f.read()

            # If no template or template not found, use a default
            if not template_html:
                template_html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>New Energy Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        h1 { color: #0066cc; }
                        .content { margin-top: 20px; }
                        .references { margin-top: 30px; font-size: 0.9em; }
                        .metrics { margin-top: 20px; background-color: #f0f0f0; padding: 10px; }
                    </style>
                </head>
                <body>
                    <h1>New Energy Report</h1>
                    <div class="content">{{CONTENT}}</div>
                    <div class="references">{{REFERENCES}}</div>
                    <div class="metrics">{{METRICS}}</div>
                    <footer>Generated on {{TIMESTAMP}}</footer>
                </body>
                </html>
                """

            # Extract content
            text_content = content.get("text_content", "")
            references = content.get("references", [])
            metrics = content.get("metrics", {})

            # Format references
            references_html = "<h2>References</h2><ul>"
            for ref in references:
                references_html += f"<li>{ref.get('source', 'Unknown source')}</li>"
            references_html += "</ul>"

            # Format metrics
            metrics_html = "<h2>Metrics</h2><ul>"
            for key, value in metrics.items():
                metrics_html += f"<li>{key}: {value}</li>"
            metrics_html += "</ul>"

            # Replace placeholders
            html_content = template_html
            html_content = html_content.replace("{{CONTENT}}", text_content)
            html_content = html_content.replace("{{REFERENCES}}", references_html)
            html_content = html_content.replace("{{METRICS}}", metrics_html)
            html_content = html_content.replace("{{TIMESTAMP}}", datetime.now().isoformat())

            # Write to file
            with open(export_path, 'w') as f:
                f.write(html_content)

            logger.info(f"Exported HTML to {export_path}")

        except Exception as e:
            logger.error(f"Error exporting HTML: {str(e)}")
            raise

    async def _export_markdown(self,
                               content: Dict[str, Any],
                               export_path: str) -> None:
        """
        Export content to Markdown.

        Args:
            content: Content to export
            export_path: Path to export to
        """
        try:
            # Extract content
            text_content = content.get("text_content", "")
            references = content.get("references", [])
            metrics = content.get("metrics", {})

            # Create markdown content
            markdown = f"# New Energy Report\n\n"
            markdown += text_content + "\n\n"

            # Add references
            markdown += "## References\n\n"
            for ref in references:
                markdown += f"- {ref.get('source', 'Unknown source')}\n"
            markdown += "\n"

            # Add metrics
            markdown += "## Metrics\n\n"
            for key, value in metrics.items():
                markdown += f"- **{key}**: {value}\n"
            markdown += "\n"

            # Add timestamp
            markdown += f"Generated on {datetime.now().isoformat()}\n"

            # Write to file
            with open(export_path, 'w') as f:
                f.write(markdown)

            logger.info(f"Exported Markdown to {export_path}")

        except Exception as e:
            logger.error(f"Error exporting Markdown: {str(e)}")
            raise

    async def _export_docx(self,
                           content: Dict[str, Any],
                           export_path: str,
                           template_id: Optional[str] = None) -> None:
        """
        Export content to DOCX.

        Args:
            content: Content to export
            export_path: Path to export to
            template_id: Optional template ID to use
        """
        try:
            # In a real implementation, you would:
            # 1. Use a library like python-docx
            # 2. Apply the template if provided
            # 3. Generate the DOCX

            # For now, we'll create a simple placeholder file
            with open(export_path, 'w') as f:
                f.write(f"Simulated DOCX export\n\n")
                f.write(f"Content: {json.dumps(content, indent=2)}\n")
                f.write(f"Template: {template_id or 'None'}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")

            logger.info(f"Exported DOCX to {export_path}")

        except Exception as e:
            logger.error(f"Error exporting DOCX: {str(e)}")
            raise

    async def _export_csv(self,
                          content: Dict[str, Any],
                          export_path: str) -> None:
        """
        Export content to CSV.

        Args:
            content: Content to export
            export_path: Path to export to
        """
        try:
            # In a real implementation, you would:
            # 1. Structure the data appropriately for CSV
            # 2. Use the csv module to write the file

            # For now, we'll create a simple placeholder file
            with open(export_path, 'w') as f:
                f.write("field1,field2,field3\n")
                f.write(f"data1,data2,data3\n")
                f.write(f"Generated,{datetime.now().isoformat()},\n")

            logger.info(f"Exported CSV to {export_path}")

        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
            raise

    async def list_templates(self) -> List[Dict[str, Any]]:
        """
        List available templates.

        Returns:
            List of templates
        """
        try:
            templates = []

            # Ensure templates directory exists
            os.makedirs(self.templates_path, exist_ok=True)

            # List templates
            for filename in os.listdir(self.templates_path):
                if os.path.isfile(os.path.join(self.templates_path, filename)):
                    name, ext = os.path.splitext(filename)
                    templates.append({
                        "id": name,
                        "name": name.replace("_", " ").title(),
                        "format": ext[1:],  # Remove the dot
                        "path": os.path.join(self.templates_path, filename)
                    })

            return templates

        except Exception as e:
            logger.error(f"Error listing templates: {str(e)}")
            return []