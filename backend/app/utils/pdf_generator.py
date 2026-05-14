import subproccess
import tempfile
from pathlib import Path

def generate_pdf_from_html(html_content: str) -> bytes:

    if not html_content:
        raise ValueError("HTML content is empty")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(tem_dir)
        html_path = temp_path / "report.html"
        pdf_path = temp_path / "report.pdf"

        html_path.write_text(html_content, encoding="utf-8")

        command = [
            "pandoc",
            str(html_path),
            "-o",
            str(pdf_path),
            "--pdf-engine=wkhtmltopdf",
            "--metadata",
            "title=CT report",
        ]

        result = subproccess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Pandoc failed with code {result.returncode}: {result.stderr}")
        
        if not pdf_path.exists():
            raise RuntimeError("Pandoc did not create PDF file")
    
    return pdf_path.read_bytes()