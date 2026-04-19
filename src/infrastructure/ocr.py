import fitz
from pathlib import Path


class VisionService:
    """Service for handling PDF to Image conversion and Visual OCR logic."""

    async def pdf_to_images(self, pdf_path: Path) -> list[bytes]:
        """
        Converts each page of a PDF into an image (bytes).
        Uses PyMuPDF (fitz) for high-quality rendering.
        """
        images = []
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # High resolution (300 DPI)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            images.append(img_data)

        doc.close()
        return images
