from typing import Type, TypeVar
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.core.config import settings
import base64

T = TypeVar("T", bound=BaseModel)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from google.api_core.exceptions import ResourceExhausted


class GeminiService:
    """Service to interact with Google Gemini models."""

    def __init__(self, model_name: str = "gemini-3.1-flash-lite-preview"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.gemini_api_key,
            temperature=0,
            max_retries=3,  # LangChain retries
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type(ResourceExhausted),
    )
    async def classify_page(self, image_bytes: bytes) -> str:
        """
        Classifies a single page image into one of the 9 DocTypes.
        """
        prompt = """
        You are an expert medical document classifier. 
        Analyze the provided image of a document page and classify it into EXACTLY ONE of the following types:
        - claim_forms
        - cheque_or_bank_details
        - identity_document
        - itemized_bill
        - discharge_summary
        - prescription
        - investigation_report
        - cash_receipt
        - other

        Return ONLY the classification name.
        """

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ]
        )

        response = await self.llm.ainvoke([message])
        content = response.content

        if isinstance(content, list):
            # Extract text from list of parts if necessary
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                else:
                    parts.append(str(part))
            content = "".join(parts)

        if not isinstance(content, str):
            content = str(content)

        raw_type = content.strip().lower()

        # Allowed types for validation
        allowed = [
            "claim_forms",
            "cheque_or_bank_details",
            "identity_document",
            "itemized_bill",
            "discharge_summary",
            "prescription",
            "investigation_report",
            "cash_receipt",
            "other",
        ]

        for t in allowed:
            if t in raw_type:
                return t

        return "other"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type(ResourceExhausted),
    )
    async def extract_structured_data(
        self, image_bytes: bytes, schema: Type[T], prompt_context: str
    ) -> T:
        """
        Extracts structured data from a document image based on a Pydantic schema.
        """
        llm_with_structured = self.llm.with_structured_output(schema)

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"Extract information from this document page. {prompt_context}",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ]
        )

        return await llm_with_structured.ainvoke([message])
