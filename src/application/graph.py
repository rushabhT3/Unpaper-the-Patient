from typing import Dict, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from src.domain.models import (
    IDExtraction,
    DischargeExtraction,
    ItemizedBillExtraction,
    ClaimProcessResult,
    DocType,
)
from src.infrastructure.ocr import VisionService
from src.infrastructure.gemini import GeminiService
from pathlib import Path
import asyncio


# Define the State of the Graph
class GraphState(TypedDict):
    pdf_path: Path
    claim_id: str
    page_images: list[bytes]
    classifications: Dict[int, DocType]  # page_index -> type
    id_result: Optional[IDExtraction]
    discharge_result: Optional[DischargeExtraction]
    bill_result: Optional[ItemizedBillExtraction]
    final_output: Optional[ClaimProcessResult]


async def segregation_node(state: GraphState):
    """AI-powered Segregator Agent."""
    ocr = VisionService()
    gemini = GeminiService()

    # Convert PDF to images
    images = await ocr.pdf_to_images(state["pdf_path"])

    # Limit concurrency to avoid rate limits
    sem = asyncio.Semaphore(3)

    async def _classify(img):
        async with sem:
            return await gemini.classify_page(img)

    # Classify each page
    tasks = [_classify(img) for img in images]
    results = await asyncio.gather(*tasks)

    classifications = {i: res for i, res in enumerate(results)}
    return {"page_images": images, "classifications": classifications}


async def id_agent_node(state: GraphState):
    """ID Agent - Extracts identity information."""
    gemini = GeminiService(model_name="gemini-3.1-flash-lite-preview")
    target_pages = [
        i for i, t in state["classifications"].items() if t == "identity_document"
    ]

    if not target_pages:
        return {"id_result": None}

    # Process the first relevant page (or combine if needed, keeping it KISS for now)
    img = state["page_images"][target_pages[0]]
    result = await gemini.extract_structured_data(
        img, IDExtraction, "Extract patient name, DOB, ID numbers, and policy details."
    )
    return {"id_result": result}


async def discharge_agent_node(state: GraphState):
    """Discharge Summary Agent - Extracts clinical details."""
    gemini = GeminiService(model_name="gemini-3.1-flash-lite-preview")
    target_pages = [
        i for i, t in state["classifications"].items() if t == "discharge_summary"
    ]

    if not target_pages:
        return {"discharge_result": None}

    img = state["page_images"][target_pages[0]]
    result = await gemini.extract_structured_data(
        img,
        DischargeExtraction,
        "Extract diagnosis, admission date, discharge date, and physician details.",
    )
    return {"discharge_result": result}


async def itemized_bill_agent_node(state: GraphState):
    """Itemized Bill Agent - Extracts bill data."""
    gemini = GeminiService(model_name="gemini-3.1-flash-lite-preview")
    target_pages = [
        i for i, t in state["classifications"].items() if t == "itemized_bill"
    ]

    if not target_pages:
        return {"bill_result": None}

    img = state["page_images"][target_pages[0]]
    result = await gemini.extract_structured_data(
        img,
        ItemizedBillExtraction,
        "Extract all items with costs and ensure the total_amount is calculated.",
    )
    return {"bill_result": result}


async def aggregator_node(state: GraphState):
    """Aggregator Node - Combines all results."""
    final = ClaimProcessResult(
        claim_id=state["claim_id"],
        identity=state.get("id_result"),
        discharge=state.get("discharge_result"),
        billing=state.get("bill_result"),
        page_classifications=[
            {"page": i + 1, "type": t} for i, t in state["classifications"].items()
        ],
    )
    return {"final_output": final}


def create_claim_pipeline():
    """Builds the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("segregator", segregation_node)
    workflow.add_node("id_agent", id_agent_node)
    workflow.add_node("discharge_agent", discharge_agent_node)
    workflow.add_node("bill_agent", itemized_bill_agent_node)
    workflow.add_node("aggregator", aggregator_node)

    # Add Edges
    workflow.add_edge(START, "segregator")

    # Parallel execution after segregation
    workflow.add_edge("segregator", "id_agent")
    workflow.add_edge("segregator", "discharge_agent")
    workflow.add_edge("segregator", "bill_agent")

    # All agents converge to aggregator
    workflow.add_edge("id_agent", "aggregator")
    workflow.add_edge("discharge_agent", "aggregator")
    workflow.add_edge("bill_agent", "aggregator")

    workflow.add_edge("aggregator", END)

    return workflow.compile()
