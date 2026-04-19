# Unpaper the Patient: Claim Processing Pipeline

This project implements an automated Claim Processing Pipeline using **FastAPI** and **LangGraph**, designed to extract structured data from medical claim documents (PDFs), including scanned and image-only files.

## Features

- **Multi-Agent Orchestration**: Powered by LangGraph to coordinate between specialized agents.
- **AI-Powered Segregator**: Classifies pages into 9 distinct document types using Gemini Vision.
- **Specialized Extraction Agents**:
  - **ID Agent**: Extracts patient identity and policy details.
  - **Discharge Agent**: Extracts clinical findings and stay dates.
  - **Itemized Bill Agent**: Extracts line items and calculates total costs.
- **Image-Only Support**: Seamlessly handles scanned PDFs by converting pages to images for Vision-based processing.
- **SOLID Architecture**: Clean separation of concerns between Infrastructure, Domain, and Application layers.

## Tech Stack

- **Framework**: FastAPI
- **Orchestration**: LangGraph
- **LLM**: Google Gemini (3.1 Flash-Lite-Preview)
- **PDF Processing**: PyMuPDF (fitz)
- **Validation**: Pydantic v2

## Setup

1. **Enable .venv in git bash**:
   ```bash
   source .venv/Scripts/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `config/.env` file and add your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Run the Service**:
   ```bash
   uvicorn src.api.main:app --reload
   ```

## API Usage

### Process Claim
- **Endpoint**: `POST /api/process`
- **Payload**:
  - `claim_id`: Unique identifier for the claim.
  - `file`: PDF document.

**Example Request**:
```bash
curl -X POST "https://unpaper-the-patient.onrender.com/api/process?claim_id=CLM123" \
     -F "file=@final_image_protected.pdf"
```

## Workflow Explanation

1. **Segregator Agent**: Analyzes each page of the PDF to determine its type (e.g., identity_document, discharge_summary).
2. **Parallel Extraction**: Relevant pages are routed to their respective agents (ID, Discharge, Bill) based on the classification.
3. **Aggregator**: Collects the outputs from all agents and synthesizes them into a single coherent JSON response.
