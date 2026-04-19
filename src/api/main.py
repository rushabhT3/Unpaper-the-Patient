from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import tempfile, uuid
from src.core.config import settings
from src.application.factory import create_pipeline

app = FastAPI(title=settings.app_name)


@app.on_event("startup")
async def startup():
    pass


@app.post("/api/process")
async def process(
    claim_id: str,
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF only")

    # Save temp file
    tmp = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.pdf"
    content = await file.read()
    tmp.write_bytes(content)

    try:
        pipe = create_pipeline()

        # Run workflow
        result = await pipe.ainvoke(
            {
                "pdf_path": tmp,
                "claim_id": claim_id,
                "page_images": [],
                "classifications": {},
                "id_result": None,
                "discharge_result": None,
                "bill_result": None,
                "final_output": None,
            }
        )

        final_data = result.get("final_output")
        if not final_data:
            return {"error": "Processing failed"}

        return final_data.model_dump()

    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        tmp.unlink(missing_ok=True)
