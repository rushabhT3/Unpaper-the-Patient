from pydantic import BaseModel, Field, field_validator
from datetime import date
from decimal import Decimal
from typing import Literal, List, Optional

# Document Types as per requirement
DocType = Literal[
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


class PatientInfo(BaseModel):
    name: str
    date_of_birth: Optional[date] = None


# ID Agent Extraction Results
class IDExtraction(BaseModel):
    patient_name: str
    date_of_birth: Optional[date] = None
    id_numbers: List[str] = Field(default_factory=list)
    policy_details: Optional[str] = None


# Discharge Summary Agent Extraction Results
class DischargeExtraction(BaseModel):
    diagnosis: str
    admission_date: Optional[date] = None
    discharge_date: Optional[date] = None
    physician_details: Optional[str] = None


# Itemized Bill Agent Extraction Results
class BillItem(BaseModel):
    description: str
    cost: Decimal


class ItemizedBillExtraction(BaseModel):
    items: List[BillItem]
    total_amount: Decimal


# Aggregated Result
class ClaimProcessResult(BaseModel):
    claim_id: str
    identity: Optional[IDExtraction] = None
    discharge: Optional[DischargeExtraction] = None
    billing: Optional[ItemizedBillExtraction] = None
    page_classifications: List[dict] = Field(
        description="List of {page: int, type: DocType}"
    )
