from pydantic import BaseModel
from typing import Dict, Any, List

class LLMRequest(BaseModel):
    query: str = "Жалоб на здоровье нет"
    patient_history: str
    patient_data: Dict[str, Any]
    guideline_paths: List[str]

class LLMResponse(BaseModel):
    report: str
    warnings: list[str]
    errors: list[str]