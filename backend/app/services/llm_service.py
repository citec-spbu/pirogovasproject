#здесь будет все миленькое вроде бы
from app.services.ml_engine import generate_medical_report
from app.core.config import settings
import asyncio
from typing import Optional

async def process_llm_request(patient_data: dict, medical_text: str,
                              guideline_paths: Optional[list[str]] = None) -> tuple[dict, dict]:
    if not guideline_paths:
        gp = getattr(settings, "GUIDELINE_PATHS", None)
        guideline_paths = [gp] if gp and isinstance(gp, str) else (gp or [])

    ml_args = {
        "query": "",
        "patient_history": medical_text.strip(),
        "patient_data": patient_data,
        "guideline_paths": guideline_paths,
    }
    llm_response = await asyncio.to_thread(generate_medical_report, **ml_args)
    trace_data = {
        "model": settings.VLLM_MODEL,
        "input_keys": list(ml_args.keys())
    }
    return llm_response, trace_data