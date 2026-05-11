#здесь будет все миленькое вроде бы
from app.services.ml_engine import generate_medical_report
from app.core.config import get_settings
import asyncio
import logging
from typing import Optional

settings = get_settings()
logger = logging.getLogger(__name__)

async def process_llm_request(patient_data: dict, medical_text: str,
                              guideline_paths: Optional[list[str]] = None) -> tuple[dict, dict]:
    if not guideline_paths:
        gp = getattr(settings, "GUIDELINE_PATHS", None)
        guideline_paths = [gp] if gp and isinstance(gp, str) else (gp or [])

    ml_args = {
        "query": medical_text.strip(),
        "patient_history": medical_text.strip(),
        "patient_data": patient_data,
        "guideline_paths": guideline_paths,
    }

    try:
        llm_response = await asyncio.to_thread(generate_medical_report, **ml_args)
    except Exception as e:
        logger.error(f"LLM/RAG error in process_llm_request: {e}", exc_info=True)
        raise

    trace_data = {
        "model": settings.VLLM_MODEL,
        "input_keys": list(ml_args.keys())
    }
    return llm_response, trace_data
