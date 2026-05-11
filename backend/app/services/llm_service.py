#здесь будет все миленькое вроде бы
from app.services.ml_engine import generate_medical_report
from app.core.config import get_settings
import asyncio
import logging
import re
from typing import Dict, Optional

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

async def get_structured_answer(patient_ dict, medical_text: str, guideline_paths: Optional[list[str]] = None) -> Dict[str, str]:
    if not guideline_paths:
        gp = getattr(settings, "GUIDELINE_PATHS", None)
        guideline_paths = [gp] if gp and isinstance(gp, str) else (gp or [])
    ml_args = {"query": medical_text.strip(), "patient_history": medical_text.strip(), "patient_data": patient_data, "guideline_paths": guideline_paths}
    try:
        llm_response = await asyncio.to_thread(generate_medical_report, **ml_args)
    except Exception as e:
        logger.error(f"LLM error in get_structured_answer: {e}", exc_info=True)
        raise
      
    raw_report = llm_response.get("report", "")
    if not raw_report:
        return {"diagnosis": "", "clinical_recommendations": ""}

    #Вырезаем только блок [Заключение]
    match = re.search(r'\[Заключение\]\s*(.*?)(?=\[|$)', raw_report, re.DOTALL | re.IGNORECASE)
    conclusion_block = match.group(1).strip() if match else raw_report.strip()

    #Убираем служебные заголовки
    conclusion_block = re.sub(r'Предварительный\s+диагноз[/\\]статус:\s*', '', conclusion_block, flags=re.IGNORECASE)
    conclusion_block = re.sub(r'Рекомендации\s+по\s+тактике:\s*', '', conclusion_block, flags=re.IGNORECASE)
  
    #Разделяем диагноз и рекомендации по началу нумерованного списка
    rec_start = re.search(r'\n\s*\d+\.\s', conclusion_block)
    if rec_start:
        diagnosis = conclusion_block[:rec_start.start()].strip()
        recommendations = conclusion_block[rec_start.start():].strip()
    else:
        diagnosis = conclusion_block
        recommendations = ""

    #Удаляем дисклеймер в конце
    disclaimer = r'(?:^|\n)Заключение\s+носит\s+информационно[ -]аналитический.*?(?:врачом|обследования)\.?'
    diagnosis = re.sub(disclaimer, '', diagnosis, flags=re.IGNORECASE | re.DOTALL).strip()
    recommendations = re.sub(disclaimer, '', recommendations, flags=re.IGNORECASE | re.DOTALL).strip()

    #Удаляем ссылки на литературу
    ref_pattern = r'\[\s*\d+(?:\s*[,\-\s]\s*\d+)*\s*\]'
    diagnosis = re.sub(ref_pattern, '', diagnosis)
    recommendations = re.sub(ref_pattern, '', recommendations)

    #Нормализация пробелов
    diagnosis = re.sub(r'\s+', ' ', diagnosis).strip()
    recommendations = re.sub(r'\n\s*\n', '\n', recommendations).strip()

    return {"diagnosis": diagnosis, "clinical_recommendations": recommendations}
