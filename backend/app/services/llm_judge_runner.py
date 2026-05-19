import asyncio
import logging
from datetime import datetime, timezone
from sqlalchemy import select
from typing import Optional

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.core.enum.call_type import CallType, CallStatus
from app.models.report import Report
from app.models.llm_calls import LLMCall
from app.models.report_templates import ReportTemplate
from app.services.llm_judge import LLMJudge

logger = logging.getLogger(__name__)
settings = get_settings()

def _report_to_trace(report: Report, generation_call: Optional[LLMCall]) -> dict:
    return {
        "measurements": report.measurements,
        "meta": report.meta,
        "llm_response": report.llm_response,
        "llm_call": {
            "input_json": generation_call.input_json if generation_call else None,
            "output_json": generation_call.output_json if generation_call else None,
            "trace_json": generation_call.trace_json if generation_call else None,
        },
    }

async def run_llm_judge_for_report(id_report: str, user_id: int) -> None:
    """Запускает оценку отчёта вызовом LLM, сохраняет результат и запись в БД."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Report).where(Report.id_report == id_report))
        report = result.scalar_one_or_none()

        if not report:
            logger.error("Judge skipped: report %s not found", id_report)
            return

        judge_call = LLMCall(report_id=report.id, user_id=user_id, status=CallStatus.PROCESSING,
                            call_type=CallType.LLM_JUDGE, is_judge=True, provider="vllm", model=settings.VLLM_MODEL,
                            prompt=" ", input_json={"report_id": id_report},
                            started_at=datetime.now(timezone.utc))
        report.judge_enabled = True
        report.judge_status = CallStatus.PROCESSING.value
        db.add(judge_call)
        await db.commit()
        await db.refresh(judge_call)

        try:
            judge = LLMJudge()

            result = await db.execute(
                select(LLMCall)
                .where(
                    LLMCall.report_id == report.id,
                    LLMCall.call_type == CallType.REPORT_GENERATION,
                )
                .order_by(LLMCall.created_at.desc())
            )
            generation_call = result.scalar_one_or_none()

            trace = _report_to_trace(report, generation_call)

            judge_call.prompt = judge.build_prompt(trace)
            judge_call.input_json = trace
            await db.commit()

            judge_result = await asyncio.to_thread(judge.evaluate, trace)

            judge_call.output_json = judge_result
            judge_call.status = CallStatus.COMPLETED
            judge_call.completed_at = datetime.now(timezone.utc)

            report.judge_response = judge_result
            report.judge_status = CallStatus.COMPLETED.value
            report.judge_completed_at = datetime.now(timezone.utc)

        except Exception as e:
            logger.error("LLM judge failed for report %s: %s", id_report, e, exc_info=True)
            report.judge_status = CallStatus.FAILED.value
            report.judge_error = str(e)
            judge_call.status = CallStatus.FAILED
            judge_call.error_message = str(e)
            judge_call.completed_at = datetime.now(timezone.utc)

        await db.commit()