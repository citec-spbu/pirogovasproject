from enum import Enum

class CallType(str, Enum):
    REPORT_GENERATION = "report_generation"
    LLM_JUDGE = "llm_judge"

class CallStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"