from enum import Enum

class ClinicalProtocolStatus(str, Enum):
    UPLOADED = "uploaded"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
    ARCHIVED = "archived"
