from typing import List, Dict, Any
from app.core.config import settings

class DocumentChunker:
    def __init__(self, chunk_size: int = None, overlap: int = None, separators: List[str] = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP
        self.separators = separators or settings.CHUNK_SEPARATORS.split(",")

    def recursive_chunk_text(self, text: str) -> List[str]:
        #Рекурсивное чанкование с приоритетом сепараторов
        raw_chunks = []
        def _split_recursive(current_text: str, sep_idx: int):
            if len(current_text) <= self.chunk_size:
                if current_text.strip():
                    raw_chunks.append(current_text.strip())
                return

            sep = self.separators[sep_idx] if sep_idx < len(self.separators) else ""
            parts = current_text.split(sep) if sep else list(current_text)

            merged = ""
            for part in parts:
                if len(merged) + len(sep) + len(part) > self.chunk_size and merged:
                    _split_recursive(merged, sep_idx + 1)
                    merged = part
                else:
                    merged = merged + sep + part if merged else part

            if merged:
                _split_recursive(merged, sep_idx + 1)

        _split_recursive(text.strip(), 0)

        # Применяем overlap
        final_chunks = []
        for i, chunk in enumerate(raw_chunks):
            if i > 0 and len(raw_chunks[i - 1]) > self.overlap:
                prefix = raw_chunks[i - 1][-self.overlap:]
                chunk = prefix + chunk if prefix[-1].isalnum() else prefix + " " + chunk
            final_chunks.append(chunk)

        return final_chunks

    def build_chunks(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for doc in docs:
            pieces = self.recursive_chunk_text(doc["text"])
            for i, piece in enumerate(pieces):
                result.append({
                    "source": doc["source"],
                    "chunk_id": i,
                    "text": piece,
                })
        return result
