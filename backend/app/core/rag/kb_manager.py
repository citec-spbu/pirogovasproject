import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from app.core.config import get_settings
from .chunker import DocumentChunker
from .embedder import EmbeddingService
from .vector_store import VectorIndexManager
from .bm25_index import BM25Manager
from .graph_builder import KnowledgeGraphBuilder

settings = get_settings()
logger = logging.getLogger(__name__)

def read_documents(folder: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if not folder.exists() or not folder.is_dir():
        return docs

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        text = ""
        try:
            if suffix == ".txt":
                text = file_path.read_text(encoding="utf-8").strip()
            elif suffix == ".pdf":
                reader = PdfReader(str(file_path))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(p for p in pages if p.strip()).strip()
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
            continue
        if text:
            docs.append({"source": file_path.name, "text": text})
    return docs

#сборка, загрузка, кэширование
class KBOrchestrator:
    def __init__(self, cache_root: str = None):
        self.cache_root = Path(cache_root or settings.KB_CACHE_ROOT)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.chunker = DocumentChunker()
        self.embedder = EmbeddingService()
        self.vector_store = VectorIndexManager(dim=self.embedder.dim)
        self.bm25_manager = BM25Manager()
        self.graph_builder = KnowledgeGraphBuilder()
        self._kb_cache: Dict[str, Dict[str, Any]] = {}

    def _get_cache_prefix(self, folder_path: str, use_bm25: bool = False) -> Path:
        import hashlib
        base_dir = Path(settings.KB_CACHE_BM25_ROOT if use_bm25 else settings.KB_CACHE_ROOT)
        resolved_path = Path(folder_path).resolve()
        path_hash = hashlib.sha256(str(resolved_path).encode()).hexdigest()[:8]
        bm25_suffix = "_bm25" if use_bm25 else ""
        safe_name = f"{resolved_path.name}_{path_hash}{bm25_suffix}".replace(" ", "_")
        return base_dir / safe_name

    def build_kb(self, folder_path: str, use_bm25: bool = True) -> Dict[str, Any]:
        folder = Path(folder_path)

        docs = read_documents(folder)
        chunks = self.chunker.build_chunks(docs)
        texts = [c["text"] for c in chunks]

        embeddings = self.embedder.encode(texts) if texts else None
        index = self.vector_store.build_index(embeddings) if embeddings is not None else None

        graph = self.graph_builder.build(chunks)

        kb = {
            "docs": docs,
            "chunks": chunks,
            "faiss_index": index,
            "dim": self.embedder.dim if embeddings is not None else 0,
            "knowledge_graph": graph,
        }

        if use_bm25 and texts:
            corpus = [BM25Manager.tokenize(t) for t in texts]
            kb["bm25_corpus"] = corpus
            kb["bm25_index"] = self.bm25_manager.build_index(corpus)

        self._save_to_cache(folder_path, kb, use_bm25)
        return kb

    def _save_to_cache(self, folder_path: str, kb: Dict[str, Any], use_bm25: bool):
        prefix = self._get_cache_prefix(folder_path, use_bm25).name
        chunks_file = self.cache_root / f"{prefix}_chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(kb["chunks"], f, ensure_ascii=False, indent=2)

        if kb.get("faiss_index"):
            self.vector_store.save_index(kb["faiss_index"], prefix)

        if use_bm25 and "bm25_corpus" in kb:
            self.bm25_manager.save_corpus(kb["bm25_corpus"], prefix)

        if kb.get("knowledge_graph"):
            self.graph_builder.save_graph(kb["knowledge_graph"], folder_path)

    def load_kb(self, folder_path: str, use_bm25: bool = True) -> Optional[Dict[str, Any]]:
        if folder_path in self._kb_cache:
            return self._kb_cache[folder_path]
        prefix = self._get_cache_prefix(folder_path, use_bm25).name
        chunks_file = self.cache_root / f"{prefix}_chunks.json"
        index_file = self.vector_store.get_cache_path(prefix)
        if not (chunks_file.exists() and index_file.exists()):
            return None

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        index = self.vector_store.load_index(prefix)
        kb = {
            "chunks": chunks,
            "faiss_index": index,
            "dim": index.d if index else 0,
        }

        graph = self.graph_builder.load_graph(folder_path)
        if graph:
            kb["knowledge_graph"] = graph

        if use_bm25:
            corpus = self.bm25_manager.load_corpus(prefix)
            if corpus:
                kb["bm25_corpus"] = corpus
                kb["bm25_index"] = self.bm25_manager.build_index(corpus)
            else:
                # Build from chunks if cached corpus not found
                texts = [chunk["text"] for chunk in chunks]
                corpus = [BM25Manager.tokenize(t) for t in texts]
                kb["bm25_corpus"] = corpus
                kb["bm25_index"] = self.bm25_manager.build_index(corpus)

        self._kb_cache[folder_path] = kb
        return kb

    def rebuild_kb(self, folder_path: str, chunk_size: Optional[int] = None,  overlap: Optional[int] = None, force: bool = False) -> Dict[str, Any]:
        if chunk_size is not None:
            self.chunker.chunk_size = chunk_size
        if overlap is not None:
            self.chunker.overlap = overlap

        if force:
            self._clear_cache(folder_path)

        return self.build_kb(folder_path)

    def _clear_cache(self, folder_path: str):
        import hashlib
        resolved_path = Path(folder_path).resolve()
        path_hash = hashlib.sha256(str(resolved_path).encode()).hexdigest()[:8]
        base_name = resolved_path.name.replace(" ", "_")

        # Clear both bm25 and non-bm25 variants
        for use_bm25 in [False, True]:
            bm25_suffix = "_bm25" if use_bm25 else ""
            prefix = f"{base_name}_{path_hash}{bm25_suffix}"

            for cache_dir in [
                Path(settings.KB_CACHE_ROOT),
                Path(settings.KB_CACHE_GRAPH_ROOT),
                Path(settings.KB_CACHE_BM25_ROOT),
            ]:
                for file in cache_dir.glob(f"{prefix}_*"):
                    try:
                        file.unlink()
                    except Exception as e:
                        logger.warning(f"Не удалось удалить {file}: {e}")

        self._kb_cache.pop(folder_path, None)

    def get_kb(self, folder_path: str, use_bm25: bool = True) -> Dict[str, Any]:
        kb = self.load_kb(folder_path, use_bm25)
        if kb is None:
            logger.info(f"KB cache not found for {folder_path}. Building from scratch.")
            kb = self.build_kb(folder_path, use_bm25)
        return kb
