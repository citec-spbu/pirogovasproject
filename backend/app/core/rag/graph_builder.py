import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle
import networkx as nx
import pymorphy3
from app.core.config import get_settings

settings = get_settings()

# Module-level singleton for morphological analysis
_morph = pymorphy3.MorphAnalyzer()

def normalize_text(text: str) -> set:
    """Лемматизация текста для нормализации ключевых слов."""
    words = re.findall(r'\b[а-яА-ЯёЁa-zA-Z]{3,}\b', text.lower())
    return {_morph.parse(w)[0].normal_form for w in words}

# Медицинские паттерны для извлечения сущностей
MEDICAL_ENTITY_PATTERNS = {
    "ANATOMY": [
        "восходящая аорта", "нисходящая аорта", "дуга аорты", "перешеек аорты",
        "грудная аорта", "брюшная аорта", "аортальный клапан",
        "левая подключичная артерия", "плечеголовной ствол",
    ],
    "DISEASE": [
        "аневризма", "расслоение", "диссекция", "стеноз",
        "разрыв", "расширение", "тромбоз",
    ],
    "CLINICAL": [
        "боль в груди", "боль в спине", "боль между лопатками",
        "одышка", "кашель", "хрипы", "затруднённое дыхание", "ком в горле",
    ],
    "TACTIC": [
        "наблюдение", "кт-контроль", "хирургическое лечение",
        "эндоваскулярное лечение", "протезирование", "стентирование",
        "консультация кардиохирурга",
    ],
    "GUIDELINE": [
        "рекомендуется", "показания", "противопоказания", "порог",
        "норма", "нормальный диаметр", "риск расслоения", "риск разрыва",
    ],
}

def _entity_node_id(entity: Dict[str, str]) -> str:
    return f"entity::{entity['type']}::{entity['name']}"

def _chunk_node_id(source: str, chunk_id: int) -> str:
    return f"chunk::{source}::{chunk_id}"

def extract_entities_from_text(text: str) -> List[Dict[str, str]]:
    """Извлекает медицинские сущности и измерения из текста."""
    text_lower = text.lower()
    entities = []

    # Поиск по паттернам
    for entity_type, terms in MEDICAL_ENTITY_PATTERNS.items():
        for term in terms:
            if term in text_lower:
                entities.append({"name": term, "type": entity_type})

    # Поиск числовых измерений
    numeric_patterns = re.findall(
        r'(?:>=|<=|≥|≤|>|<)?\s?\d+[.,]?\d*\s?(?:мм|см)',
        text_lower
    )
    for value in numeric_patterns:
        entities.append({"name": value.strip(), "type": "MEASUREMENT"})

    # Уникализация
    unique = {}
    for e in entities:
        key = (e["name"], e["type"])
        unique[key] = e
    return list(unique.values())

class KnowledgeGraphBuilder:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or settings.KB_CACHE_GRAPH_ROOT)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build(self, chunks: List[Dict[str, Any]]) -> nx.Graph:
        #Строит граф знаний из списка чанков
        graph = nx.Graph()

        for idx, chunk in enumerate(chunks):
            source = chunk["source"]
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]

            chunk_node = _chunk_node_id(source, chunk_id)
            graph.add_node(chunk_node, node_type="CHUNK", source=source,
                           chunk_id=chunk_id, text=text, index=idx)

            # Добавляем сущности и связи
            entities = extract_entities_from_text(text)
            for entity in entities:
                entity_node = _entity_node_id(entity)
                graph.add_node(
                    entity_node,
                    node_type="ENTITY",
                    entity_type=entity["type"],
                    name=entity["name"],
                )
                graph.add_edge(chunk_node, entity_node, relation="MENTIONS", weight=1.0)

            # Связываем сущности внутри чанка
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1 = _entity_node_id(entities[i])
                    e2 = _entity_node_id(entities[j])
                    if graph.has_edge(e1, e2):
                        graph[e1][e2]["weight"] += 1.0
                    else:
                        graph.add_edge(e1, e2, relation="CO_OCCURS", weight=1.0)

        return graph

    def save_graph(self, graph: nx.Graph, folder_path: str) -> Path:
        safe_name = Path(folder_path).name.replace(" ", "_")
        path = self.cache_dir / f"{safe_name}_graph.pkl"
        with open(path, "wb") as f:
            pickle.dump(graph, f)
        return path

    def load_graph(self, folder_path: str) -> Optional[nx.Graph]:
        safe_name = Path(folder_path).name.replace(" ", "_")
        path = self.cache_dir / f"{safe_name}_graph.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
