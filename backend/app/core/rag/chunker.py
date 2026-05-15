from pathlib import Path
from typing import Any, Dict, List
import re


MEDICAL_ENTITY_PATTERNS = {
    "ANATOMY": [
        "восходящая аорта",
        "нисходящая аорта",
        "дуга аорты",
        "перешеек аорты",
        "грудная аорта",
        "брюшная аорта",
        "аортальный клапан",
        "левая подключичная артерия",
        "плечеголовной ствол",
    ],
    "DISEASE": [
        "аневризма",
        "расслоение",
        "диссекция",
        "стеноз",
        "разрыв",
        "расширение",
        "тромбоз",
    ],
    "CLINICAL": [
        "боль в груди",
        "боль в спине",
        "боль между лопатками",
        "одышка",
        "кашель",
        "хрипы",
        "затруднённое дыхание",
        "ком в горле",
    ],
    "TACTIC": [
        "наблюдение",
        "кт-контроль",
        "хирургическое лечение",
        "эндоваскулярное лечение",
        "протезирование",
        "стентирование",
        "консультация кардиохирурга",
    ],
    "GUIDELINE": [
        "рекомендуется",
        "показания",
        "противопоказания",
        "порог",
        "норма",
        "нормальный диаметр",
        "риск расслоения",
        "риск разрыва",
    ],
}

CLUSTER_PATTERNS = {
    "thoracic_aorta": [
        "грудная аорта",
        "восходящая аорта",
        "нисходящая аорта",
        "дуга аорты",
        "перешеек аорты",
    ],
    "abdominal_aorta": [
        "брюшная аорта",
        "инфраренальный отдел",
        "супраренальный отдел",
    ],
    "ascending_aorta": [
        "восходящая аорта",
        "аортальный клапан",
        "синусы вальсальвы",
        "синотубулярное соединение",
    ],
    "aortic_arch": [
        "дуга аорты",
        "плечеголовной ствол",
        "левая подключичная артерия",
        "левая общая сонная артерия",
    ],
    "descending_aorta": [
        "нисходящая аорта",
        "грудной отдел аорты",
    ],
    "aneurysm": [
        "аневризма",
        "расширение",
        "дилатация",
    ],
    "dissection": [
        "расслоение",
        "диссекция",
        "ложный просвет",
        "истинный просвет",
    ],
    "stenosis_thrombosis": [
        "стеноз",
        "тромбоз",
        "окклюзия",
    ],
    "surgery_indications": [
        "показания к операции",
        "хирургическое лечение",
        "эндоваскулярное лечение",
        "протезирование",
        "стентирование",
        "порог вмешательства",
    ],
    "follow_up": [
        "наблюдение",
        "кт-контроль",
        "динамическое наблюдение",
        "контроль через",
    ],
}

_TEXT_SPLITTER = None


def get_text_splitter():
    """Lazy initialization keeps module import fast and prevents side effects."""
    global _TEXT_SPLITTER
    if _TEXT_SPLITTER is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        _TEXT_SPLITTER = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
    return _TEXT_SPLITTER


def assign_clusters_to_text(text: str) -> List[str]:
    text_lower = text.lower()
    clusters: List[str] = []

    for cluster_name, patterns in CLUSTER_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in text_lower:
                clusters.append(cluster_name)
                break

    if not clusters:
        clusters.append("general")

    return clusters


def extract_entities_from_text(text: str) -> List[Dict[str, str]]:
    text_lower = text.lower()
    entities: List[Dict[str, str]] = []

    for entity_type, terms in MEDICAL_ENTITY_PATTERNS.items():
        for term in terms:
            if term in text_lower:
                entities.append({"name": term, "type": entity_type})

    numeric_patterns = re.findall(
        r"(?:>=|<=|≥|≤|>|<)?\s?\d+[.,]?\d*\s?(?:мм|см)",
        text_lower,
    )

    for value in numeric_patterns:
        entities.append({"name": value.strip(), "type": "MEASUREMENT"})

    unique = {}
    for entity in entities:
        key = (entity["name"], entity["type"])
        unique[key] = entity

    return list(unique.values())


def build_chunks(docs: List[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    split_docs = get_text_splitter().split_documents(docs)
    source_counters: Dict[str, int] = {}

    for doc in split_docs:
        source_path = doc.metadata.get("source", "unknown")
        source = Path(source_path).name

        chunk_id = source_counters.get(source, 0)
        source_counters[source] = chunk_id + 1

        result.append(
            {
                "source": source,
                "chunk_id": chunk_id,
                "text": doc.page_content,
                "metadata": doc.metadata,
                "clusters": assign_clusters_to_text(doc.page_content),
                "entities": extract_entities_from_text(doc.page_content),
            }
        )

    return result
