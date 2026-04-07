import os
from uuid import uuid4
import json
import csv
from io import StringIO


def save_photo_to_disk(photo_data: bytes) -> str:
    """Saves one image file to disk and returns the path"""
    filename = f"{uuid4()}.jpg"
    path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    with open(path, "wb") as f:
        f.write(photo_data)
    return path

def extract_images_from_zip(zip_bytes: bytes, filename:str) -> str:
    """Unpacks the ZIP archive and saves all photos to disk"""
    pass

def parse_measurements_file(file_bytes: bytes, filename: str) -> dict:
    """Parses the measurements file (CSV or JSON) and returns a structured dict"""
    if filename.endswith(".json"):
        return json.loads(file_bytes.decode('utf-8'))
    elif filename.endswith(".csv"):
        content = file_bytes.decode('utf-8-sig')
        reader = csv.reader(StringIO(content))
        data = {}
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                raise ValueError("Invalid CSV format: each row must have at least 2 columns")
            key = row[0].strip()
            value = row[1].strip()
            data[key] = value
        return data
    else:
        raise ValueError("Unsupported file format: only CSV and JSON are allowed")
        