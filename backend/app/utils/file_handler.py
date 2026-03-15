async def save_photo_to_disk(photo_data: bytes) -> str:
    import os
    from uuid import uuid4
    filename = f"{uuid4()}.jpg"
    path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    with open(path, "wb") as f:
        f.write(photo_data)
    return path