def is_file_allowed(file_extension: str) -> bool:
    return file_extension.lower() in [".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp", ".png"]
    