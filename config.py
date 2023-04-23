from pydantic import BaseSettings

class Settings(BaseSettings):
    upload_dest: str = "uploads/"
    result_dest: str = "results/"

# global instance
settings = Settings()