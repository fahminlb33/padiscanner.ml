from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Padi Scanner Analysis API"
    applicationinsights_connection_string: str = "InstrumentationKey="
    azure_storage_connection_string: str = ""
    azure_storage_container_name: str = "padi"

    auth_basic_username: str = "fahmi"
    auth_basic_password: str = "fahmi"

    model_path: str = "tensorflow.h5"
    class_names_path: str = "class_names.z"

