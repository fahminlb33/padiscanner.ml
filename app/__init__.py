# imports
import os
import secrets
import logging
from functools import lru_cache

from opencensus.trace import config_integration
from opencensus.trace.span import SpanKind
from opencensus.trace.tracer import Tracer
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.attributes_helper import COMMON_ATTRIBUTES
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.ext.azure.log_exporter import AzureLogHandler

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from starlette import status
from starlette.exceptions import HTTPException
from starlette.requests import Request

from app.config import Settings

HTTP_URL = COMMON_ATTRIBUTES['HTTP_URL']
HTTP_STATUS_CODE = COMMON_ATTRIBUTES['HTTP_STATUS_CODE']

# --- dependencies

@lru_cache()
def get_settings():
    return Settings()

def callback_add_role_name(envelope):
    envelope.tags["ai.cloud.role"] = "Analysis API"
    return True


# --- main app

# global settings
settings = get_settings()

# fastapi App
if os.environ.get("PYTHON_ENV", "development") == "development":
    app = FastAPI()
else:
    app = FastAPI(docs_url=None, redoc_url=None)

# auth
security = HTTPBasic()

# CORS
origins = [
    "https://padi-scanner.kodesiana.com",
    "https://kodesiana.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# enable trace integrations
config_integration.trace_integrations(["logging", "requests"])

# setup logging with Azure App Insights
handler = AzureLogHandler(connection_string=settings.applicationinsights_connection_string)
handler.add_telemetry_processor(callback_add_role_name)

logger = logging.getLogger(__name__)
logger.addHandler(handler)

# metrics exporter
sampler = ProbabilitySampler(1.0)
exporter = AzureExporter(connection_string=settings.applicationinsights_connection_string)
exporter.add_telemetry_processor(callback_add_role_name)

# --- dependencies

def get_current_username(credentials: HTTPBasicCredentials = Depends(security), settings: Settings = Depends(get_settings)):
    username_correct = secrets.compare_digest(
        credentials.username.encode("utf8"),
        settings.auth_basic_username.encode("utf8")
    )
    password_correct = secrets.compare_digest(
        credentials.password.encode("utf8"),
        settings.auth_basic_password.encode("utf8")
    )

    if not (username_correct and password_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


# --- app middlewares

@app.middleware("http")
async def opencensus_trace(request: Request, call_next):
    tracer = Tracer(exporter=exporter, sampler=sampler)
    with tracer.span("main") as span:
        span.span_kind = SpanKind.SERVER

        response = await call_next(request)

        tracer.add_attribute_to_current_span(attribute_key="name", attribute_value=f"{request.method.upper()} {request.url}")
        tracer.add_attribute_to_current_span(attribute_key=HTTP_STATUS_CODE, attribute_value=response.status_code)
        tracer.add_attribute_to_current_span(attribute_key=HTTP_URL, attribute_value=str(request.url))

    return response


# --- app routes

from app.domain import image

app.include_router(image.router)


@app.get("/")
async def root():
    return "Welcome to Project Smart Economy API! Please contact the developers to get access to this API."

