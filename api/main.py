from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from cloud.client import MQTTClient
from database.db import initialize_database


mqtt_service = MQTTClient()


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n========================================")
    print("       STRESS PREDICTION API")
    print("========================================")

    initialize_database()

    mqtt_service.start()

    yield

    mqtt_service.stop()


app = FastAPI(
    title="Stress Prediction API",
    description=(
        "REST API for wearable-based "
        "stress level prediction using PyTorch."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)