from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import archives, chatbot, iot
from services.mqtt_client import mqtt_manager

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend Service Box Archive: Smart Class & Class Archive System",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    mqtt_manager.start()


app.include_router(iot.router)
app.include_router(archives.router)
app.include_router(chatbot.router)


@app.get("/", tags=["Health Check"])
async def root_health_check():
    return {
        "status": "online",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }
