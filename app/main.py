from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Spam Classifier API", version="1.0.0")

# Register routes
app.include_router(router, prefix="/api", tags=["Spam Classifier"])

@app.get("/")
def home():
    return {"message": "Welcome to the Spam Classifier API"}
