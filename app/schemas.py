from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
