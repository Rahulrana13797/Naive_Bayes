from fastapi import APIRouter
from app.model import SpamClassifier
from app.schemas import TextInput, PredictionResponse

# Initialize router and classifier
router = APIRouter()
classifier = SpamClassifier()
classifier.load_model('app/model.pkl', 'app/vectorizer.pkl')

@router.post("/predict", response_model=PredictionResponse)
async def predict_spam(input_data: TextInput):
    prediction = classifier.predict(input_data.text)
    return PredictionResponse(prediction=prediction)
