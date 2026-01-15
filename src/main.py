import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline import FastTextSpamPipeline

# 1. Настраиваем логирование
logging.basicConfig(
    filename='api_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="FastText Spam Filter API")
pipeline = FastTextSpamPipeline("model/fasttext_spam_model.bin")

class Message(BaseModel):
    text: str

class Feedback(BaseModel):
    text: str
    correct_label: str

@app.post("/predict")
def get_prediction(message: Message):
    
    if not message.text or message.text.isspace():
        logger.warning("Received empty text for prediction")
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start_time = datetime.now()
    
    try:
        result = pipeline.predict(message.text)
        
        latency = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"Text: {message.text[:50]}... | "
            f"Pred: {result['label']} | "
            f"Prob: {result['probability']} | "
            f"Latency: {latency}s"
        )
        
        return {"prediction": result, "latency": latency}
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/feedback")
def collect_feedback(feedback: Feedback):
    if feedback.correct_label not in ["0", "1"]:
        raise HTTPException(status_code=400, detail="Label must be '0' (ham) or '1' (spam)")

    try:
        clean_text = feedback.text.replace("\n", " ")
        
        with open("feedback_data.txt", "a", encoding="utf-8") as f:
            f.write(f"__label__{feedback.correct_label} {clean_text}\n")
        
        return {"status": "success", "message": "Feedback saved for retraining"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")