import fasttext
import re

class FastTextSpamPipeline:
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)

    def _preprocess(self, text):
        text = text.replace("\n", " ").strip().lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def predict(self, raw_text):
        clean_text = self._preprocess(raw_text)
        labels, probabilities = self.model.predict(clean_text, k=1)
        label = labels[0].replace("__label__", "")
        label_name = "spam" if label == "1" else "ham"
        return {
            "label": label_name,
            "probability": round(float(probabilities[0]), 4)
        }