import fasttext
import os
import shutil
from datetime import datetime

# Пути к файлам
ORIGINAL_TRAIN_DATA = "data/train_ft.txt"
FEEDBACK_DATA = "feedback_data.txt"
COMBINED_DATA = "data/combined_train_ft.txt"
MODEL_PATH = "model/fasttext_spam_model.bin"
BACKUP_PATH = f"model/backups/fasttext_spam_model_{datetime.now().strftime('%Y%m%d')}.bin"

def retrain():
    print("--- Запуск процесса переобучения ---")

    if not os.path.exists(FEEDBACK_DATA) or os.stat(FEEDBACK_DATA).st_size == 0:
        print("Новых данных для переобучения нет. Выход.")
        return

    os.makedirs("model/backups", exist_ok=True)
    if os.path.exists(MODEL_PATH):
        shutil.copy(MODEL_PATH, BACKUP_PATH)
        print(f"Бэкап старой модели сохранен: {BACKUP_PATH}")

    with open(COMBINED_DATA, 'w', encoding='utf-8') as outfile:
        for fname in [ORIGINAL_TRAIN_DATA, FEEDBACK_DATA]:
            with open(fname, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write("\n")
    
    print(f"Данные объединены в {COMBINED_DATA}")

    # 4. Обучаем новую модель
    # Используем те же параметры, что и при первом обучении
    ft_model = fasttext.train_supervised(
        input=COMBINED_DATA,
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        dim=100,
        loss='softmax'
    )

    ft_model.save_model(MODEL_PATH)
    print(f"Новая модель успешно обучена и сохранена в {MODEL_PATH}")

    shutil.copy(FEEDBACK_DATA, f"data/archive_feedback_{datetime.now().strftime('%Y%m%d')}.txt")
    open(FEEDBACK_DATA, 'w').close() 
    print("Файл фидбека очищен и архивирован.")

if __name__ == "__main__":
    retrain()