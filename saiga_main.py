import os
import pandas as pd
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import normalize
import requests
import time

# **Клиент для LM Studio**
class LMStudioClient:
    def __init__(self, base_url="http://localhost:1234/v1", api_key="lm-studio"):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def generate_response(self, prompt, max_tokens=1024):
        payload = {
            "model": "saiga2_7b_gguf",
            "messages": [{"role": "system", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.9
        }
        response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

# **Функция для загрузки файла с корректной кодировкой**
def load_file(file_path):
    for encoding in ['utf-8', 'utf-16', 'cp1251']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Не удалось загрузить файл {file_path} с проверенными кодировками.")

# **Настройка базы знаний**
FILE_PATH = r'C:\Users\Dell\Documents\GitHub\База знаний УИИ. Версия от 12.06.23 (1).txt'
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Файл базы знаний не найден: {FILE_PATH}")

text = load_file(FILE_PATH)
loader = TextLoader(FILE_PATH, encoding="utf-8")
documents = loader.load()

# Разделение текста на чанки
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Создание эмбеддингов и векторной базы
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embedding_model.encode([doc.page_content for doc in docs], convert_to_tensor=True, dtype=np.float32)  # Преобразование к float32
chunk_embeddings = normalize(chunk_embeddings)
db = FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
print("База знаний успешно создана!")

# **Загрузка вопросов**
QUESTIONS_PATH = r'C:\Users\Dell\Documents\GitHub\вопросы.csv'
if not os.path.exists(QUESTIONS_PATH):
    raise FileNotFoundError(f"Файл с вопросами не найден: {QUESTIONS_PATH}")

questions_df = pd.read_csv(QUESTIONS_PATH, encoding='utf-8')
questions = questions_df['Вопрос'].values

# **Функции для оценки ответов**
def generate_embeddings(texts, model):
    embeddings = model.encode(texts, convert_to_tensor=True, dtype=np.float32)  # Преобразование к float32
    return normalize(embeddings, norm='l2')

def compute_cosine_similarity(question_embedding, chunk_embeddings):
    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)
    return similarities

def evaluate_model_responses(questions, lm_studio_client, docs, chunk_embeddings, embedding_model, top_k=3):
    results = []

    for question in questions:
        # Генерация эмбеддингов для вопроса
        question_embedding = generate_embeddings([question], embedding_model)

        # Вычисление косинусного сходства
        similarities = compute_cosine_similarity(question_embedding, chunk_embeddings)

        # Получение индексов наиболее релевантных чанков
        top_indices = np.argsort(-similarities[0].cpu().numpy())[:top_k]
        relevant_chunks = [docs[i].page_content for i in top_indices]

        # Формирование промпта для модели
        chunks_text = "\n".join(relevant_chunks)
        prompt = (
            "Ты менеджер поддержки в чате Российской компании Университет Искусственного Интеллекта. "
            "Компания продает курсы по AI. У компании есть большой документ со всеми материалами о продуктах компании на русском языке. "
            "Тебе задает вопрос клиент в чате, дай ему ответ на языке оригинала, опираясь на отрывки из этого документа, "
            "постарайся ответить так, чтобы человек захотел после ответа купить обучение. Отвечай максимально точно по документу, "
            "не придумывай ничего от себя. Никогда не ссылайся на название документа или названия его отрывков при ответе, "
            "клиент ничего не должен знать о документе, по которому ты отвечаешь. Отвечай от первого лица без ссылок на источники, "
            "на которые ты опираешься. Если ты не знаешь ответа или его нет в документе, то ответь 'Я не могу ответить на этот вопрос'.\n"
            f"Вопрос клиента: {question}\nОтрывки из документа:\n{chunks_text}"
        )

        # Взаимодействие с LM Studio
        start_time = time.time()
        model_response = lm_studio_client.generate_response(prompt)
        response_time = time.time() - start_time

        # Оценка качества ответа
        response_embedding = generate_embeddings([model_response], embedding_model)
        response_similarity = compute_cosine_similarity(response_embedding, question_embedding)[0][0]

        # Присвоение оценки
        if response_similarity > 0.8:
            score = 2
        elif response_similarity > 0.6:
            score = 1
        elif response_similarity > 0.5:
            score = 0
        elif response_similarity > 0.3:
            score = -1
        else:
            score = -2

        # Сохранение результата
        results.append({
            "Вопрос": question,
            "Ответ модели": model_response,
            "Время ответа": response_time,
            "Оценка": score
        })

    return results

def save_results_to_csv(results, file_path):
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)

def main():
    lm_studio_client = LMStudioClient()
    results = evaluate_model_responses(questions, lm_studio_client, docs, chunk_embeddings, embedding_model)
    save_results_to_csv(results, 'results_saiga.csv')
    print("Результаты сохранены в 'results_saiga.csv'")

if __name__ == "__main__":
    main()
