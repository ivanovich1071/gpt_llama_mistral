import os
import nltk
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from dotenv import load_dotenv
from openai import OpenAI

# Загрузка необходимых ресурсов для nltk
nltk.download('punkt')
nltk.download('punkt_tab')

class LlamaChat:
    def __init__(self):
        self.model = "llama-3.2-1b-instruct"
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def generate_response(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def analyze_relevance(self, prompt_text, creative_text):
        #
        #  косинусное сходство между эмбеддингами prompt_text и creative_text
        model = SentenceTransformer('all-mpnet-base-v2')
        prompt_embedding = model.encode([prompt_text], convert_to_tensor=True)
        creative_embedding = model.encode([creative_text], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(prompt_embedding, creative_embedding)[0][0]
        return similarity.item()

# Шаг 2: Подготовка базы знаний
def load_and_preprocess_knowledge_base(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = nltk.sent_tokenize(text, language='russian')
    return sentences


def chunk_text(text, chunk_size=500):
    # Проверяем, является ли text списком
    if isinstance(text, list):
        # Объединяем элементы списка в одну строку
        text = ' '.join(text)
    elif not isinstance(text, str):
        raise TypeError(f"Ожидалась строка или список, получен объект типа {type(text)}")

    # Теперь text гарантированно является строкой
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(nltk.word_tokenize(sentence))
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Шаг 3: Генерация эмбеддингов
def generate_embeddings(texts, model_name='sentence-transformers/all-mpnet-base-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

# Шаг 4: Вычисление косинусного сходства
def compute_cosine_similarity(question_embedding, chunk_embeddings):
    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)
    return similarities

# Шаг 5: Оценка качества ответов моделей
def evaluate_model_responses(questions, chunks, chunk_embeddings, llama_chat):
    results = []

    for question in questions:
        question_embedding = generate_embeddings([question])
        similarities = compute_cosine_similarity(question_embedding, chunk_embeddings)
        max_similarity_index = np.argmax(similarities)
        max_similarity_value = similarities[0][max_similarity_index]

        if max_similarity_value > 0.8:
            relevant_chunk = chunks[max_similarity_index]
            prompt = (
                "Ты менеджер поддержки в чате Российской компании Университет Искусственного Интеллекта. "
                "Компания продает курсы по AI. У компании есть большой документ со всеми материалами о продуктах компании на русском языке. "
                "Тебе задает вопрос клиент в чате, дай ему ответ на языке оригинала, опираясь на отрывки из этого документа, "
                "постарайся ответить так, чтобы человек захотел после ответа купить обучение. Отвечай максимально точно по документу, "
                "не придумывай ничего от себя. Никогда не ссылайся на название документа или названия его отрывков при ответе, "
                "клиент ничего не должен знать о документе, по которому ты отвечаешь. Отвечай от первого лица без ссылок на источники, "
                "на которые ты опираешься. Если ты не знаешь ответа или его нет в документе, то ответь 'Я не могу ответить на этот вопрос'.\n"
                f"Вопрос клиента: {question}\nОтрывок из документа: {relevant_chunk}"
            )
            start_time = time.time()
            model_response = llama_chat.generate_response(prompt)
            response_time = time.time() - start_time

            response_embedding = generate_embeddings([model_response])
            response_similarity = compute_cosine_similarity(response_embedding, question_embedding)[0][0]

            # Оценка по шкале от -2 до 2
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

            results.append({
                "Вопрос": question,
                "Ответ модели": model_response,
                "Время ответа": response_time,
                "Оценка": score
            })

    return results

# Шаг 6: Сохранение результатов
def save_results_to_csv(results, file_path):
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)

# Шаг 7: Анализ результатов
def analyze_results(results):
    df = pd.DataFrame(results)
    if "Время ответа" not in df.columns:
        raise KeyError("Столбец 'Время ответа' отсутствует в результатах.")
    if "Оценка" not in df.columns:
        raise KeyError("Столбец 'Оценка' отсутствует в результатах.")

    average_response_time = df["Время ответа"].mean()
    average_score = df["Оценка"].mean()
    print(f"Среднее время ответа: {average_response_time:.2f} сек")
    print(f"Средняя оценка: {average_score:.2f}")
# Основная функция
def main():
    knowledge_base_path = r'C:\Users\Dell\Documents\GitHub\База знаний УИИ. Версия от 12.06.23 (1).txt'
    questions_path = r'C:\Users\Dell\Documents\GitHub\вопросы.csv'

    # Загрузка базы знаний
    sentences = load_and_preprocess_knowledge_base(knowledge_base_path)
    chunks = chunk_text(sentences)
    chunk_embeddings = generate_embeddings(chunks)

    # Загрузка тестовых вопросов
    with open(questions_path, 'r', encoding='utf-8') as file:
        questions = file.read().splitlines()

    llama_chat = LlamaChat()
    results = evaluate_model_responses(questions, chunks, chunk_embeddings, llama_chat)
    save_results_to_csv(results, 'results_llama.csv')
    analyze_results(results)

if __name__ == "__main__":
    main()
