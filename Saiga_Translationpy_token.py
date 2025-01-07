import os
import time
import requests

# Клиент для LM Studio
class LMStudioClient:
    def __init__(self, base_url="http://localhost:1234/v1", api_key="lm-studio"):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def generate_response(self, prompt, max_tokens=4096):
        payload = {
            "model": "saiga2_7b_gguf",
            "messages": [{"role": "system", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "top_p": 0.9
        }
        response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"], response.json()["usage"]

# Функция для токенизации текста
def count_tokens_locally(text, client):
    prompt = f"Токенизируй текст:\n{text}"
    _, usage = client.generate_response(prompt, max_tokens=1)
    return usage["prompt_tokens"]

# Функция перевода текста
def translate_text(file_path, output_path, client):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
    translations = []

    for chunk in chunks:
        prompt = f"Переведи текст на английский:\n{chunk}"
        start_time = time.time()
        translation, usage = client.generate_response(prompt)
        response_time = time.time() - start_time
        token_count = usage["prompt_tokens"]
        translations.append((translation, token_count, response_time))

    with open(output_path, "w", encoding="utf-8") as f:
        for translation, token_count, response_time in translations:
            f.write(f"Перевод: {translation}\nТокены: {token_count}\nВремя ответа: {response_time:.2f} секунд\n\n")

    print(f"Перевод сохранен в файл {output_path}")

# Функция перевода на несколько языков
def translate_to_multiple_languages(input_file, output_file, client):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Разбиваем текст на чанки по 1000 токенов
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in text.split('.'):
        sentence = sentence.strip() + '.'
        sentence_tokens = count_tokens_locally(sentence, client)
        if current_tokens + sentence_tokens > 1000 and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            current_tokens = 0
        current_chunk += " " + sentence
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    translations = []

    for chunk in chunks:
        prompt = f"Переведи текст на три языка: русский, немецкий и итальянский:\n{chunk}"
        start_time = time.time()
        translation, usage = client.generate_response(prompt, max_tokens=3000)
        response_time = time.time() - start_time
        translations.append((translation, usage["prompt_tokens"], response_time))

    with open(output_file, "w", encoding="utf-8") as f:
        for translation, token_count, response_time in translations:
            f.write(f"Перевод:\n{translation}\nТокены: {token_count}\nВремя ответа: {response_time:.2f} секунд\n\n")

    print(f"Многоязычный перевод сохранен в файл {output_file}")

# Функция анализа токенов
def analyze_tokens(input_file, output_file, client):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Берем первый фрагмент текста
    chunk = text[:2000]

    # Переводим фрагмент на английский
    prompt_en = f"Переведи текст на английский:\n{chunk}"
    start_time_en = time.time()
    translation_en, usage_en = client.generate_response(prompt_en)
    response_time_en = time.time() - start_time_en
    token_count_en = usage_en["prompt_tokens"]

    # Переводим английский текст обратно на русский
    prompt_ru = f"Переведи текст на русский:\n{translation_en}"
    start_time_ru = time.time()
    translation_ru, usage_ru = client.generate_response(prompt_ru)
    response_time_ru = time.time() - start_time_ru
    token_count_ru = usage_ru["prompt_tokens"]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Оригинальный текст:\n{chunk}\n\n")
        f.write(f"Перевод на английский:\n{translation_en}\nТокены: {token_count_en}\nВремя ответа: {response_time_en:.2f} секунд\n\n")
        f.write(f"Обратный перевод на русский:\n{translation_ru}\nТокены: {token_count_ru}\nВремя ответа: {response_time_ru:.2f} секунд\n\n")

    print(f"Анализ токенов сохранен в файл {output_file}")

# Основной блок
if __name__ == "__main__":
    input_file = "C:/Users/Dell/Documents/GitHub/gpt_llama_mistral/База знаний -фрагмент.txt"
    output_file = "C:/Users/Dell/Documents/GitHub/gpt_llama_mistral/UII_Knowledge_Base.txt"
    output_multilang_file = "C:/Users/Dell/Documents/GitHub/gpt_llama_mistral/UII_Knowledge_Base_transl.txt"
    token_analysis_file = "C:/Users/Dell/Documents/GitHub/gpt_llama_mistral/token_analysis.txt"

    # Инициализация клиента LM Studio
    lm_studio_client = LMStudioClient()

    # Перевод базы знаний
    translate_text(input_file, output_file, lm_studio_client)

    # Перевод на несколько языков
    translate_to_multiple_languages(output_file, output_multilang_file, lm_studio_client)

    # Анализ токенов
    analyze_tokens(input_file, token_analysis_file, lm_studio_client)
