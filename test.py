import os
import logging
import requests
from bs4 import BeautifulSoup
from decouple import config
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackQueryHandler
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

OPENAI_API_KEY = config('OPENAI_API_KEY')
TELEGRAM_TOKEN = config('TELEGRAM_TOKEN')
SOURCES = [
    "https://coda.io/@latoken/latoken-talent/latoken-161",
    "https://deliver.latoken.com/hackathon",
    "https://latoken.me/culture-139"
]

client = OpenAI(api_key=OPENAI_API_KEY)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

documents = []
embeddings = None

def parse_sources():
    global documents, embeddings
    documents = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    
    for url in SOURCES:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])[:10000]
            
            documents.append({
                "url": url,
                "content": content
            })
            logging.info(f"успешно спарсено: {url}")
            
        except Exception as e:
            logging.error(f"ошибка парсинга {url}: {str(e)}")
    
    if documents:
        embeddings = embedder.encode([doc["content"] for doc in documents])
    else:
        embeddings = np.array([])

def get_relevant_context(query: str, top_k: int = 2) -> str:
    if embeddings is None or len(embeddings) == 0:
        return ""
    
    query_embedding = embedder.encode([query])
    similarities = util.dot_score(query_embedding, embeddings).flatten()
    top_indices = np.argsort(similarities)[-top_k:]
    
    return "\n\n".join([
        documents[i]["content"] for i in top_indices
    ])

def generate_answer(question: str) -> str:
    context = get_relevant_context(question)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ты эксперт по компании Latoken. Отвечай на русском языке, используя предоставленный контекст."},
                {"role": "user", "content": f"Вопрос: {question}\n\nКонтекст:\n{context}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logging.error(f"ошибка генерации ответа: {str(e)}")
        return "ошибка при обработке запроса"

async def start(update: Update, context) -> None:
    await update.message.reply_text(
        "Привет! Я бот Latoken. Могу ответить на вопросы о компании, хакатонах и корпоративной культуре."
    )

QUIZ = {
    "question": "Какой актив Latoken запустил первым?",
    "options": ["BTC", "ETH", "LA Token"],
    "answer": 2  
}

async def handle_message(update: Update, context) -> None:
    user_message = update.message.text.lower()
    if user_message == "хочу квиз":
        keyboard = [
            [InlineKeyboardButton(option, callback_data=f"quiz_{i}")] 
            for i, option in enumerate(QUIZ["options"])
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(QUIZ["question"], reply_markup=reply_markup)
    user_question = update.message.text
    logging.info(f"Получен вопрос: {user_question}")
    
    answer = generate_answer(user_question)
    await update.message.reply_text(answer)


async def quiz_callback(update: Update, context):
    query = update.callback_query
    await query.answer()  
    selected_option = int(query.data.split("_")[1])
    
    if selected_option == QUIZ["answer"]:
        response = "✅ Правильно!!!"
    else:
        correct_answer = QUIZ["options"][QUIZ["answer"]]
        response = f"❌ Неправильно(( Правильный ответ был: {correct_answer}"
    
    await query.edit_message_text(text=response)


async def start(update: Update, context):
    keyboard = [
        [InlineKeyboardButton("О компании", callback_data="company_info")],
        [InlineKeyboardButton("Хакатоны", callback_data="hackathon")],
        [InlineKeyboardButton("Корпоративная культура", callback_data="culture")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите тему:", reply_markup=reply_markup)



if __name__ == "__main__":
    parse_sources()
    
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(quiz_callback))
    
    application.run_polling()

