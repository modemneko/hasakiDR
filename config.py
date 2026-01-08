import os
import google.generativeai as genai
from browser_use.llm.google.chat import ChatGoogle
from browser_use.llm.ollama.chat import ChatOllama

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
# os.environ["HTTPS_PROXY"] = "https://127.0.0.1:7897"

GOOGLE_API_KEY = "AIzaSyAsSJkbRztMYkbY79Z3jJT2Dq_cM8ZBzZ4"
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# LLM = ChatGoogle(
#     model="gemini-2.0-flash",
#     api_key=GOOGLE_API_KEY,
#     temperature=0.5,
# )

LLM = ChatOllama(
    model="Qwen3:0.6b",
)