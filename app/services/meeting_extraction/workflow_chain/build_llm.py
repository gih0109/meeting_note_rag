from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.config import settings


def build_llm(model_name: str):
    if "gemini-2" in model_name:
        return ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            api_key=settings.GOOGLE_API_KEY,
        )
    elif "gemini-3" in model_name:
        return ChatGoogleGenerativeAI(
            model_name=settings.LLM_MODEL_NAME,
            temperature=1.0,
            api_key=settings.GOOGLE_API_KEY,
        )
