from fastapi import APIRouter
from src.api.controllers.langchain import langchain_router

router = APIRouter()

router.include_router(
    router=langchain_router, prefix="/message", tags=["LangChain"]
)

__all__ = ["router"]
