from fastapi import APIRouter

from src.app.services.langchain import validity_check

langchain_router = APIRouter()


@langchain_router.get("/{text}")
async def langchain(text: str):
    result = validity_check(text)
    return {"message": result}