from fastapi import APIRouter

from src.app.services.langchain import validity_check

langchain_router = APIRouter()


@langchain_router.get("/{text}")
async def langchain(text: str):
    result = validity_check(text)
    if isinstance(result, dict):
        result = result.get("output", "Can't retrieve any answer")
    elif isinstance(result, str):
        result = result
    return {"message": result}