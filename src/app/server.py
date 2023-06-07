from fastapi import FastAPI
from src.api import router
from src.config.logger.log_config import LogConfig
from logging.config import dictConfig
from src.common.helpers import vectorstore

def init_routers(app_: FastAPI) -> None:
    app_.include_router(router)


def create_agent():
    return vectorstore.agent


def init_logger():
    dictConfig(LogConfig().dict())


def create_app() -> FastAPI:
    app_ = FastAPI(
        title="CHATGPT HACKATHON API",
        description="GPT3 LangChain <i><b>#GETİR</b><i/>💜💛",
        version="1.0.0",
    )
    init_routers(app_=app_)
    init_logger()
    create_agent()
    return app_


app = create_app()