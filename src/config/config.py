import os
from decouple import config
from pydantic import BaseSettings


class Config(BaseSettings):
    APP_NAME: str = config('APP_NAME')
    APP_VERSION: str = config('APP_VERSION')
    ENV: str = config('ENV')
    BODY_LIMIT: int = config('BODY_LIMIT')
    DEBUG: bool = True
    APP_HOST: str = config('APP_HOST')
    APP_PORT: int = config('APP_PORT')
    LOG_LEVEL: str = config('LOG_LEVEL')
    SWAGGER_ENABLED: str = config('SWAGGER_ENABLED')


class DevelopmentConfig(Config):
    ENV: str = "dev"


class LocalConfig(Config):
    ENV: str = "local"


class ProductionConfig(Config):
    ENV: str = "prod"


def get_config():
    env = os.getenv("ENV", "local")
    config_type = {
        "dev": DevelopmentConfig(),
        "local": LocalConfig(),
        "prod": ProductionConfig(),
    }
    return config_type[env]


config: Config = get_config()