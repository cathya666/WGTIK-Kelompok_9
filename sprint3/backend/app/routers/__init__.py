# routers/__init__.py
from .predict import router as predict_router
# from .chat import router as chat_router

__all__ = ["predict_router"]