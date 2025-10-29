from .years import router as years_router
from .curations import router as curations_router
from .swaps import router as swaps_router
from .sync import router as sync_router

__all__ = [
    "years_router",
    "curations_router",
    "swaps_router",
    "sync_router"
]
