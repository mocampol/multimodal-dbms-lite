import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import engine
from routers import tables, query as query_router

from catalog.exceptions import (
    TableAlreadyExistsError,
    TableNotFoundError,
    ColumnNotFoundError,
    UniqueConstraintError,
)
from query.query_engine import QueryError

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    engine.flush_all_buffers()


app = FastAPI(title="multimodal-dbms-lite API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tables.router)
app.include_router(query_router.router)


@app.get("/health")
def health():
    return {"status": "ok", "tables": len(engine.catalog.tables)}


def _known_error(status: int):
    def handler(_request: Request, exc: Exception):
        return JSONResponse(status_code=status, content={"error": str(exc)})
    return handler


def _unexpected_error(_request: Request, exc: Exception):
    logger.exception("Error no manejado")
    return JSONResponse(status_code=500, content={"error": "Error interno del servidor"})


app.add_exception_handler(QueryError, _known_error(400))
app.add_exception_handler(TableNotFoundError, _known_error(404))
app.add_exception_handler(ColumnNotFoundError, _known_error(404))
app.add_exception_handler(TableAlreadyExistsError, _known_error(409))
app.add_exception_handler(UniqueConstraintError, _known_error(409))
app.add_exception_handler(ValueError, _known_error(400))
app.add_exception_handler(Exception, _unexpected_error)
