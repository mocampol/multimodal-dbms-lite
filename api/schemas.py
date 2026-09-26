from typing import Literal

from pydantic import BaseModel


class QueryRequest(BaseModel):
    sql: str


class CsvImportColumn(BaseModel):
    name: str
    type: str
    size: int | None = None
    nullable: bool = True
    is_unique: bool = False
    is_primary_key: bool = False


class CsvImportConfirm(BaseModel):
    upload_id: str
    table_name: str
    columns: list[CsvImportColumn]
    on_error: Literal["stop", "ignore"] = "stop"
