from typing import Literal

from pydantic import BaseModel


class QueryRequest(BaseModel):
    sql: str


class PointCoordinates(BaseModel):
    longitude: float
    latitude: float


class DistanceValue(BaseModel):
    value: float
    unit: Literal["meters", "coordinate_units"]


class IndexInfo(BaseModel):
    index_id: int
    column_name: str
    index_type: Literal["btree", "hash", "rtree"]
    root_page_id: int | None = None
    is_spatial: bool


class TableColumnSummary(BaseModel):
    name: str
    type: str
    is_primary_key: bool


class TableColumnDetail(TableColumnSummary):
    size: int | None = None
    nullable: bool
    is_unique: bool


class TableSummary(BaseModel):
    name: str
    storage_type: str
    columns: list[TableColumnSummary]
    indexes: list[IndexInfo]


class TableDetail(BaseModel):
    name: str
    storage_type: str
    columns: list[TableColumnDetail]
    indexes: list[IndexInfo]


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
