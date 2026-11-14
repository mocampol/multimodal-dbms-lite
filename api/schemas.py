from typing import Any, Literal

from pydantic import BaseModel, Field


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
    index_type: Literal["btree", "hash", "rtree", "gin", "spimi", "ivf", "hnsw"]
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


class MediaImportRequest(BaseModel):
    directory: str
    table_name: str
    kind: Literal["image", "audio", "text"]
    index: Literal["IVF", "HNSW", "SPIMI"] | None = None


class TableRowsInsertRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1)


class CreateIndexRequest(BaseModel):
    column_name: str
    index_type: Literal["btree", "hash", "rtree", "gin", "spimi", "ivf", "hnsw"]
    index_name: str | None = None
    dim: int | None = Field(default=None, gt=0)
    index_params: dict[str, Any] | None = None


class VectorSearchRequest(BaseModel):
    column: str | None = None
    vector: list[float] | None = None
    vector_b64: str | None = None
    k: int = Field(default=5, gt=0)
    index_type: Literal["ivf", "hnsw"] = "hnsw"
    metric: Literal["cosine", "euclidean", "dot_product"] = "cosine"
    ef_search: int | None = Field(default=None, gt=0)
    n_probe: int | None = Field(default=None, gt=0)
