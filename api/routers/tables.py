from fastapi import APIRouter

from engine import catalog
from schemas import TableDetail, TableSummary

router = APIRouter()


@router.get("/tables", response_model=list[TableSummary])
def list_tables():
    return [
        {
            "name": name,
            "storage_type": tm.storage_type.value,
            "columns": [
                {"name": c.name, "type": c.data_type.value, "is_primary_key": c.is_primary_key}
                for c in tm.schema.columns
            ],
            "indexes": _index_metadata(name),
        }
        for name, tm in catalog.tables.items()
    ]


@router.get("/tables/{table_name}", response_model=TableDetail)
def get_table(table_name: str):
    schema = catalog.get_schema(table_name)
    return {
        "name": table_name,
        "storage_type": catalog.get_table(table_name).storage_type.value,
        "columns": [
            {
                "name": c.name,
                "type": c.data_type.value,
                "size": c.size,
                "nullable": c.nullable,
                "is_unique": c.is_unique,
                "is_primary_key": c.is_primary_key,
            }
            for c in schema.columns
        ],
        "indexes": _index_metadata(table_name),
    }


def _index_metadata(table_name: str) -> list[dict]:
    return [
        {
            **entry,
            "is_spatial": entry["index_type"] == "rtree",
        }
        for entry in catalog.get_indexes(table_name)
    ]
