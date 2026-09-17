from fastapi import APIRouter

from engine import catalog

router = APIRouter()


@router.get("/tables")
def list_tables():
    return [
        {
            "name": name,
            "storage_type": tm.storage_type.value,
            "columns": [
                {"name": c.name, "type": c.data_type.value, "is_primary_key": c.is_primary_key}
                for c in tm.schema.columns
            ],
        }
        for name, tm in catalog.tables.items()
    ]


@router.get("/tables/{table_name}")
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
        "indexes": catalog.get_indexes(table_name),
    }
