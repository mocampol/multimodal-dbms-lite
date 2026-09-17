from pydantic import BaseModel


class QueryRequest(BaseModel):
    sql: str
