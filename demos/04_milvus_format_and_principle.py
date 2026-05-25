"""
Milvus data format and principle demo.

Milvus 是向量数据库服务，不只是一个向量索引：
1. Collection: 类似表，有 schema。
2. Field: 标量字段和向量字段，例如 id/title/embedding。
3. Segment: 数据分段写入；sealed segment 才适合建立索引。
4. Index: 向量字段上建立 IVF_FLAT/HNSW/DISKANN/PQ 等索引。
5. Query/Search:
   - 可先做标量过滤，例如 tag == "database"。
   - 再做向量 top-k。
   - 最后返回 id、distance 和 output fields。

这个 demo 用纯 Python 模拟一个迷你 Milvus：
- Collection schema
- Insert 到 growing segment
- Flush 形成 sealed segment
- Build index
- Search with scalar filter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math


Vector = list[float]


class DataType(str, Enum):
    INT64 = "INT64"
    VARCHAR = "VARCHAR"
    FLOAT_VECTOR = "FLOAT_VECTOR"


@dataclass(frozen=True)
class FieldSchema:
    name: str
    dtype: DataType
    primary: bool = False
    dim: int | None = None


@dataclass(frozen=True)
class CollectionSchema:
    name: str
    fields: list[FieldSchema]

    def vector_field(self) -> FieldSchema:
        for field_schema in self.fields:
            if field_schema.dtype == DataType.FLOAT_VECTOR:
                return field_schema
        raise ValueError("no vector field")


@dataclass
class Row:
    values: dict[str, object]


@dataclass
class Segment:
    id: int
    state: str
    rows: list[Row] = field(default_factory=list)
    index: "FlatVectorIndex | None" = None


def l2_distance(a: Vector, b: Vector) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


class FlatVectorIndex:
    """A tiny exact vector index for one sealed segment."""

    def __init__(self, vector_field: str) -> None:
        self.vector_field = vector_field
        self.entries: list[tuple[int, Vector, Row]] = []

    def add_rows(self, rows: list[Row]) -> None:
        for offset, row in enumerate(rows):
            self.entries.append((offset, row.values[self.vector_field], row))

    def search(self, query: Vector, k: int, filter_expr: "FilterExpr | None") -> list[tuple[float, Row]]:
        scored = []
        for _, vector, row in self.entries:
            if filter_expr is not None and not filter_expr.match(row):
                continue
            scored.append((l2_distance(query, vector), row))
        return sorted(scored, key=lambda item: item[0])[:k]


@dataclass(frozen=True)
class FilterExpr:
    """A tiny scalar filter.

    真实 Milvus 支持更完整的表达式语法。
    这里用 field == value 表示，例如 tag == "database"。
    """

    field_name: str
    equals: object

    def match(self, row: Row) -> bool:
        return row.values.get(self.field_name) == self.equals


class MiniMilvusCollection:
    def __init__(self, schema: CollectionSchema, segment_max_rows: int = 4) -> None:
        self.schema = schema
        self.segment_max_rows = segment_max_rows
        self.segments: list[Segment] = [Segment(id=1, state="growing")]
        self.next_segment_id = 2

    def _active_segment(self) -> Segment:
        segment = self.segments[-1]
        if segment.state != "growing":
            segment = Segment(id=self.next_segment_id, state="growing")
            self.next_segment_id += 1
            self.segments.append(segment)
        return segment

    def insert(self, row_values: dict[str, object]) -> None:
        """Insert one row after simple schema validation."""
        vector_field = self.schema.vector_field()
        vector = row_values[vector_field.name]
        assert isinstance(vector, list)
        assert len(vector) == vector_field.dim

        segment = self._active_segment()
        segment.rows.append(Row(row_values))

        # Demo rule: segment full 后自动 flush。
        if len(segment.rows) >= self.segment_max_rows:
            self.flush()

    def flush(self) -> None:
        """Seal current growing segment.

        在真实 Milvus 中，growing segment 负责接收写入；
        flush 后变成 sealed segment，随后可构建向量索引并被查询节点加载。
        """
        segment = self._active_segment()
        if segment.rows:
            segment.state = "sealed"

    def build_index(self, index_type: str = "FLAT") -> None:
        vector_field = self.schema.vector_field()
        for segment in self.segments:
            if segment.state == "sealed" and segment.index is None:
                if index_type != "FLAT":
                    raise ValueError("this demo only implements FLAT")
                index = FlatVectorIndex(vector_field=vector_field.name)
                index.add_rows(segment.rows)
                segment.index = index

    def search(
        self,
        query: Vector,
        k: int,
        filter_expr: FilterExpr | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, object]]:
        results = []
        vector_field = self.schema.vector_field()

        for segment in self.segments:
            if segment.state == "growing":
                # Growing segment 还没有索引；真实系统通常走 brute force 搜索。
                rows = [
                    row
                    for row in segment.rows
                    if filter_expr is None or filter_expr.match(row)
                ]
                results.extend((l2_distance(query, row.values[vector_field.name]), row) for row in rows)
            elif segment.index is not None:
                results.extend(segment.index.search(query, k=k, filter_expr=filter_expr))

        fields = output_fields or []
        top = sorted(results, key=lambda item: item[0])[:k]
        return [
            {
                "distance": round(distance, 4),
                **{field_name: row.values[field_name] for field_name in fields},
            }
            for distance, row in top
        ]

    def debug_layout(self) -> None:
        print(f"collection: {self.schema.name}")
        for field_schema in self.schema.fields:
            dim = "" if field_schema.dim is None else f", dim={field_schema.dim}"
            primary = ", primary" if field_schema.primary else ""
            print(f"  field {field_schema.name}: {field_schema.dtype.value}{primary}{dim}")

        print("segments:")
        for segment in self.segments:
            print(
                f"  segment {segment.id}: state={segment.state}, "
                f"rows={len(segment.rows)}, index={'yes' if segment.index else 'no'}"
            )


def embed_tiny(text: str) -> Vector:
    """Toy embedding function.

    真实系统会用模型把文本转为几百或几千维向量。
    这里为了可读性只用 3 维：
    - database-ish
    - vector-ish
    - cache-ish
    """
    text = text.lower()
    database_score = sum(word in text for word in ["database", "sql", "table", "milvus"])
    vector_score = sum(word in text for word in ["vector", "embedding", "search", "faiss"])
    cache_score = sum(word in text for word in ["cache", "redis", "memory"])
    norm = math.sqrt(database_score**2 + vector_score**2 + cache_score**2) or 1.0
    return [database_score / norm, vector_score / norm, cache_score / norm]


def main() -> None:
    schema = CollectionSchema(
        name="doc_chunks",
        fields=[
            FieldSchema("id", DataType.INT64, primary=True),
            FieldSchema("title", DataType.VARCHAR),
            FieldSchema("tag", DataType.VARCHAR),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=3),
        ],
    )
    collection = MiniMilvusCollection(schema=schema, segment_max_rows=3)

    docs = [
        (1, "SQLite stores rows in B-tree pages", "database"),
        (2, "Redis is an in-memory cache and data store", "cache"),
        (3, "FAISS searches dense vectors", "vector"),
        (4, "Milvus manages vector collections and indexes", "database"),
        (5, "Embedding search returns nearest neighbors", "vector"),
    ]

    print("1) Collection/schema：Milvus 像表一样管理字段")
    for doc_id, title, tag in docs:
        collection.insert(
            {
                "id": doc_id,
                "title": title,
                "tag": tag,
                "embedding": embed_tiny(title),
            }
        )

    # 确保最后一个未满 segment 也 sealed，方便建索引。
    collection.flush()
    collection.build_index(index_type="FLAT")
    collection.debug_layout()

    print("\n2) Search：向量 top-k")
    query = embed_tiny("vector search database")
    print(collection.search(query=query, k=3, output_fields=["id", "title", "tag"]))

    print("\n3) Search + scalar filter：先过滤 tag，再做向量 top-k")
    print(
        collection.search(
            query=query,
            k=3,
            filter_expr=FilterExpr(field_name="tag", equals="database"),
            output_fields=["id", "title", "tag"],
        )
    )

    print(
        "\n核心记忆：Milvus = collection schema + segment 管理 + 向量索引 + 标量过滤 + 分布式服务层。"
        "它通常会调用 FAISS/HNSW 等索引能力，但额外负责数据生命周期和查询系统。"
    )


if __name__ == "__main__":
    main()

