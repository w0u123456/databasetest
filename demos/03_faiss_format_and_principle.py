"""
FAISS data format and principle demo.

FAISS 不是传统“业务数据库”，更准确地说是向量检索库：
1. 数据通常是 N x D 的 float32 矩阵。
2. 查询是 1 x D 或 Q x D 的 float32 矩阵。
3. 输出是 top-k 的向量 id 和距离/相似度。
4. 常见索引：
   - IndexFlatL2: 暴力精确搜索，逐个计算距离。
   - IVF: 先聚类成倒排桶，只查离 query 最近的几个桶。
   - PQ: 把向量压缩成 code，牺牲精度换内存和速度。

这个 demo 用纯 Python 实现小型 Flat 和 IVF，帮助理解 FAISS 的数据格式和索引原理。
真实 FAISS 会用 C++/SIMD/GPU 加速，数据格式仍然围绕 float32 向量矩阵和 index。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import struct


Vector = list[float]


def l2_distance(a: Vector, b: Vector) -> float:
    """Squared L2 distance, same family as FAISS IndexFlatL2."""
    return sum((x - y) ** 2 for x, y in zip(a, b))


def dot(a: Vector, b: Vector) -> float:
    """Inner product. Cosine similarity usually means normalize first, then dot."""
    return sum(x * y for x, y in zip(a, b))


def normalize(v: Vector) -> Vector:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def pack_float32_matrix(vectors: list[Vector]) -> bytes:
    """Show the physical shape of a vector matrix.

    FAISS expects contiguous numeric memory, commonly float32.
    Python float is float64 internally, so we pack as little-endian float32 bytes.
    """
    if not vectors:
        return b""
    dim = len(vectors[0])
    flat = [x for vector in vectors for x in vector]
    return struct.pack("<" + "f" * (len(vectors) * dim), *flat)


class FlatIndex:
    """Exact search: store all vectors and scan all vectors for every query."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: list[int] = []
        self.vectors: list[Vector] = []

    def add(self, ids: list[int], vectors: list[Vector]) -> None:
        for vector in vectors:
            assert len(vector) == self.dim
        self.ids.extend(ids)
        self.vectors.extend(vectors)

    def search_l2(self, query: Vector, k: int) -> list[tuple[int, float]]:
        scored = [(idx, l2_distance(query, vector)) for idx, vector in zip(self.ids, self.vectors)]
        return sorted(scored, key=lambda item: item[1])[:k]

    def search_ip(self, query: Vector, k: int) -> list[tuple[int, float]]:
        scored = [(idx, dot(query, vector)) for idx, vector in zip(self.ids, self.vectors)]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:k]


@dataclass
class IVFIndex:
    """Tiny IVF index.

    IVF = Inverted File index.
    训练阶段：用 centroid 把空间分成多个 cluster。
    add 阶段：每条向量进入最近 centroid 对应的 inverted list。
    search 阶段：query 只访问最近的 nprobe 个 inverted list。
    """

    centroids: list[Vector]
    inverted_lists: dict[int, list[tuple[int, Vector]]]

    @classmethod
    def train_with_fixed_centroids(cls, centroids: list[Vector]) -> "IVFIndex":
        return cls(centroids=centroids, inverted_lists={i: [] for i in range(len(centroids))})

    def _nearest_centroids(self, vector: Vector, nprobe: int) -> list[int]:
        scored = [(i, l2_distance(vector, centroid)) for i, centroid in enumerate(self.centroids)]
        return [i for i, _ in sorted(scored, key=lambda item: item[1])[:nprobe]]

    def add(self, ids: list[int], vectors: list[Vector]) -> None:
        for idx, vector in zip(ids, vectors):
            bucket = self._nearest_centroids(vector, nprobe=1)[0]
            self.inverted_lists[bucket].append((idx, vector))

    def search(self, query: Vector, k: int, nprobe: int) -> list[tuple[int, float]]:
        candidate_buckets = self._nearest_centroids(query, nprobe=nprobe)
        candidates = []
        for bucket in candidate_buckets:
            candidates.extend(self.inverted_lists[bucket])

        scored = [(idx, l2_distance(query, vector)) for idx, vector in candidates]
        return sorted(scored, key=lambda item: item[1])[:k]


def make_demo_vectors() -> tuple[list[int], list[Vector]]:
    random.seed(7)
    ids = []
    vectors = []
    centers = [(-2.0, -2.0), (2.0, 2.0), (2.0, -2.0)]
    next_id = 100
    for cx, cy in centers:
        for _ in range(5):
            ids.append(next_id)
            vectors.append([cx + random.uniform(-0.35, 0.35), cy + random.uniform(-0.35, 0.35)])
            next_id += 1
    return ids, vectors


def main() -> None:
    ids, vectors = make_demo_vectors()
    query = [2.1, 1.8]

    print("1) 向量数据格式：N x D float32 matrix")
    print(f"N={len(vectors)}, D={len(vectors[0])}")
    print("first 3 rows:", vectors[:3])
    matrix_bytes = pack_float32_matrix(vectors)
    print(f"float32 bytes length = N * D * 4 = {len(matrix_bytes)}")
    print("first 24 bytes:", matrix_bytes[:24].hex(" "))

    print("\n2) Flat 精确搜索：逐个算距离")
    flat = FlatIndex(dim=2)
    flat.add(ids, vectors)
    print("query:", query)
    print("top-3 L2:", flat.search_l2(query, k=3))

    print("\n3) 内积与 cosine：cosine 通常先归一化，再做 inner product")
    normalized_index = FlatIndex(dim=2)
    normalized_index.add(ids, [normalize(v) for v in vectors])
    print("top-3 cosine-like:", normalized_index.search_ip(normalize(query), k=3))

    print("\n4) IVF 近似搜索：先找桶，再在桶内搜索")
    ivf = IVFIndex.train_with_fixed_centroids(
        centroids=[[-2.0, -2.0], [2.0, 2.0], [2.0, -2.0]]
    )
    ivf.add(ids, vectors)
    for bucket, items in ivf.inverted_lists.items():
        print(f"bucket {bucket}, centroid={ivf.centroids[bucket]}, ids={[idx for idx, _ in items]}")

    print("IVF search nprobe=1:", ivf.search(query, k=3, nprobe=1))
    print("IVF search nprobe=2:", ivf.search(query, k=3, nprobe=2))

    print(
        "\n核心记忆：FAISS = float32 向量矩阵 + 距离函数 + 索引结构。"
        "Flat 精确但慢，IVF/PQ/HNSW 等用召回率换速度和内存。"
    )


if __name__ == "__main__":
    main()

