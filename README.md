# Database format and principle demos

这是一组“边跑边看”的数据库学习 demo，覆盖：

- `SQLite`: 单文件关系型数据库，重点看文件头、page、B-tree、record 编码。
- `Redis`: 内存 KV 数据库，重点看 key/value 类型、对象编码、过期、AOF/RDB 思路。
- `FAISS`: 向量检索库，重点看向量矩阵、距离度量、Flat/IVF/PQ 的索引思想。
- `Milvus`: 向量数据库，重点看 collection/schema/segment/index/query pipeline。

所有脚本默认只依赖 Python 标准库。这样你可以先理解格式和原理，再决定是否安装真实 Redis、FAISS、Milvus。

## Run

```powershell
python .\run_all.py
```

或者单独运行：

```powershell
python .\demos\01_sqlite_format_and_principle.py
python .\demos\02_redis_format_and_principle.py
python .\demos\03_faiss_format_and_principle.py
python .\demos\04_milvus_format_and_principle.py
```

运行后会在 `artifacts/` 下生成一些可观察文件，例如 SQLite 的 `.db` 文件和 Redis 风格的 AOF 日志。

## Learning path

建议顺序：

1. 先跑 SQLite demo，看一个数据库文件如何被拆成 page、B-tree、record。
2. 再跑 Redis demo，看内存对象如何根据数据类型选择不同编码。
3. 再跑 FAISS demo，看向量搜索为什么离不开矩阵、距离和索引。
4. 最后跑 Milvus demo，看一个“向量数据库”如何把向量索引、标量过滤、分段存储、collection schema 组合成服务。

这四者的定位可以粗略记成：

| 系统 | 核心定位 | 典型数据格式 | 核心原理 |
| --- | --- | --- | --- |
| SQLite | 嵌入式关系型数据库 | 单个 `.db` 文件，page + B-tree + record | SQL 层把行记录组织进 B-tree 页 |
| Redis | 内存 KV 数据库 | key -> typed object，外加 AOF/RDB 持久化 | 用内存结构换极低延迟 |
| FAISS | 向量相似度检索库 | float32 向量矩阵 + index | 用距离计算和近似索引加速 top-k |
| Milvus | 向量数据库服务 | collection/schema/segment/index | 在 FAISS/HNSW/IVF 等索引之上加数据管理和查询服务 |

