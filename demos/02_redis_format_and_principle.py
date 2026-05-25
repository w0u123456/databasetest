"""
Redis data format and principle demo.

Redis 的核心特点：
1. 它首先是内存中的 key-value 数据库。
2. key 通常是字符串，value 是 typed object：string/list/hash/set/zset/stream 等。
3. 每个 value 类型内部还会选择不同 encoding，例如：
   - string: int / embstr / raw
   - list: quicklist
   - hash: listpack / hashtable
   - zset: listpack / skiplist + dict
4. 持久化不是“每次直接改磁盘 page”，常见方式是：
   - AOF: 追加写命令日志。
   - RDB: 某一刻的内存快照。

这个 demo 不需要 redis-server。它用一个极小的 MiniRedis 模拟：
- key -> RedisObject
- TTL 过期
- String/List/Hash/ZSet 的基本编码思想
- AOF append-only log 的样子
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import bisect
import json
import time


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
AOF_PATH = ARTIFACTS / "mini_redis.aof"


@dataclass
class RedisObject:
    type_name: str
    encoding: str
    value: object
    expires_at: float | None = None


class MiniRedis:
    def __init__(self, aof_path: Path) -> None:
        self.store: dict[str, RedisObject] = {}
        self.aof_path = aof_path
        self.aof_path.parent.mkdir(exist_ok=True)
        self.aof_path.write_text("", encoding="utf-8")

    def _append_aof(self, command: list[object]) -> None:
        """Append one command to an AOF-like log.

        真实 Redis AOF 使用 RESP 协议文本追加命令。
        为了好读，这里用 JSON Lines 表示同样的“追加命令”思想。
        """
        with self.aof_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(command, ensure_ascii=False) + "\n")

    def _choose_string_encoding(self, value: str) -> str:
        """Roughly mimic Redis string encoding decisions."""
        if value.lstrip("-").isdigit():
            return "int"
        if len(value) <= 44:
            return "embstr"
        return "raw"

    def _choose_hash_encoding(self, mapping: dict[str, str]) -> str:
        """Small hashes can be packed; larger hashes become hash tables."""
        total_bytes = sum(len(k) + len(v) for k, v in mapping.items())
        if len(mapping) <= 4 and total_bytes <= 128:
            return "listpack"
        return "hashtable"

    def _choose_zset_encoding(self, items: list[tuple[float, str]]) -> str:
        """Small sorted sets can be packed; larger sorted sets need skiplist-like indexes."""
        if len(items) <= 4 and sum(len(member) for _, member in items) <= 128:
            return "listpack"
        return "skiplist+dict"

    def _get_live(self, key: str) -> RedisObject | None:
        obj = self.store.get(key)
        if obj is None:
            return None
        if obj.expires_at is not None and time.time() >= obj.expires_at:
            del self.store[key]
            return None
        return obj

    def set(self, key: str, value: str, ex_seconds: int | None = None) -> None:
        expires_at = None if ex_seconds is None else time.time() + ex_seconds
        self.store[key] = RedisObject(
            type_name="string",
            encoding=self._choose_string_encoding(value),
            value=value,
            expires_at=expires_at,
        )
        command = ["SET", key, value] if ex_seconds is None else ["SET", key, value, "EX", ex_seconds]
        self._append_aof(command)

    def lpush(self, key: str, *values: str) -> None:
        obj = self._get_live(key)
        if obj is None:
            obj = RedisObject(type_name="list", encoding="quicklist", value=[])
            self.store[key] = obj
        assert obj.type_name == "list"
        # Redis LPUSH 把元素从左侧推入，所以后来的值在更前面。
        obj.value[0:0] = reversed(values)
        self._append_aof(["LPUSH", key, *values])

    def hset(self, key: str, mapping: dict[str, str]) -> None:
        obj = self._get_live(key)
        if obj is None:
            obj = RedisObject(type_name="hash", encoding="listpack", value={})
            self.store[key] = obj
        assert obj.type_name == "hash"
        obj.value.update(mapping)
        obj.encoding = self._choose_hash_encoding(obj.value)
        self._append_aof(["HSET", key, mapping])

    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        obj = self._get_live(key)
        if obj is None:
            obj = RedisObject(type_name="zset", encoding="listpack", value=[])
            self.store[key] = obj
        assert obj.type_name == "zset"

        # 用有序列表模拟 zset 的 score 排序。
        by_member = {member: score for score, member in obj.value}
        by_member.update(mapping)
        obj.value = sorted((score, member) for member, score in by_member.items())
        obj.encoding = self._choose_zset_encoding(obj.value)
        self._append_aof(["ZADD", key, mapping])

    def zrange(self, key: str, start: int, stop: int) -> list[str]:
        obj = self._get_live(key)
        if obj is None:
            return []
        assert obj.type_name == "zset"
        pairs = obj.value[start : stop + 1]
        return [member for _, member in pairs]

    def debug_object_table(self) -> None:
        print("key -> object metadata")
        for key, obj in self.store.items():
            live = self._get_live(key)
            if live is None:
                continue
            ttl = None if obj.expires_at is None else round(obj.expires_at - time.time(), 3)
            print(
                f"{key!r}: type={obj.type_name}, encoding={obj.encoding}, "
                f"ttl={ttl}, value={obj.value!r}"
            )


def explain_resp_example() -> None:
    print("\nRESP 协议例子：客户端发送 SET name redis")
    print("*3\\r\\n$3\\r\\nSET\\r\\n$4\\r\\nname\\r\\n$5\\r\\nredis\\r\\n")
    print("含义：数组 3 个元素，每个元素是 bulk string。Redis 网络协议和内存编码是两层东西。")


def main() -> None:
    r = MiniRedis(AOF_PATH)

    print("1) 写入不同类型的 key")
    r.set("counter", "100")
    r.set("short_name", "redis")
    r.set("long_text", "x" * 80)
    r.set("session:1", "token", ex_seconds=60)
    r.lpush("recent", "doc3", "doc2", "doc1")
    r.hset("user:1", {"name": "Ada", "city": "London"})
    r.zadd("rank", {"alice": 10.0, "bob": 8.0, "cindy": 13.0})
    r.zadd("rank", {"dylan": 7.0, "eve": 11.0})

    print("\n2) Redis object table：key 指向 typed object")
    r.debug_object_table()

    print("\n3) ZSET 查询：按 score 排序取 top")
    print("ZRANGE rank 0 2 ->", r.zrange("rank", 0, 2))

    print("\n4) AOF 持久化思想：把写命令追加到日志")
    print(f"AOF path: {AOF_PATH}")
    print(AOF_PATH.read_text(encoding="utf-8").rstrip())

    explain_resp_example()

    print("\n核心记忆：Redis = dict(key -> object) + 类型专用编码 + 事件循环 + 可选 AOF/RDB 持久化。")


if __name__ == "__main__":
    main()

