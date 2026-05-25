"""
SQLite data format and principle demo.

SQLite 的核心特点：
1. 整个数据库通常就是一个普通文件，例如 artifacts/learn_sqlite.db。
2. 文件被切成固定大小的 page，默认常见大小是 4096 字节。
3. 表和索引主要使用 B-tree 存储。
4. 一行数据不是简单的 CSV，而是 SQLite record format：
   - record header: 每列的 serial type，也就是类型/长度描述。
   - record body: 每列真实字节。

这个 demo 会：
1. 创建真实 SQLite 数据库。
2. 插入几行不同类型的数据。
3. 读取 .db 二进制文件头。
4. 找到 user table 的 root page。
5. 手动解析 table leaf B-tree page 上的 cell 和 record。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import struct


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DB_PATH = ARTIFACTS / "learn_sqlite.db"


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Read SQLite varint.

    SQLite 在很多地方使用变长整数：
    - 小数字用 1 个字节。
    - 大数字最多用 9 个字节。

    返回值是 (value, next_offset)。
    """
    value = 0
    for i in range(9):
        byte = data[offset + i]
        if i == 8:
            value = (value << 8) | byte
            return value, offset + 9

        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset + i + 1

    raise ValueError("invalid SQLite varint")


def decode_serial_value(serial_type: int, data: bytes, offset: int) -> tuple[object, int]:
    """Decode one SQLite record field by serial type.

    常见 serial type：
    - 0: NULL
    - 1..6: 不同长度的整数
    - 7: 8 字节 float64
    - 8/9: 常量整数 0/1，不占 body 空间
    - >= 12: BLOB/TEXT，偶数是 BLOB，奇数是 TEXT
    """
    if serial_type == 0:
        return None, offset
    if serial_type == 1:
        return int.from_bytes(data[offset : offset + 1], "big", signed=True), offset + 1
    if serial_type == 2:
        return int.from_bytes(data[offset : offset + 2], "big", signed=True), offset + 2
    if serial_type == 3:
        return int.from_bytes(data[offset : offset + 3], "big", signed=True), offset + 3
    if serial_type == 4:
        return int.from_bytes(data[offset : offset + 4], "big", signed=True), offset + 4
    if serial_type == 5:
        return int.from_bytes(data[offset : offset + 6], "big", signed=True), offset + 6
    if serial_type == 6:
        return int.from_bytes(data[offset : offset + 8], "big", signed=True), offset + 8
    if serial_type == 7:
        return struct.unpack(">d", data[offset : offset + 8])[0], offset + 8
    if serial_type == 8:
        return 0, offset
    if serial_type == 9:
        return 1, offset
    if serial_type >= 12:
        length = (serial_type - 12) // 2
        raw = data[offset : offset + length]
        if serial_type % 2 == 0:
            return raw, offset + length
        return raw.decode("utf-8"), offset + length

    raise ValueError(f"unsupported serial type: {serial_type}")


@dataclass
class SQLiteHeader:
    signature: bytes
    page_size: int
    write_version: int
    read_version: int
    page_count: int
    text_encoding: int


def create_database() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA page_size = 4096")
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute(
            """
            CREATE TABLE notes (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                score REAL,
                body TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO notes(title, score, body) VALUES (?, ?, ?)",
            [
                ("sqlite file", 9.5, "single-file database"),
                ("btree page", 8.0, "rows live inside B-tree pages"),
                ("record format", 9.0, "header serial types + body bytes"),
            ],
        )
        conn.commit()


def read_header(db_bytes: bytes) -> SQLiteHeader:
    signature = db_bytes[:16]
    page_size_raw = int.from_bytes(db_bytes[16:18], "big")
    # SQLite 特例：值 1 表示 65536 字节 page。
    page_size = 65536 if page_size_raw == 1 else page_size_raw
    return SQLiteHeader(
        signature=signature,
        page_size=page_size,
        write_version=db_bytes[18],
        read_version=db_bytes[19],
        page_count=int.from_bytes(db_bytes[28:32], "big"),
        text_encoding=int.from_bytes(db_bytes[56:60], "big"),
    )


def find_table_root_page() -> int:
    """Use SQL to ask sqlite_schema where our table root B-tree page is."""
    with sqlite3.connect(DB_PATH) as conn:
        (root_page,) = conn.execute(
            "SELECT rootpage FROM sqlite_schema WHERE type = 'table' AND name = 'notes'"
        ).fetchone()
    return int(root_page)


def parse_table_leaf_page(page: bytes, page_no: int) -> list[int]:
    """Parse a table leaf B-tree page and return cell offsets.

    Table leaf page header layout:
    - byte 0: page type, 0x0D means table leaf page
    - byte 1..2: first freeblock offset
    - byte 3..4: number of cells
    - byte 5..6: start of cell content area
    - byte 7: fragmented free bytes
    - then cell pointer array, each pointer is 2 bytes
    """
    page_type = page[0]
    first_freeblock = int.from_bytes(page[1:3], "big")
    cell_count = int.from_bytes(page[3:5], "big")
    cell_content_start = int.from_bytes(page[5:7], "big")
    fragmented_free_bytes = page[7]

    print(f"\n[B-tree page {page_no}]")
    print(f"page type             : 0x{page_type:02X} (0x0D = table leaf)")
    print(f"first freeblock offset: {first_freeblock}")
    print(f"cell count            : {cell_count}")
    print(f"cell content starts at: {cell_content_start}")
    print(f"fragmented free bytes : {fragmented_free_bytes}")

    offsets = []
    for i in range(cell_count):
        pointer_offset = 8 + i * 2
        cell_offset = int.from_bytes(page[pointer_offset : pointer_offset + 2], "big")
        offsets.append(cell_offset)
        print(f"cell pointer[{i}]      : page byte offset {cell_offset}")
    return offsets


def parse_table_leaf_cell(page: bytes, cell_offset: int) -> tuple[int, list[object]]:
    """Parse one table leaf cell.

    Table leaf cell format:
    - varint payload_size: record 总字节数
    - varint rowid: INTEGER PRIMARY KEY 会成为 rowid
    - payload: SQLite record format
    """
    payload_size, offset = read_varint(page, cell_offset)
    rowid, offset = read_varint(page, offset)
    payload = page[offset : offset + payload_size]

    header_size, header_offset = read_varint(payload, 0)
    serial_types = []
    while header_offset < header_size:
        serial_type, header_offset = read_varint(payload, header_offset)
        serial_types.append(serial_type)

    body_offset = header_size
    values = []
    for serial_type in serial_types:
        value, body_offset = decode_serial_value(serial_type, payload, body_offset)
        values.append(value)

    return rowid, values


def main() -> None:
    create_database()

    print("1) SQL 逻辑视角：表、列、行")
    with sqlite3.connect(DB_PATH) as conn:
        for row in conn.execute("SELECT id, title, score, body FROM notes ORDER BY id"):
            print(row)

    db_bytes = DB_PATH.read_bytes()
    header = read_header(db_bytes)

    print("\n2) SQLite 文件头：数据库首先是一个二进制文件")
    print(f"path         : {DB_PATH}")
    print(f"signature    : {header.signature!r}")
    print(f"page size    : {header.page_size} bytes")
    print(f"write/read   : {header.write_version}/{header.read_version}")
    print(f"page count   : {header.page_count}")
    print(f"text encoding: {header.text_encoding} (1=UTF-8)")

    root_page = find_table_root_page()
    page_start = (root_page - 1) * header.page_size
    page_end = page_start + header.page_size
    root_page_bytes = db_bytes[page_start:page_end]

    print("\n3) notes 表的位置：sqlite_schema 记录了每张表的 root page")
    print(f"notes root page: {root_page}")

    cell_offsets = parse_table_leaf_page(root_page_bytes, root_page)

    print("\n4) 手动解析 cell：从 B-tree cell 还原为行")
    print("注意：id 是 INTEGER PRIMARY KEY，所以它是 rowid；record 里 id 列本身显示为 NULL。")
    for cell_offset in cell_offsets:
        rowid, values = parse_table_leaf_cell(root_page_bytes, cell_offset)
        print(f"rowid={rowid}, record_columns={values}")

    print("\n核心记忆：SQLite = SQL 层 + page cache + B-tree + record 编码 + 单文件持久化。")


if __name__ == "__main__":
    main()

