from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent
DEMOS = [
    ROOT / "demos" / "01_sqlite_format_and_principle.py",
    ROOT / "demos" / "02_redis_format_and_principle.py",
    ROOT / "demos" / "03_faiss_format_and_principle.py",
    ROOT / "demos" / "04_milvus_format_and_principle.py",
]


def main() -> None:
    for demo in DEMOS:
        print("\n" + "=" * 88)
        print(f"RUN {demo.relative_to(ROOT)}")
        print("=" * 88)
        runpy.run_path(str(demo), run_name="__main__")


if __name__ == "__main__":
    main()

