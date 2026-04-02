"""
验证数据准备结果的正确性。
检查生成的文件格式是否符合 OmniColPress 框架要求。
"""

import json
import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def check(condition: bool, msg: str):
    global CHECKS_PASSED, CHECKS_FAILED
    if condition:
        CHECKS_PASSED += 1
        print(f"  ✓ {msg}")
    else:
        CHECKS_FAILED += 1
        print(f"  ✗ {msg}")


def read_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print("=" * 60)
    print("数据验证")
    print("=" * 60)

    # 1. 检查文件存在
    print("\n[文件存在性]")
    expected_files = [
        "train_corpus.jsonl",
        "test_corpus.jsonl",
        "train.jsonl",
        "test_queries.csv",
        "test_qrels.jsonl",
        "stats.json",
    ]
    for fname in expected_files:
        check((DATA_DIR / fname).exists(), f"{fname} exists")

    if CHECKS_FAILED > 0:
        print("\n文件缺失，请先运行 prepare_data.py")
        return

    # 2. 检查 corpus 格式
    print("\n[训练语料 train_corpus.jsonl]")
    train_corpus = read_jsonl(DATA_DIR / "train_corpus.jsonl")
    check(len(train_corpus) == 7010, f"count = {len(train_corpus)} (expected 7010)")
    check(all("docid" in d and "text" in d for d in train_corpus), "all entries have 'docid' and 'text'")
    check(all(d["text"].strip() for d in train_corpus), "no empty text")
    sample = train_corpus[0]
    check(sample["docid"].startswith("video"), f"sample docid = {sample['docid']}")
    check(len(sample["text"]) > 50, f"sample text length = {len(sample['text'])}")

    print("\n[测试语料 test_corpus.jsonl]")
    test_corpus = read_jsonl(DATA_DIR / "test_corpus.jsonl")
    check(len(test_corpus) == 2990, f"count = {len(test_corpus)} (expected 2990)")
    check(all("docid" in d and "text" in d for d in test_corpus), "all entries have 'docid' and 'text'")
    # 检查 test video 范围
    test_docids = {d["docid"] for d in test_corpus}
    check("video7010" in test_docids, "video7010 in test corpus")
    check("video9999" in test_docids, "video9999 in test corpus")
    check("video0" not in test_docids, "video0 NOT in test corpus")

    # 3. 检查训练数据格式
    print("\n[训练数据 train.jsonl]")
    train_data = read_jsonl(DATA_DIR / "train.jsonl")
    check(len(train_data) > 100000, f"count = {len(train_data)} (expected ~140k)")
    required_keys = {"query_id", "query", "positive_document_ids", "negative_document_ids"}
    check(all(required_keys.issubset(d.keys()) for d in train_data), "all entries have required keys")
    check(all(len(d["positive_document_ids"]) == 1 for d in train_data), "each query has exactly 1 positive doc")
    # 检查正样本都在训练语料中
    train_docids = {d["docid"] for d in train_corpus}
    sample_pos = {d["positive_document_ids"][0] for d in train_data[:1000]}
    check(sample_pos.issubset(train_docids), "positive doc ids exist in train corpus")

    # 4. 检查测试查询
    print("\n[测试查询 test_queries.csv]")
    with open(DATA_DIR / "test_queries.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        queries = list(reader)
    check(len(queries) == 1000, f"count = {len(queries)} (expected 1000)")
    check(all("query_id" in q and "query" in q for q in queries), "all entries have 'query_id' and 'query'")
    check(all(q["query"].strip() for q in queries), "no empty queries")

    # 5. 检查 qrels
    print("\n[相关性标注 test_qrels.jsonl]")
    qrels = read_jsonl(DATA_DIR / "test_qrels.jsonl")
    check(len(qrels) == 1000, f"count = {len(qrels)} (expected 1000)")
    check(all("query_id" in q and "document_id" in q and "relevance" in q for q in qrels), "all entries have required keys")
    # 检查 qrels 中的 document_id 都在测试语料中
    qrel_docids = {q["document_id"] for q in qrels}
    check(qrel_docids.issubset(test_docids), "all qrel doc ids exist in test corpus")
    # 检查 query_id 对应关系
    query_ids_from_csv = {q["query_id"] for q in queries}
    query_ids_from_qrels = {q["query_id"] for q in qrels}
    check(query_ids_from_csv == query_ids_from_qrels, "query_ids match between queries and qrels")

    # 6. 检查无重叠
    print("\n[数据隔离]")
    check(len(train_docids & test_docids) == 0, "train and test corpus have no overlap")

    # 总结
    print("\n" + "=" * 60)
    print(f"验证完成: {CHECKS_PASSED} passed, {CHECKS_FAILED} failed")
    if CHECKS_FAILED == 0:
        print("所有检查通过！数据可用于 OmniColPress 训练。")
    else:
        print("存在问题，请检查数据准备脚本。")
    print("=" * 60)


if __name__ == "__main__":
    main()
