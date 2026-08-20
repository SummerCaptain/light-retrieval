# -*- coding: utf-8 -*-
"""
DocParser 单元测试

验证文档解析、分块、Hash 索引、文件系统存储的核心逻辑。
运行方式: python -m pytest test/test_doc_parser.py -v
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.doc_parser import DocParser, DocParseResult


class TestDocParser(unittest.TestCase):
    """DocParser 单元测试"""

    def setUp(self):
        """每个测试用例运行前创建临时工作目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.parser = DocParser(
            chunk_size=512,
            chunk_overlap=64,
            output_dir=self.temp_dir,
        )

    def tearDown(self):
        """每个测试用例运行后清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_temp_file(self, content: str, suffix: str = ".txt") -> str:
        """在临时目录创建测试文件，返回文件路径"""
        fd, path = tempfile.mkstemp(dir=self.temp_dir, suffix=suffix)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    # ================================================================
    # 测试1: 文本分块 - 基本功能
    # ================================================================
    def test_chunk_basic(self):
        """测试基本文本分块：短文本应产生单个 chunk"""
        content = "这是一段很短的文本，用于测试基本分块功能。"
        file_path = self._create_temp_file(content)

        result = self.parser.parse_file(file_path)

        self.assertIsInstance(result, DocParseResult)
        self.assertEqual(result.total_chunks, 1)
        self.assertEqual(result.chunks[0], content)
        self.assertGreater(len(result.doc_hash), 0)

    # ================================================================
    # 测试2: 文本分块 - 长文本自动切分
    # ================================================================
    def test_chunk_long_text(self):
        """测试长文本分块：应产生多个 chunk 且每个不超限"""
        # 生成 2000 字的文本，确保 > chunk_size(512) 产生多个 chunk
        content = "这是一段用于测试的文本。" * 200  # 约 2000 字
        file_path = self._create_temp_file(content)

        result = self.parser.parse_file(file_path)

        self.assertGreater(result.total_chunks, 1)
        # 每个 chunk 不应超过 chunk_size + overlap
        for chunk in result.chunks:
            self.assertLessEqual(len(chunk), self.parser.chunk_size + self.parser.chunk_overlap)

    # ================================================================
    # 测试3: 分块重叠 - 相邻 chunk 存在重叠内容
    # ================================================================
    def test_chunk_overlap(self):
        """测试相邻 chunk 之间的重叠逻辑"""
        # 构造一段可预测的文本，方便验证重叠
        unique_words = [f"第{i}段" for i in range(50)]
        content = "。".join(unique_words)
        file_path = self._create_temp_file(content)

        result = self.parser.parse_file(file_path)

        if result.total_chunks >= 2:
            # 验证 chunk1 的尾部与 chunk2 的头部有重叠
            chunk1 = result.chunks[0]
            chunk2 = result.chunks[1]
            # 取 chunk1 末尾的 overlap 长度子串与 chunk2 开头比较
            overlap_len = min(self.parser.chunk_overlap, len(chunk1), len(chunk2))
            if overlap_len > 0:
                tail = chunk1[-overlap_len:]
                head = chunk2[:overlap_len]
                self.assertEqual(tail, head, "相邻 chunk 应存在重叠内容")

    # ================================================================
    # 测试4: Hash 索引 - 确定性
    # ================================================================
    def test_hash_deterministic(self):
        """测试相同内容产生相同 Hash"""
        content = "相同的文档内容，应该产生相同的哈希值。"
        file1 = self._create_temp_file(content)
        file2 = self._create_temp_file(content)

        result1 = self.parser.parse_file(file1)
        result2 = self.parser.parse_file(file2)

        self.assertEqual(result1.doc_hash, result2.doc_hash,
                         "相同内容必须产生相同 Hash")

    # ================================================================
    # 测试5: Hash 索引 - 碰撞避免
    # ================================================================
    def test_hash_unique(self):
        """测试不同内容产生不同 Hash"""
        content_a = "这是文档A的内容。"
        content_b = "这是文档B的内容，与A不同。"
        file_a = self._create_temp_file(content_a)
        file_b = self._create_temp_file(content_b)

        result_a = self.parser.parse_file(file_a)
        result_b = self.parser.parse_file(file_b)

        self.assertNotEqual(result_a.doc_hash, result_b.doc_hash,
                            "不同内容必须产生不同 Hash")

    # ================================================================
    # 测试6: 文件系统存储 - 分块写入
    # ================================================================
    def test_chunks_written_to_disk(self):
        """测试分块文件是否正确写入磁盘"""
        content = "测试文件系统存储。" * 100
        file_path = self._create_temp_file(content)

        result = self.parser.parse_file(file_path)

        # 验证分块目录存在
        chunk_dir = Path(self.temp_dir) / result.doc_hash
        self.assertTrue(chunk_dir.exists(), "分块目录应存在")
        self.assertTrue(chunk_dir.is_dir())

        # 验证每个分块文件存在且内容正确
        for i, expected_content in enumerate(result.chunks):
            chunk_file = chunk_dir / f"chunk_{i}.txt"
            self.assertTrue(chunk_file.exists(), f"chunk_{i}.txt 应存在")
            actual = chunk_file.read_text(encoding="utf-8")
            self.assertEqual(actual, expected_content,
                             f"chunk_{i}.txt 内容应一致")

    # ================================================================
    # 测试7: 文件系统存储 - Manifest 文件
    # ================================================================
    def test_manifest_generated(self):
        """测试 manifest.json 是否正确生成"""
        content = "测试 manifest 文件生成。" * 50
        file_path = self._create_temp_file(content, suffix=".txt")

        result = self.parser.parse_file(file_path)

        chunk_dir = Path(self.temp_dir) / result.doc_hash
        manifest_path = chunk_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "manifest.json 应存在")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["doc_hash"], result.doc_hash)
        self.assertEqual(manifest["source_file"], os.path.basename(file_path))
        self.assertEqual(manifest["total_chunks"], result.total_chunks)
        self.assertEqual(manifest["chunk_size"], self.parser.chunk_size)
        self.assertEqual(manifest["chunk_overlap"], self.parser.chunk_overlap)
        self.assertIn("created_at", manifest)
        self.assertIn("file_type", manifest)

    # ================================================================
    # 测试8: 按 Hash 读取分块
    # ================================================================
    def test_read_chunk_by_hash(self):
        """测试通过 doc_hash 和 chunk_id 读取分块内容"""
        content = "测试按 Hash 读取分块。" * 100
        file_path = self._create_temp_file(content)

        result = self.parser.parse_file(file_path)

        # 按 hash 读取指定分块
        for i, expected in enumerate(result.chunks):
            chunk_content = self.parser.get_chunk_by_hash(result.doc_hash, i)
            self.assertEqual(chunk_content, expected,
                             f"get_chunk_by_hash(chunk_{i}) 应返回正确内容")

    # ================================================================
    # 测试9: 按 Hash 读取 Manifest
    # ================================================================
    def test_read_manifest_by_hash(self):
        """测试通过 doc_hash 读取 manifest"""
        content = "测试读取 manifest。" * 50
        file_path = self._create_temp_file(content)

        result = self.parser.parse_file(file_path)

        manifest = self.parser.get_manifest(result.doc_hash)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["doc_hash"], result.doc_hash)
        self.assertEqual(manifest["total_chunks"], result.total_chunks)

    # ================================================================
    # 测试10: 不存在的 Hash 返回 None
    # ================================================================
    def test_invalid_hash_returns_none(self):
        """测试不存在的 doc_hash 返回 None"""
        chunk = self.parser.get_chunk_by_hash("nonexistent_hash_12345", 0)
        manifest = self.parser.get_manifest("nonexistent_hash_12345")
        self.assertIsNone(chunk)
        self.assertIsNone(manifest)

    # ================================================================
    # 测试11: 批量解析目录
    # ================================================================
    def test_parse_directory(self):
        """测试批量解析目录下所有文档"""
        # 创建多个测试文件
        contents = [
            "这是第一个文档的内容。" * 30,
            "这是第二个文档的内容，完全不同。" * 30,
            "第三个文档，用于测试批量解析。" * 30,
        ]
        for content in contents:
            self._create_temp_file(content)

        results = self.parser.parse_directory(self.temp_dir)

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, DocParseResult)
            self.assertGreater(r.total_chunks, 0)
            self.assertEqual(len(r.doc_hash), 32)  # MD5 长度为 32

    # ================================================================
    # 测试12: 边界情况 - 空文件
    # ================================================================
    def test_empty_file(self):
        """测试空文件处理"""
        file_path = self._create_temp_file("")

        result = self.parser.parse_file(file_path)

        self.assertEqual(result.total_chunks, 0)
        self.assertEqual(len(result.chunks), 0)


if __name__ == "__main__":
    unittest.main()