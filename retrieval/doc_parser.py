# -*- coding: utf-8 -*-
"""
DocParser - 基于文件系统 + Hash 索引的轻量级 RAG 文档解析方案

提供文档解析、智能分块、Hash 索引生成、文件系统存储功能。
适用于中小规模场景（文档量 < 10万 chunk），无需向量库即可运行。

支持格式: .txt, .md, .pdf, .docx
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class DocParseResult:
    """文档解析结果"""

    def __init__(self, doc_hash: str, source_file: str):
        self.doc_hash = doc_hash
        self.source_file = source_file
        self.total_chunks = 0
        self.chunks: List[str] = []
        self.metadata: Dict = {}

    def to_dict(self) -> Dict:
        return {
            "doc_hash": self.doc_hash,
            "source_file": self.source_file,
            "total_chunks": self.total_chunks,
            "chunks": self.chunks,
            "metadata": self.metadata,
        }


class DocParser:
    """文档解析器 - 负责文档读取、分块、Hash 索引、文件系统存储"""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        """
        初始化 DocParser

        Args:
            chunk_size: 每个分块的最大字符数，默认读取全局配置 CHUNK_SIZE
            chunk_overlap: 相邻分块的重叠字符数，默认读取全局配置 CHUNK_OVERLAP
            output_dir: 分块文件输出目录，默认为 data/chunks
        """
        if chunk_size is None:
            from config.settings import CHUNK_SIZE
            chunk_size = CHUNK_SIZE
        if chunk_overlap is None:
            from config.settings import CHUNK_OVERLAP
            chunk_overlap = CHUNK_OVERLAP
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if output_dir is None:
            from config.settings import CHUNKS_DIR
            output_dir = str(CHUNKS_DIR)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 公开接口
    # ================================================================

    def parse_file(self, file_path: str) -> DocParseResult:
        """
        解析单个文件，生成分块并写入磁盘

        Args:
            file_path: 文件路径

        Returns:
            DocParseResult: 包含 doc_hash、chunks、metadata 的解析结果
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取文件内容
        content = self._read_file(file_path)
        if not content.strip():
            return DocParseResult("", str(file_path.name))

        # 计算文档 Hash
        doc_hash = self._compute_hash(content)

        # 文本分块
        chunks = self._split_text(content)

        # 构建结果
        result = DocParseResult(doc_hash, str(file_path.name))
        result.total_chunks = len(chunks)
        result.chunks = chunks
        result.metadata = {
            "file_type": file_path.suffix.lower(),
            "original_size": os.path.getsize(file_path),
            "content_length": len(content),
        }

        # 写入文件系统
        self._write_chunks(doc_hash, chunks, result)

        return result

    def parse_directory(self, dir_path: str) -> List[DocParseResult]:
        """
        批量解析目录下所有支持的文件

        Args:
            dir_path: 目录路径

        Returns:
            List[DocParseResult]: 所有文件的解析结果列表
        """
        dir_path = Path(dir_path)
        results = []

        supported_extensions = {".txt", ".md", ".pdf", ".docx"}
        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    result = self.parse_file(str(file_path))
                    results.append(result)
                except Exception as e:
                    print(f"解析文件失败 [{file_path.name}]: {e}")

        return results

    def get_chunk_by_hash(self, doc_hash: str, chunk_id: int) -> Optional[str]:
        """
        通过文档 Hash 和分块编号读取分块内容

        Args:
            doc_hash: 文档 MD5 哈希值
            chunk_id: 分块编号（从 0 开始）

        Returns:
            str: 分块文本内容，不存在时返回 None
        """
        chunk_file = self.output_dir / doc_hash / f"chunk_{chunk_id}.txt"
        if not chunk_file.exists():
            return None
        return chunk_file.read_text(encoding="utf-8")

    def get_manifest(self, doc_hash: str) -> Optional[Dict]:
        """
        通过文档 Hash 读取 manifest 元数据

        Args:
            doc_hash: 文档 MD5 哈希值

        Returns:
            Dict: manifest 内容，不存在时返回 None
        """
        manifest_path = self.output_dir / doc_hash / "manifest.json"
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    # ================================================================
    # 内部方法
    # ================================================================

    def _read_file(self, file_path: Path) -> str:
        """根据文件类型读取内容"""
        suffix = file_path.suffix.lower()

        if suffix in (".txt", ".md"):
            # 尝试多种编码
            for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
                try:
                    return file_path.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"无法解码文件: {file_path}")

        elif suffix == ".pdf":
            return self._read_pdf(file_path)

        elif suffix == ".docx":
            return self._read_docx(file_path)

        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

    def _read_pdf(self, file_path: Path) -> str:
        """读取 PDF 文件内容"""
        try:
            import pymupdf
            doc = pymupdf.open(str(file_path))
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except ImportError:
            raise ImportError("读取 PDF 需要安装 pymupdf: pip install pymupdf")

    def _read_docx(self, file_path: Path) -> str:
        """读取 Word 文档内容"""
        try:
            from docx import Document
            doc = Document(str(file_path))
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except ImportError:
            raise ImportError("读取 DOCX 需要安装 python-docx: pip install python-docx")

    def _compute_hash(self, content: str) -> str:
        """计算文档内容的 MD5 哈希值"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _split_text(self, text: str) -> List[str]:
        """
        递归字符分割文本为指定大小的分块

        使用中文友好的分隔符优先级：
        双换行 → 单换行 → 句号 → 感叹号 → 问号 → 分号 → 逗号 → 空格 → 字符
        """
        from config.settings import CHUNK_SEPARATORS

        separators = CHUNK_SEPARATORS
        return self._split_recursive(text, separators)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """递归分割文本，直到每个分块不超过 chunk_size"""
        # 找到最合适的分隔符
        best_separator = None
        for sep in separators:
            if sep in text:
                best_separator = sep
                break

        if best_separator is None:
            # 无法再分割，按字符截断
            return self._split_by_length(text)

        # 用分隔符切分
        segments = text.split(best_separator)

        chunks = []
        current_chunk = ""
        for segment in segments:
            # 拼接后检查是否超限
            test_chunk = current_chunk + best_separator + segment if current_chunk else segment

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # 如果单个 segment 就超过限制，递归分割
                if len(segment) > self.chunk_size:
                    sub_chunks = self._split_recursive(segment, separators[1:])
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = segment

        if current_chunk:
            chunks.append(current_chunk)

        # 生成重叠内容
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return chunks

    def _split_by_length(self, text: str) -> List[str]:
        """按固定长度截断文本"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            end = min(i + self.chunk_size, len(text))
            chunks.append(text[i:end])
            if end == len(text):
                break
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """为相邻 chunk 添加重叠内容"""
        if self.chunk_overlap <= 0:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            overlap_len = min(self.chunk_overlap, len(prev))
            if overlap_len > 0:
                # 取前一个 chunk 的末尾 overlap_len 个字符，拼接到当前 chunk 开头
                curr = prev[-overlap_len:] + curr
            overlapped.append(curr)

        return overlapped

    def _write_chunks(self, doc_hash: str, chunks: List[str], result: DocParseResult):
        """将分块写入文件系统，并生成 manifest.json"""
        chunk_dir = self.output_dir / doc_hash
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # 写入每个分块
        for i, chunk_text in enumerate(chunks):
            chunk_file = chunk_dir / f"chunk_{i}.txt"
            chunk_file.write_text(chunk_text, encoding="utf-8")

        # 生成 manifest.json
        manifest = {
            "doc_hash": doc_hash,
            "source_file": result.source_file,
            "total_chunks": len(chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "file_type": result.metadata.get("file_type", ""),
            "original_size": result.metadata.get("original_size", 0),
            "content_length": result.metadata.get("content_length", 0),
            "created_at": datetime.now().isoformat(),
        }
        manifest_path = chunk_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )