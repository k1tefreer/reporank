"""文档切分策略。

W2 的基线是「一个文件 = 一个 Document」。问题在于长文件会稀释信号：
django/db/models/query.py 有两千多行，issue 只关心其中一个方法，
但 BM25 看到的是整个文件的词频分布。

W3 引入三种切分策略做对照实验：

  file    整文件（W2 基线，对照组 A）
  window  固定行数滑动窗口（对照组 B）
  ast     按函数/类切分（实验组）

为什么必须有 window 对照组：如果只比 file 和 ast，就算 ast 涨了，
也分不清是「切分本身有用」还是「AST 感知的切分有用」。
加上等价 chunk 数量的固定窗口切分，才能把功劳归对地方。

切分后 doc_id 变成 "path::qualified_name"，但 path 字段不变，
评测层与 search_files() 的文件级折叠逻辑完全不用改。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..retrieval.base import Document

# 单个 chunk 的字符上限。超长函数会被截断——极少见，但要防止
# 一个五千行的自动生成函数吃掉整个索引。
MAX_CHUNK_CHARS = 12_000
# 窗口切分的参数，设成和 AST chunk 的中位数量级接近，保证对照公平
WINDOW_LINES = 60
WINDOW_STRIDE = 45  # 有重叠，避免边界处的定义被切断


@dataclass(slots=True)
class Chunk:
    qualname: str  # "ClassName.method_name" / "func_name" / "<module>"
    start_line: int
    end_line: int
    text: str


def _header(path: str, qualname: str) -> str:
    """给每个 chunk 加一行上下文头。

    chunk 脱离了文件语境，所以要把「它是谁、住在哪」显式写进文本，
    否则 BM25 看不到类名和路径信号。这一行会参与分词，
    等效于把结构信息注入词袋。
    """
    return f"# file: {path}\n# symbol: {qualname}\n"


def _truncate(text: str) -> str:
    return text if len(text) <= MAX_CHUNK_CHARS else text[:MAX_CHUNK_CHARS]


# --------------------------------------------------------------------------
# 策略 1：整文件（W2 基线）
# --------------------------------------------------------------------------

def chunk_whole_file(path: str, content: str) -> list[Document]:
    return [Document(doc_id=path, path=path, content=content)]


# --------------------------------------------------------------------------
# 策略 2：固定行数窗口（对照组）
# --------------------------------------------------------------------------

def chunk_by_window(
    path: str,
    content: str,
    *,
    size: int = WINDOW_LINES,
    stride: int = WINDOW_STRIDE,
) -> list[Document]:
    lines = content.splitlines()
    if not lines:
        return []
    out: list[Document] = []
    for i in range(0, max(len(lines), 1), stride):
        window = lines[i : i + size]
        if not window:
            break
        qual = f"L{i + 1}-{i + len(window)}"
        out.append(
            Document(
                doc_id=f"{path}::{qual}",
                path=path,
                content=_truncate(_header(path, qual) + "\n".join(window)),
                meta={"start_line": i + 1, "end_line": i + len(window)},
            )
        )
        if i + size >= len(lines):
            break
    return out


# --------------------------------------------------------------------------
# 策略 3：AST 切分（实验组）
# --------------------------------------------------------------------------

_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _node_span(node: ast.AST) -> tuple[int, int]:
    """节点的行范围，包含装饰器。"""
    start = getattr(node, "lineno", 1)
    for dec in getattr(node, "decorator_list", []):
        start = min(start, dec.lineno)
    end = getattr(node, "end_lineno", start)
    return start, end


def _extract(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1 : end])


def _collect_chunks(tree: ast.Module, lines: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    covered: set[int] = set()  # 已被函数/类占用的行

    for node in tree.body:
        if isinstance(node, _FUNC_TYPES):
            s, e = _node_span(node)
            chunks.append(Chunk(node.name, s, e, _extract(lines, s, e)))
            covered.update(range(s, e + 1))

        elif isinstance(node, ast.ClassDef):
            cs, ce = _node_span(node)
            covered.update(range(cs, ce + 1))

            # 类头：class 语句 + docstring + 类属性，到第一个方法为止
            methods = [b for b in node.body if isinstance(b, _FUNC_TYPES)]
            head_end = _node_span(methods[0])[0] - 1 if methods else ce
            if head_end >= cs:
                chunks.append(
                    Chunk(node.name, cs, head_end, _extract(lines, cs, head_end))
                )

            # 每个方法一个 chunk，qualname 带上类名 —— 这是关键的结构信号
            for m in methods:
                ms, me = _node_span(m)
                chunks.append(
                    Chunk(f"{node.name}.{m.name}", ms, me, _extract(lines, ms, me))
                )

    # 模块级残余：import、常量、模块 docstring。
    # 这些往往包含关键的领域词汇，不能丢。
    rest = [i for i in range(1, len(lines) + 1) if i not in covered]
    if rest:
        text = "\n".join(lines[i - 1] for i in rest).strip()
        if text:
            chunks.append(Chunk("<module>", rest[0], rest[-1], text))

    return chunks


def chunk_by_ast(path: str, content: str) -> list[Document]:
    """按函数/类切分。解析失败时回退到整文件。

    回退很重要：SWE-bench 里有些 base_commit 的代码带语法错误，
    或者是 Python 2 语法。硬失败会让整个实例被 skip，污染评测。
    """
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, RecursionError):
        return chunk_whole_file(path, content)

    lines = content.splitlines()
    chunks = _collect_chunks(tree, lines)
    if not chunks:
        return chunk_whole_file(path, content)

    return [
        Document(
            doc_id=f"{path}::{c.qualname}",
            path=path,
            content=_truncate(_header(path, c.qualname) + c.text),
            meta={"qualname": c.qualname, "start_line": c.start_line, "end_line": c.end_line},
        )
        for c in chunks
    ]


# --------------------------------------------------------------------------

CHUNKERS = {
    "file": chunk_whole_file,
    "window": chunk_by_window,
    "ast": chunk_by_ast,
}


def get_chunker(name: str):
    if name not in CHUNKERS:
        raise ValueError(f"unknown chunking strategy: {name} (choose from {sorted(CHUNKERS)})")
    return CHUNKERS[name]