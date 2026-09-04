"""代码感知分词器。

通用 IR 分词器直接用在代码上会有严重问题：
  - `getUserName` 被当成一个词，issue 里写的 "get user name" 匹配不上
  - `user_id` 同理
  - 路径 `src/auth/token.py` 里的信号被丢掉

这个模块负责把标识符按驼峰/下划线/数字边界拆开，同时保留原始形式
（原始形式对精确匹配有用，拆分形式对语义匹配有用，两者都进索引）。

W1 只做规则分词。W3 会在这之上加 AST 结构信息。
"""

from __future__ import annotations

import re
from typing import Iterable

# 先按非字母数字切分，再处理驼峰
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
# 驼峰边界：aB / ABc / a1
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=[0-9])")

# 编程语言关键字停用词。
# 注意：这里不能照搬 NLP 的英文停用词表。代码里 "if"/"for" 是噪声，
# 但 "self"/"class" 在某些查询下反而是信号，所以停用词表要保守。
CODE_STOPWORDS = frozenset(
    """
    if else elif for while return def class import from as pass break continue
    try except finally raise with lambda and or not in is none true false
    self cls print len str int float bool list dict set tuple
    the a an of to be this that it
    """.split()
)

MIN_TOKEN_LEN = 2
MAX_TOKEN_LEN = 40


def split_identifier(token: str) -> list[str]:
    """把一个标识符拆成子词。

    >>> split_identifier("getUserName")
    ['getusername', 'get', 'user', 'name']
    >>> split_identifier("user_id")
    ['user', 'id']
    >>> split_identifier("HTTPResponse")
    ['httpresponse', 'http', 'response']
    """
    parts = [p for p in _CAMEL.split(token) if p]
    lowered = [p.lower() for p in parts]
    if len(lowered) <= 1:
        return lowered
    # 拆分的同时保留原始整体形式，让精确匹配仍然能命中
    return [token.lower()] + lowered


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """把一段代码或自然语言文本转成 token 列表。"""
    out: list[str] = []
    for raw in _NON_ALNUM.split(text):
        if not raw:
            continue
        for tok in split_identifier(raw):
            if len(tok) < MIN_TOKEN_LEN or len(tok) > MAX_TOKEN_LEN:
                continue
            if drop_stopwords and tok in CODE_STOPWORDS:
                continue
            out.append(tok)
    return out


def tokenize_path(path: str, *, repeat: int = 3) -> list[str]:
    """路径单独分词并加权。

    文件路径是极强的信号——issue 提到 "auth" 时 `src/auth/` 下的文件
    大概率相关。BM25 本身没有字段加权机制，这里用「重复 token」
    的土办法实现 field boosting。repeat 是需要调的超参，
    W2 的消融实验里会扫这个值。
    """
    toks = tokenize(path.replace("/", " ").replace(".", " "))
    return toks * repeat


def tokenize_document(path: str, content: str, *, path_boost: int = 3) -> list[str]:
    """文档侧分词：路径 token（加权）+ 内容 token。"""
    return tokenize_path(path, repeat=path_boost) + tokenize(content)


def tokenize_query(text: str) -> list[str]:
    """查询侧分词。issue 正文是自然语言 + 代码片段混合，用同一套规则。"""
    return tokenize(text)


def iter_ngrams(tokens: Iterable[str], n: int = 2) -> list[str]:
    """预留给后续实验：bigram 对 `def foo` 这类模式可能有用。W1 暂不启用。"""
    toks = list(tokens)
    return ["_".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
