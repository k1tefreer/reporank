"""生成一个离线的迷你测试集。

目的：在你还没下载 SWE-bench、没 clone 任何仓库之前，
就能把「建索引 → 检索 → 算指标」整条链路跑通。
先让管道通，再灌真实数据 —— 这是 W1 最重要的纪律。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "fixtures"
REPO = ROOT / "mini_repo"

FILES = {
    "src/auth/token.py": '''
import time

class TokenValidator:
    """Validates access tokens."""

    def __init__(self, secret, ttl=3600):
        self.secret = secret
        self.ttl = ttl

    def isExpired(self, issuedAt):
        # BUG: uses > instead of >=, tokens live one second too long
        return time.time() - issuedAt > self.ttl

    def validateToken(self, token):
        payload = self.decode(token)
        if self.isExpired(payload["iat"]):
            raise ValueError("token expired")
        return payload
''',
    "src/auth/session.py": '''
class SessionStore:
    def __init__(self):
        self._sessions = {}

    def get_session(self, session_id):
        return self._sessions.get(session_id)

    def drop_session(self, session_id):
        self._sessions.pop(session_id, None)
''',
    "src/http/request.py": '''
class Request:
    def __init__(self, headers=None):
        self.headers = headers or {}

    def get_header(self, name):
        # header lookup is case sensitive here
        return self.headers.get(name)
''',
    "src/http/response.py": '''
class Response:
    def __init__(self, status=200, body=b""):
        self.status = status
        self.body = body

    def set_cookie(self, name, value, max_age=None):
        self.headers = getattr(self, "headers", {})
        self.headers["Set-Cookie"] = f"{name}={value}"
''',
    "src/utils/dates.py": '''
from datetime import datetime, timezone

def parseTimestamp(raw):
    return datetime.fromtimestamp(float(raw), tz=timezone.utc)

def formatDuration(seconds):
    minutes, sec = divmod(int(seconds), 60)
    return f"{minutes}m{sec}s"
''',
    "src/db/query.py": '''
class QueryBuilder:
    def __init__(self, table):
        self.table = table
        self._where = []

    def where(self, clause):
        self._where.append(clause)
        return self

    def build(self):
        sql = f"SELECT * FROM {self.table}"
        if self._where:
            sql += " WHERE " + " AND ".join(self._where)
        return sql
''',
    "tests/test_token.py": '''
from src.auth.token import TokenValidator

def test_expiry():
    v = TokenValidator("s", ttl=10)
    assert v.isExpired(0)
''',
}

INSTANCES = [
    {
        "instance_id": "mini__token-expiry-off-by-one",
        "repo": "acme/mini",
        "base_commit": "deadbeef",
        "problem_statement": (
            "Access tokens remain valid for one extra second after they expire.\n\n"
            "When the token TTL is reached exactly, is_expired still returns False, "
            "so validate_token accepts an expired token. The comparison against the "
            "TTL looks off by one."
        ),
        "patch": (
            "diff --git a/src/auth/token.py b/src/auth/token.py\n"
            "--- a/src/auth/token.py\n"
            "+++ b/src/auth/token.py\n"
            "@@ -10,7 +10,7 @@\n"
            "-        return time.time() - issuedAt > self.ttl\n"
            "+        return time.time() - issuedAt >= self.ttl\n"
        ),
        "test_patch": "",
        "hints_text": "",
    },
    {
        "instance_id": "mini__header-case-insensitive",
        "repo": "acme/mini",
        "base_commit": "deadbeef",
        "problem_statement": (
            "Request header lookup should be case insensitive.\n\n"
            "Calling get_header('Content-Type') fails when the client sent "
            "'content-type'. HTTP header names are case insensitive per RFC 7230."
        ),
        "patch": (
            "diff --git a/src/http/request.py b/src/http/request.py\n"
            "--- a/src/http/request.py\n"
            "+++ b/src/http/request.py\n"
            "@@ -5,4 +5,4 @@\n"
            "-        return self.headers.get(name)\n"
            "+        return self.headers.get(name.lower())\n"
        ),
        "test_patch": "",
        "hints_text": "",
    },
    {
        "instance_id": "mini__duration-format-hours",
        "repo": "acme/mini",
        "base_commit": "deadbeef",
        "problem_statement": (
            "format_duration does not handle durations longer than an hour.\n\n"
            "Passing 7200 returns '120m0s' instead of '2h0m0s'. The duration "
            "formatting helper needs an hours component."
        ),
        "patch": (
            "diff --git a/src/utils/dates.py b/src/utils/dates.py\n"
            "--- a/src/utils/dates.py\n"
            "+++ b/src/utils/dates.py\n"
            "@@ -7,3 +7,5 @@\n"
            "-    minutes, sec = divmod(int(seconds), 60)\n"
            "-    return f\"{minutes}m{sec}s\"\n"
            "+    hours, rem = divmod(int(seconds), 3600)\n"
            "+    minutes, sec = divmod(rem, 60)\n"
            "+    return f\"{hours}h{minutes}m{sec}s\"\n"
        ),
        "test_patch": "",
        "hints_text": "",
    },
]


def main() -> None:
    for rel, body in FILES.items():
        p = REPO / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body.lstrip(), encoding="utf-8")

    out = ROOT / "sample_instances.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for ins in INSTANCES:
            f.write(json.dumps(ins, ensure_ascii=False) + "\n")

    print(f"wrote {len(FILES)} files to {REPO}")
    print(f"wrote {len(INSTANCES)} instances to {out}")


if __name__ == "__main__":
    main()
