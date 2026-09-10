"""In-memory session store.

Sessions hold the uploaded dataframe plus whatever the pipeline has produced so far.
Deliberately process-local: this is a demo service, not a system of record. Swap for
Redis or a database if it ever needs to survive a restart or run on more than one worker.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import settings


@dataclass
class Session:
    id: str
    filename: str
    df: pd.DataFrame
    created_at: float = field(default_factory=time.time)
    profile: Any = None
    plan: Any = None
    assessment: Any = None
    result: Any = None
    report: Any = None

    def touch(self) -> None:
        self.created_at = time.time()


class SessionStore:
    def __init__(self, ttl_seconds: int, max_sessions: int = 200) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max = max_sessions

    def _evict(self) -> None:
        now = time.time()
        stale = [k for k, s in self._sessions.items() if now - s.created_at > self._ttl]
        for k in stale:
            self._sessions.pop(k, None)
        if len(self._sessions) > self._max:
            oldest = sorted(self._sessions.items(), key=lambda kv: kv[1].created_at)
            for k, _ in oldest[: len(self._sessions) - self._max]:
                self._sessions.pop(k, None)

    def create(self, filename: str, df: pd.DataFrame) -> Session:
        with self._lock:
            self._evict()
            sid = uuid.uuid4().hex[:16]
            session = Session(id=sid, filename=filename, df=df)
            self._sessions[sid] = session
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


store = SessionStore(settings.session_ttl_seconds)
