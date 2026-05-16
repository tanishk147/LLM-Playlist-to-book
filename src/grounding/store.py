"""SQLite-backed claim store. The 'grounding database' for the whole book."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    video_id TEXT NOT NULL,
    ts_start REAL NOT NULL,
    ts_end REAL NOT NULL,
    frame_ids TEXT NOT NULL,             -- JSON array
    confidence REAL NOT NULL,
    source_hash TEXT NOT NULL,
    extra TEXT                           -- JSON
);
CREATE INDEX IF NOT EXISTS idx_claims_video ON claims(video_id);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(type);

CREATE TABLE IF NOT EXISTS chapter_assignments (
    chapter_id TEXT NOT NULL,
    claim_id INTEGER NOT NULL,
    PRIMARY KEY (chapter_id, claim_id),
    FOREIGN KEY (claim_id) REFERENCES claims(id)
);
CREATE INDEX IF NOT EXISTS idx_assign_chapter ON chapter_assignments(chapter_id);

CREATE TABLE IF NOT EXISTS citations (
    chapter_id TEXT NOT NULL,
    sentence_index INTEGER NOT NULL,
    sentence_text TEXT NOT NULL,
    sentence_type TEXT NOT NULL,
    claim_id INTEGER,                    -- nullable for unsupported/narrative
    status TEXT NOT NULL,                -- grounded|unsupported|contradicted|needs_external_check|narrative_ok|external_ok
    reason TEXT,
    PRIMARY KEY (chapter_id, sentence_index, claim_id)
);
CREATE INDEX IF NOT EXISTS idx_cit_chapter ON citations(chapter_id);
CREATE INDEX IF NOT EXISTS idx_cit_status ON citations(status);
"""


@dataclass
class Claim:
    type: str
    content: str
    video_id: str
    ts_start: float
    ts_end: float
    frame_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source_hash: str = ""
    extra: dict | None = None
    id: int | None = None  # set after insert


class ClaimStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------- claims ----------
    def add_claims(self, claims: Iterable[Claim]) -> list[int]:
        ids: list[int] = []
        with self._conn() as conn:
            cur = conn.cursor()
            for cl in claims:
                cur.execute(
                    """
                    INSERT INTO claims (type, content, video_id, ts_start, ts_end,
                                        frame_ids, confidence, source_hash, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cl.type,
                        cl.content,
                        cl.video_id,
                        cl.ts_start,
                        cl.ts_end,
                        json.dumps(cl.frame_ids),
                        cl.confidence,
                        cl.source_hash,
                        json.dumps(cl.extra) if cl.extra else None,
                    ),
                )
                cl.id = cur.lastrowid
                ids.append(cur.lastrowid)
        return ids

    def get_claim(self, claim_id: int) -> Claim | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
            return self._row_to_claim(row) if row else None

    def get_claims(self, claim_ids: Iterable[int]) -> list[Claim]:
        ids = list(claim_ids)
        if not ids:
            return []
        qmarks = ",".join("?" * len(ids))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM claims WHERE id IN ({qmarks})", ids
            ).fetchall()
        order = {cid: i for i, cid in enumerate(ids)}
        claims = [self._row_to_claim(r) for r in rows]
        claims.sort(key=lambda cl: order.get(cl.id or -1, 1 << 30))
        return claims

    def claims_for_video(self, video_id: str) -> list[Claim]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM claims WHERE video_id = ? ORDER BY ts_start", (video_id,)
            ).fetchall()
            return [self._row_to_claim(r) for r in rows]

    def all_videos(self) -> list[str]:
        with self._conn() as c:
            return [r["video_id"] for r in c.execute("SELECT DISTINCT video_id FROM claims")]

    def all_claims(self) -> list[Claim]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM claims ORDER BY video_id, ts_start").fetchall()
            return [self._row_to_claim(r) for r in rows]

    def existing_ids(self, ids: Iterable[int]) -> set[int]:
        ids = list({int(i) for i in ids if i is not None})
        if not ids:
            return set()
        qmarks = ",".join("?" * len(ids))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT id FROM claims WHERE id IN ({qmarks})", ids
            ).fetchall()
            return {r["id"] for r in rows}

    def _row_to_claim(self, row: sqlite3.Row) -> Claim:
        return Claim(
            id=row["id"],
            type=row["type"],
            content=row["content"],
            video_id=row["video_id"],
            ts_start=row["ts_start"],
            ts_end=row["ts_end"],
            frame_ids=json.loads(row["frame_ids"]),
            confidence=row["confidence"],
            source_hash=row["source_hash"],
            extra=json.loads(row["extra"]) if row["extra"] else None,
        )

    # ---------- chapter assignments ----------
    def assign_claims(self, chapter_id: str, claim_ids: Iterable[int]) -> None:
        with self._conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO chapter_assignments (chapter_id, claim_id) VALUES (?, ?)",
                [(chapter_id, cid) for cid in claim_ids],
            )

    def claims_for_chapter(self, chapter_id: str) -> list[Claim]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT claims.* FROM claims
                JOIN chapter_assignments ON claims.id = chapter_assignments.claim_id
                WHERE chapter_assignments.chapter_id = ?
                ORDER BY claims.video_id, claims.ts_start
                """,
                (chapter_id,),
            ).fetchall()
            return [self._row_to_claim(r) for r in rows]

    # ---------- citations / audit ----------
    def record_citations(self, chapter_id: str, sentences: list[dict]) -> None:
        """sentences: items conforming to verify-prompt's per-sentence schema."""
        rows = []
        for s in sentences:
            cited = s.get("cited_claim_ids") or []
            if not cited:
                rows.append(
                    (chapter_id, s["index"], s["text"], s["type"], None,
                     s["status"], s.get("reason", ""))
                )
            else:
                for cid in cited:
                    # cited_claim_ids in the prompt schema are strings like "c47"
                    cid_int = _parse_claim_ref(cid)
                    rows.append(
                        (chapter_id, s["index"], s["text"], s["type"], cid_int,
                         s["status"], s.get("reason", ""))
                    )
        with self._conn() as c:
            c.executemany(
                """
                INSERT OR REPLACE INTO citations
                (chapter_id, sentence_index, sentence_text, sentence_type,
                 claim_id, status, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def audit_summary(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT status, COUNT(DISTINCT chapter_id || ':' || sentence_index) AS n "
                "FROM citations GROUP BY status"
            ).fetchall()
            return {r["status"]: r["n"] for r in rows}

    def audit_for_chapter(self, chapter_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM citations WHERE chapter_id = ? "
                "ORDER BY sentence_index",
                (chapter_id,),
            ).fetchall()
            return [dict(r) for r in rows]


def _parse_claim_ref(ref: str | int) -> int | None:
    """Parse a claim reference like 'c47', 'C47', or 47 to integer."""
    if isinstance(ref, int):
        return ref
    s = ref.strip().lstrip("cC").rstrip("]").strip()
    try:
        return int(s)
    except ValueError:
        return None
