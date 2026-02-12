"""SQLite database setup and CRUD operations."""
import sqlite3
import threading
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from config import config

_local = threading.local()


def now_iso() -> str:
    """Current UTC datetime as ISO string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(config.database_path, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=5000")
    return _local.conn


@contextmanager
def get_db():
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ── Schema ─────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist."""
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS animals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                animal TEXT,
                height_cm TEXT,
                weight_kg TEXT,
                color TEXT,
                lifespan_years TEXT,
                diet TEXT,
                habitat TEXT,
                predators TEXT,
                average_speed_kmh TEXT,
                countries_found TEXT,
                conservation_status TEXT,
                family TEXT,
                gestation_period_days TEXT,
                top_speed_kmh TEXT,
                social_structure TEXT,
                offspring_per_birth TEXT,
                scientific_name TEXT,
                safety_info TEXT,
                is_dangerous INTEGER
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tracking_id TEXT NOT NULL,
                bbox TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                last_seen TEXT NOT NULL,
                start_frame INTEGER,
                end_frame INTEGER
            );

            CREATE TABLE IF NOT EXISTS ai_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tracking_id TEXT NOT NULL UNIQUE,
                common_name TEXT,
                scientific_name TEXT,
                description TEXT
            );

            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tracking_id TEXT NOT NULL,
                animal_id INTEGER,
                FOREIGN KEY (animal_id) REFERENCES animals(id)
            );

            CREATE INDEX IF NOT EXISTS idx_events_tracking ON events(tracking_id);
            CREATE INDEX IF NOT EXISTS idx_ai_det_tracking ON ai_detections(tracking_id);
            CREATE INDEX IF NOT EXISTS idx_detections_tracking ON detections(tracking_id);
            CREATE INDEX IF NOT EXISTS idx_animals_name ON animals(animal);
            CREATE INDEX IF NOT EXISTS idx_animals_sci ON animals(scientific_name);
        """)
        # Add safety_info and is_dangerous to existing DBs (no-op if already present)
        for col, ctype in [("safety_info", "TEXT"), ("is_dangerous", "INTEGER")]:
            try:
                db.execute(f"ALTER TABLE animals ADD COLUMN {col} {ctype}")
            except sqlite3.OperationalError:
                pass  # column already exists


# ── Events ─────────────────────────────────────────────────────

def create_event(tracking_id: str, bbox: list, start_frame: int | None = None) -> int:
    ts = now_iso()
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO events (tracking_id, bbox, start_time, last_seen, start_frame) VALUES (?, ?, ?, ?, ?)",
            (tracking_id, json.dumps(bbox), ts, ts, start_frame),
        )
        return cur.lastrowid


def end_event(tracking_id: str, end_frame: int | None = None):
    ts = now_iso()
    with get_db() as db:
        db.execute(
            "UPDATE events SET end_time = ?, last_seen = ?, end_frame = ? WHERE tracking_id = ? AND end_time IS NULL",
            (ts, ts, end_frame, tracking_id),
        )


def update_event_last_seen(tracking_id: str):
    ts = now_iso()
    with get_db() as db:
        db.execute(
            "UPDATE events SET last_seen = ? WHERE tracking_id = ? AND end_time IS NULL",
            (ts, tracking_id),
        )


def get_active_events() -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            "SELECT e.*, ad.common_name, ad.scientific_name, ad.description "
            "FROM events e LEFT JOIN ai_detections ad ON e.tracking_id = ad.tracking_id "
            "WHERE e.end_time IS NULL ORDER BY e.start_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_events(limit: int = 100) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            "SELECT e.*, ad.common_name, ad.scientific_name "
            "FROM events e LEFT JOIN ai_detections ad ON e.tracking_id = ad.tracking_id "
            "ORDER BY e.start_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_event_by_tracking_id(tracking_id: str) -> dict | None:
    """Return the event row for a tracking_id, or None if not found."""
    with get_db() as db:
        row = db.execute(
            "SELECT e.*, ad.common_name, ad.scientific_name, ad.description "
            "FROM events e LEFT JOIN ai_detections ad ON e.tracking_id = ad.tracking_id "
            "WHERE e.tracking_id = ?",
            (tracking_id,),
        ).fetchone()
        return dict(row) if row else None


# ── AI Detections ──────────────────────────────────────────────

def upsert_ai_detection(tracking_id: str, common_name: str, scientific_name: str, description: str):
    with get_db() as db:
        db.execute(
            """INSERT INTO ai_detections (tracking_id, common_name, scientific_name, description)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(tracking_id) DO UPDATE SET
                 common_name=excluded.common_name,
                 scientific_name=excluded.scientific_name,
                 description=excluded.description""",
            (tracking_id, common_name, scientific_name, description),
        )


def get_ai_detection(tracking_id: str) -> dict | None:
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM ai_detections WHERE tracking_id = ?", (tracking_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_tracker_entry(tracking_id: str):
    """Remove all DB entries for a tracking_id (events, ai_detections, detections)."""
    with get_db() as db:
        db.execute("DELETE FROM detections WHERE tracking_id = ?", (tracking_id,))
        db.execute("DELETE FROM ai_detections WHERE tracking_id = ?", (tracking_id,))
        db.execute("DELETE FROM events WHERE tracking_id = ?", (tracking_id,))


def reset_db():
    """Delete all data from events, ai_detections, detections, and animals tables."""
    with get_db() as db:
        db.execute("DELETE FROM detections")
        db.execute("DELETE FROM ai_detections")
        db.execute("DELETE FROM events")
        db.execute("DELETE FROM animals")


# ── Animals ────────────────────────────────────────────────────

ANIMAL_COLUMNS = [
    "animal", "height_cm", "weight_kg", "color", "lifespan_years",
    "diet", "habitat", "predators", "average_speed_kmh", "countries_found",
    "conservation_status", "family", "gestation_period_days", "top_speed_kmh",
    "social_structure", "offspring_per_birth", "scientific_name",
    "safety_info", "is_dangerous",
]


def _normalize_name(name: str | None) -> str | None:
    """Return stripped non-empty name or None for consistent lookups."""
    if name is None:
        return None
    s = name.strip()
    return s if s else None


# Lock so concurrent sync_detection calls don't both insert the same animal.
_animal_creation_lock = threading.Lock()


def find_animal(common_name: str = None, scientific_name: str = None) -> dict | None:
    common_name = _normalize_name(common_name)
    scientific_name = _normalize_name(scientific_name)
    with get_db() as db:
        if common_name:
            row = db.execute(
                "SELECT * FROM animals WHERE LOWER(TRIM(animal)) = LOWER(?)", (common_name,)
            ).fetchone()
            if row:
                return dict(row)
        if scientific_name:
            row = db.execute(
                "SELECT * FROM animals WHERE LOWER(TRIM(scientific_name)) = LOWER(?)",
                (scientific_name,),
            ).fetchone()
            if row:
                return dict(row)
    return None


def insert_animal(data: dict) -> int:
    """Insert animal; normalize 'animal' and 'scientific_name' by stripping for consistency."""
    data = dict(data)
    if "animal" in data and data["animal"] is not None:
        data["animal"] = data["animal"].strip() or data["animal"]
    if "scientific_name" in data and data["scientific_name"] is not None:
        data["scientific_name"] = (data["scientific_name"].strip() or data["scientific_name"])
    with get_db() as db:
        cols = [c for c in ANIMAL_COLUMNS if c in data]
        def _val(c, v):
            if v is None:
                return None
            if c == "is_dangerous":
                return 1 if v else 0
            return str(v)
        vals = [_val(c, data[c]) for c in cols]
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        cur = db.execute(
            f"INSERT INTO animals ({col_str}) VALUES ({placeholders})", vals
        )
        return cur.lastrowid


def get_all_animals() -> list[dict]:
    with get_db() as db:
        rows = db.execute("SELECT * FROM animals ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


# ── Detections ─────────────────────────────────────────────────

def create_detection(tracking_id: str, animal_id: int) -> int:
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO detections (tracking_id, animal_id) VALUES (?, ?)",
            (tracking_id, animal_id),
        )
        return cur.lastrowid


def get_all_detections(limit: int = 200) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            """SELECT d.id, d.tracking_id, d.animal_id,
                      a.animal, a.scientific_name, a.conservation_status, a.family,
                      a.safety_info, a.is_dangerous,
                      ad.description,
                      e.start_time, e.end_time, e.last_seen, e.bbox, e.start_frame, e.end_frame
               FROM detections d
               LEFT JOIN animals a ON d.animal_id = a.id
               LEFT JOIN ai_detections ad ON d.tracking_id = ad.tracking_id
               LEFT JOIN events e ON d.tracking_id = e.tracking_id
               ORDER BY d.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_detection_by_tracking(tracking_id: str) -> dict | None:
    with get_db() as db:
        row = db.execute(
            "SELECT d.*, a.animal, a.scientific_name FROM detections d "
            "LEFT JOIN animals a ON d.animal_id = a.id WHERE d.tracking_id = ?",
            (tracking_id,),
        ).fetchone()
        return dict(row) if row else None


def get_detection_detail(tracking_id: str) -> dict | None:
    """Return full detection with animal, event, and ai_detection data for modal display."""
    with get_db() as db:
        row = db.execute(
            """SELECT d.id AS detection_id, d.tracking_id, d.animal_id,
                      a.animal, a.scientific_name, a.conservation_status, a.family,
                      a.height_cm, a.weight_kg, a.color, a.lifespan_years, a.diet, a.habitat,
                      a.predators, a.average_speed_kmh, a.countries_found, a.gestation_period_days,
                      a.top_speed_kmh, a.social_structure, a.offspring_per_birth,
                      a.safety_info, a.is_dangerous,
                      ad.description AS ai_description,
                      e.start_time, e.end_time, e.last_seen, e.bbox, e.start_frame, e.end_frame
               FROM detections d
               LEFT JOIN animals a ON d.animal_id = a.id
               LEFT JOIN ai_detections ad ON d.tracking_id = ad.tracking_id
               LEFT JOIN events e ON d.tracking_id = e.tracking_id
               WHERE d.tracking_id = ?""",
            (tracking_id,),
        ).fetchone()
        return dict(row) if row else None


# ── Agent: Schema & read-only query ─────────────────────────────

def get_schema_txt() -> str:
    """Return full database schema as plain text for documentation."""
    lines = ["# Database schema (SQLite)", ""]
    with get_db() as db:
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for (tname,) in tables:
            lines.append(f"## Table: {tname}")
            rows = db.execute(f"PRAGMA table_info({tname})").fetchall()
            for r in rows:
                _cid, name, ctype, _notnull, _default, pk = r
                pk_str = " PRIMARY KEY" if pk else ""
                lines.append(f"  - {name}: {ctype}{pk_str}")
            lines.append("")
        # Indexes
        lines.append("# Indexes")
        idx = db.execute(
            "SELECT name, tbl_name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL ORDER BY tbl_name, name"
        ).fetchall()
        for name, tbl, _sql in idx:
            lines.append(f"  - {name} on {tbl}")
        lines.append("")
    return "\n".join(lines)


def execute_read_only_query(sql: str) -> tuple[list[dict], str | None]:
    """
    Execute a read-only SQL query (SELECT only). Returns (rows, error).
    If error is not None, rows is empty.
    """
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        return [], "Only SELECT queries are allowed"
    try:
        with get_db() as db:
            rows = db.execute(sql).fetchall()
            return [dict(r) for r in rows], None
    except sqlite3.Error as e:
        return [], str(e)
