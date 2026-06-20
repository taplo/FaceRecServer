import logging
import os
import sqlite3
import uuid
import numpy as np
import faiss
from PIL import Image
from datetime import datetime, timezone
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FaceRecord:
    face_id: str
    name: str = ""
    created_at: str = ""
    image_path: str = ""
    employee_id: str = ""
    person_id: int = 0


@dataclass
class PersonRecord:
    person_id: int
    name: str
    employee_id: str
    created_at: str
    face_count: int = 0


@dataclass
class GalleryStats:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    failures: list = field(default_factory=list)


class GalleryRepository:
    DIM = 512

    def __init__(self, db_dir: str, db_name: str):
        os.makedirs(db_dir, exist_ok=True)
        self.gallery_dir = db_dir
        self.db_path = os.path.join(db_dir, f"{db_name}.db")
        self.index_path = os.path.join(db_dir, f"{db_name}.faiss")
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_db()
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIM))
        self._load_index()

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                employee_id TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                image_path TEXT DEFAULT '',
                employee_id TEXT DEFAULT '',
                person_id INTEGER REFERENCES persons(id)
            )
        """)
        try:
            self._conn.execute("ALTER TABLE faces ADD COLUMN employee_id TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute("ALTER TABLE faces ADD COLUMN person_id INTEGER REFERENCES persons(id)")
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute("ALTER TABLE faces ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_face_id ON faces(face_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_person_id ON faces(person_id)
        """)
        self._migrate_persons()
        self._conn.commit()

    def _migrate_persons(self):
        orphan_count = self._conn.execute(
            "SELECT COUNT(*) FROM faces WHERE person_id IS NULL OR person_id = 0"
        ).fetchone()[0]
        if orphan_count == 0:
            return
        rows = self._conn.execute(
            "SELECT DISTINCT name, employee_id FROM faces WHERE person_id IS NULL OR person_id = 0"
        ).fetchall()
        for name, employee_id in rows:
            if not name:
                continue
            now = datetime.now(timezone.utc).isoformat()
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO persons (name, employee_id, created_at) VALUES (?, ?, ?)",
                (name, employee_id, now),
            )
            if cur.lastrowid:
                self._conn.execute(
                    "UPDATE faces SET person_id = ? WHERE name = ? AND employee_id = ? AND (person_id IS NULL OR person_id = 0)",
                    (cur.lastrowid, name, employee_id),
                )

    def _load_index(self) -> None:
        rows = self._conn.execute("SELECT id FROM faces").fetchall()
        if not rows:
            return
        vec_path = self.index_path
        if os.path.exists(vec_path):
            self._index = faiss.read_index(vec_path)
        logger.info("加载底库: %s 条记录", len(rows))

    def _save_index(self) -> None:
        faiss.write_index(self._index, self.index_path)

    def rebuild_index(self) -> int:
        rows = self._conn.execute(
            "SELECT id, embedding FROM faces WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIM))
            self._save_index()
            return 0
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIM))
        ids = []
        vecs = []
        for row_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            if vec.shape[0] != self.DIM:
                continue
            vecs.append(vec)
            ids.append(row_id)
        if vecs:
            vectors = np.stack(vecs).astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms
            index.add_with_ids(vectors, np.array(ids, dtype=np.int64))
        self._index = index
        self._save_index()
        logger.info("重建索引完成: %d 条记录", len(ids))
        return len(ids)

    def get_or_create_person(self, name: str, employee_id: str = "") -> int:
        row = self._conn.execute(
            "SELECT id FROM persons WHERE name = ? AND employee_id = ?",
            (name, employee_id),
        ).fetchone()
        if row:
            return row[0]
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO persons (name, employee_id, created_at) VALUES (?, ?, ?)",
            (name, employee_id, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def add(self, embedding: np.ndarray, name: str, employee_id: str = "", image: np.ndarray | None = None, image_path: str = "") -> str:
        face_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        normalized = embedding / np.linalg.norm(embedding)
        person_id = self.get_or_create_person(name, employee_id)

        if image is not None:
            faces_dir = os.path.join(self.gallery_dir, "faces")
            os.makedirs(faces_dir, exist_ok=True)
            save_path = os.path.join(faces_dir, f"{face_id}.jpg")
            h, w = image.shape[:2]
            max_dim = 360
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                img_pil = Image.fromarray(image).resize(new_size, Image.LANCZOS)
            else:
                img_pil = Image.fromarray(image)
            img_pil.save(save_path, "JPEG", quality=85)
            image_path = f"faces/{face_id}.jpg"

        emb_blob = normalized.astype(np.float32).tobytes()
        cursor = self._conn.execute(
            "INSERT INTO faces (face_id, name, employee_id, person_id, created_at, image_path, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (face_id, name, employee_id, person_id, now, image_path, emb_blob),
        )
        faiss_id = cursor.lastrowid
        self._index.add_with_ids(normalized.reshape(1, -1).astype(np.float32), np.array([faiss_id]))
        self._conn.commit()
        self._save_index()
        return face_id

    def add_batch(self, embeddings: list, names: list, employee_ids: list | None = None, image_paths: list | None = None, images: list | None = None) -> GalleryStats:
        stats = GalleryStats(total=len(embeddings))
        for i, (emb, name) in enumerate(zip(embeddings, names)):
            try:
                emp_id = employee_ids[i] if employee_ids else ""
                img = images[i] if images else None
                path = image_paths[i] if image_paths else ""
                self.add(emb, name, employee_id=emp_id, image=img, image_path=path)
                stats.succeeded += 1
            except Exception as e:
                stats.failed += 1
                stats.failures.append({"file": name, "reason": str(e)})
        return stats

    def delete(self, face_id: str) -> bool:
        row = self._conn.execute("SELECT id FROM faces WHERE face_id = ?", (face_id,)).fetchone()
        if row is None:
            return False
        faiss_id = row[0]
        self._index.remove_ids(np.array([faiss_id]))
        self._conn.execute("DELETE FROM faces WHERE face_id = ?", (face_id,))
        self._conn.commit()
        self._save_index()
        return True

    def delete_person(self, person_id: int) -> bool:
        row = self._conn.execute("SELECT id FROM persons WHERE id = ?", (person_id,)).fetchone()
        if row is None:
            return False
        face_ids = self._conn.execute(
            "SELECT face_id FROM faces WHERE person_id = ?", (person_id,)
        ).fetchall()
        for (face_id,) in face_ids:
            self._conn.execute("DELETE FROM faces WHERE face_id = ?", (face_id,))
        self._conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        self._conn.commit()
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIM))
        self._save_index()
        return True

    def clear(self) -> None:
        self._conn.execute("DELETE FROM faces")
        self._conn.execute("DELETE FROM persons")
        self._conn.commit()
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIM))
        self._save_index()

    def list_faces(self, page: int = 1, page_size: int = 20, search: str = "") -> tuple:
        offset = (page - 1) * page_size
        if search:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM faces WHERE name LIKE ? OR employee_id LIKE ?", (f"%{search}%", f"%{search}%")
            ).fetchone()[0]
            rows = self._conn.execute(
                "SELECT face_id, name, created_at, image_path, employee_id, person_id FROM faces WHERE name LIKE ? OR employee_id LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (f"%{search}%", f"%{search}%", page_size, offset),
            ).fetchall()
        else:
            count = self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            rows = self._conn.execute(
                "SELECT face_id, name, created_at, image_path, employee_id, person_id FROM faces ORDER BY id DESC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
        items = [FaceRecord(face_id=r[0], name=r[1], created_at=r[2], image_path=r[3], employee_id=r[4], person_id=r[5]) for r in rows]
        return items, count

    def list_persons(self, page: int = 1, page_size: int = 20, search: str = "") -> tuple:
        offset = (page - 1) * page_size
        base_query = "FROM persons p"
        count_params = ()
        query_params = ()
        if search:
            where = " WHERE p.name LIKE ? OR p.employee_id LIKE ?"
            count_params = (f"%{search}%", f"%{search}%")
            query_params = (f"%{search}%", f"%{search}%", page_size, offset)
        else:
            where = ""
            query_params = (page_size, offset)
        count = self._conn.execute(
            f"SELECT COUNT(*) {base_query}{where}", count_params
        ).fetchone()[0]
        rows = self._conn.execute(
            f"SELECT p.id, p.name, p.employee_id, p.created_at, COUNT(f.id) AS face_count "
            f"{base_query} LEFT JOIN faces f ON f.person_id = p.id{where} "
            f"GROUP BY p.id ORDER BY p.id DESC LIMIT ? OFFSET ?",
            query_params,
        ).fetchall()
        items = [PersonRecord(person_id=r[0], name=r[1], employee_id=r[2], created_at=r[3], face_count=r[4]) for r in rows]
        return items, count

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if self._index.ntotal == 0:
            return []
        if top_k < 1:
            top_k = 1
        normalized = embedding / np.linalg.norm(embedding)
        query = normalized.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query, top_k * 3)
        person_best = {}
        for score, faiss_id in zip(distances[0], indices[0]):
            if faiss_id == -1:
                continue
            row = self._conn.execute(
                "SELECT f.face_id, f.name, f.image_path, f.employee_id, f.person_id FROM faces f WHERE f.id = ?",
                (int(faiss_id),),
            ).fetchone()
            if row is None:
                continue
            person_id = row[4]
            current = person_best.get(person_id)
            if current is None or score > current["score"]:
                path = row[2]
                image_url = None
                if path:
                    abs_path = os.path.join(self.gallery_dir, path)
                    if os.path.exists(abs_path):
                        image_url = f"/api/v1/gallery/{row[0]}/image"
                person_best[person_id] = {
                    "face_id": row[0],
                    "name": row[1],
                    "score": float(score),
                    "image_url": image_url,
                    "employee_id": row[3] or "",
                    "person_id": person_id,
                }
        results = sorted(person_best.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return results

    def get_image_path(self, face_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT image_path FROM faces WHERE face_id = ?", (face_id,)
        ).fetchone()
        return row[0] if row else None

    def get_image_url(self, face_id: str) -> str | None:
        path = self.get_image_path(face_id)
        if not path:
            return None
        abs_path = os.path.join(self.gallery_dir, path)
        return f"/api/v1/gallery/{face_id}/image" if os.path.exists(abs_path) else None

    def get_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def get_person_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]

    def get_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        person_count = self._conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        return {
            "total_faces": total,
            "total_persons": person_count,
            "index_size": self._index.ntotal,
            "dimension": self.DIM,
        }

    def close(self) -> None:
        self._conn.close()
