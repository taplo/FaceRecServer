import os
import sqlite3
import uuid
import numpy as np
import faiss
from PIL import Image
from datetime import datetime, timezone
from dataclasses import dataclass, field


@dataclass
class FaceRecord:
    face_id: str
    name: str
    created_at: str
    image_path: str = ""


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
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                image_path TEXT DEFAULT ''
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_face_id ON faces(face_id)
        """)
        self._conn.commit()

    def _load_index(self) -> None:
        rows = self._conn.execute("SELECT id FROM faces").fetchall()
        if not rows:
            return
        vec_path = self.index_path
        if os.path.exists(vec_path):
            self._index = faiss.read_index(vec_path)
        print(f"[Gallery] 加载底库: {len(rows)} 条记录")

    def _save_index(self) -> None:
        faiss.write_index(self._index, self.index_path)

    def add(self, embedding: np.ndarray, name: str, image: np.ndarray | None = None, image_path: str = "") -> str:
        face_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        normalized = embedding / np.linalg.norm(embedding)

        if image is not None:
            faces_dir = os.path.join(self.gallery_dir, "faces")
            os.makedirs(faces_dir, exist_ok=True)
            save_path = os.path.join(faces_dir, f"{face_id}.jpg")
            Image.fromarray(image).save(save_path, "JPEG")
            image_path = f"faces/{face_id}.jpg"

        cursor = self._conn.execute(
            "INSERT INTO faces (face_id, name, created_at, image_path) VALUES (?, ?, ?, ?)",
            (face_id, name, now, image_path),
        )
        faiss_id = cursor.lastrowid
        self._index.add_with_ids(normalized.reshape(1, -1).astype(np.float32), np.array([faiss_id]))
        self._conn.commit()
        self._save_index()
        return face_id

    def add_batch(self, embeddings: list, names: list, image_paths: list | None = None, images: list | None = None) -> GalleryStats:
        stats = GalleryStats(total=len(embeddings))
        for i, (emb, name) in enumerate(zip(embeddings, names)):
            try:
                img = images[i] if images else None
                path = image_paths[i] if image_paths else ""
                self.add(emb, name, image=img, image_path=path)
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

    def clear(self) -> None:
        self._conn.execute("DELETE FROM faces")
        self._conn.commit()
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIM))
        self._save_index()

    def list_faces(self, page: int = 1, page_size: int = 20, search: str = "") -> tuple:
        offset = (page - 1) * page_size
        if search:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM faces WHERE name LIKE ?", (f"%{search}%",)
            ).fetchone()[0]
            rows = self._conn.execute(
                "SELECT face_id, name, created_at, image_path FROM faces WHERE name LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (f"%{search}%", page_size, offset),
            ).fetchall()
        else:
            count = self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            rows = self._conn.execute(
                "SELECT face_id, name, created_at, image_path FROM faces ORDER BY id DESC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
        items = [FaceRecord(face_id=r[0], name=r[1], created_at=r[2], image_path=r[3]) for r in rows]
        return items, count

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if self._index.ntotal == 0:
            return []
        if top_k < 1:
            top_k = 1
        normalized = embedding / np.linalg.norm(embedding)
        query = normalized.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query, top_k)
        results = []
        for score, faiss_id in zip(distances[0], indices[0]):
            if faiss_id == -1:
                continue
            row = self._conn.execute(
                "SELECT face_id, name, image_path FROM faces WHERE id = ?", (int(faiss_id),)
            ).fetchone()
            if row:
                results.append({
                    "face_id": row[0],
                    "name": row[1],
                    "score": float(score),
                    "image_url": f"/api/v1/gallery/{row[0]}/image" if row[2] else None,
                })
        return results

    def get_image_path(self, face_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT image_path FROM faces WHERE face_id = ?", (face_id,)
        ).fetchone()
        return row[0] if row else None

    def get_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def get_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        return {
            "total_faces": total,
            "index_size": self._index.ntotal,
            "dimension": self.DIM,
        }

    def close(self) -> None:
        self._conn.close()
