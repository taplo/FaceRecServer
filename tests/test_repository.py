import pytest
import numpy as np
from facerecserver.gallery.repository import GalleryRepository


class TestGalleryRepository:

    def test_init_creates_db_and_index(self, gallery_dir):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        assert repo.get_count() == 0
        repo.close()

    def test_add_face(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        face_id = repo.add(sample_embedding, "测试用户")
        assert face_id is not None
        assert len(face_id) > 0  # UUID
        assert repo.get_count() == 1
        repo.close()

    def test_add_face_with_image(self, gallery_dir, sample_embedding, sample_image):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        face_id = repo.add(sample_embedding, "测试用户", image=sample_image)
        # Check image was saved
        image_path = repo.get_image_path(face_id)
        assert image_path is not None
        assert image_path.endswith(".jpg")
        assert "faces" in image_path
        repo.close()

    def test_list_faces_pagination(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        for i in range(5):
            repo.add(sample_embedding, f"用户{i}")
        items, total = repo.list_faces(page=1, page_size=2)
        assert total == 5
        assert len(items) == 2
        repo.close()

    def test_list_faces_search(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        repo.add(sample_embedding, "张三")
        repo.add(sample_embedding, "李四")
        repo.add(sample_embedding, "张伟")
        items, total = repo.list_faces(search="张")
        assert total == 2
        assert all("张" in item.name for item in items)
        repo.close()

    def test_delete_face(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        face_id = repo.add(sample_embedding, "测试用户")
        assert repo.get_count() == 1
        result = repo.delete(face_id)
        assert result is True
        assert repo.get_count() == 0
        repo.close()

    def test_delete_nonexistent(self, gallery_dir):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        result = repo.delete("nonexistent-id")
        assert result is False
        repo.close()

    def test_clear_gallery(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        repo.add(sample_embedding, "用户1")
        repo.add(sample_embedding, "用户2")
        assert repo.get_count() == 2
        repo.clear()
        assert repo.get_count() == 0
        repo.close()

    def test_search_returns_sorted_by_score(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        # Add different embeddings
        rng = np.random.default_rng(1)
        for name in ["A", "B", "C"]:
            emb = rng.random(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            repo.add(emb, name)
        # Search with first embedding
        results = repo.search(sample_embedding, top_k=3)
        assert len(results) <= 3
        # Scores should be descending
        scores = [r["score"] for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        repo.close()

    def test_search_empty_gallery(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        results = repo.search(sample_embedding, top_k=5)
        assert results == []
        repo.close()

    def test_search_results_have_image_url(self, gallery_dir, sample_embedding, sample_image):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        face_id = repo.add(sample_embedding, "测试用户", image=sample_image)
        results = repo.search(sample_embedding, top_k=1)
        assert len(results) == 1
        assert results[0]["image_url"] is not None
        assert face_id in results[0]["image_url"]
        repo.close()

    def test_search_image_url_null_when_no_image(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        repo.add(sample_embedding, "测试用户")
        results = repo.search(sample_embedding, top_k=1)
        assert len(results) == 1
        assert results[0]["image_url"] is None
        repo.close()

    def test_get_image_path(self, gallery_dir, sample_embedding, sample_image):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        face_id = repo.add(sample_embedding, "测试用户", image=sample_image)
        path = repo.get_image_path(face_id)
        assert path is not None
        # Verify file exists on disk
        import os
        abs_path = os.path.join(gallery_dir, path)
        assert os.path.exists(abs_path)
        repo.close()

    def test_get_image_path_nonexistent(self, gallery_dir):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        path = repo.get_image_path("nonexistent-id")
        assert path is None
        repo.close()

    def test_get_stats(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        repo.add(sample_embedding, "用户1")
        repo.add(sample_embedding, "用户2")
        stats = repo.get_stats()
        assert stats["total_faces"] == 2
        assert stats["index_size"] == 2
        assert stats["dimension"] == 512
        repo.close()

    def test_add_batch(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        embs = [sample_embedding.copy() for _ in range(3)]
        names = ["批量1", "批量2", "批量3"]
        stats = repo.add_batch(embs, names)
        assert stats.succeeded == 3
        assert stats.failed == 0
        assert repo.get_count() == 3
        repo.close()

    def test_add_batch_with_images(self, gallery_dir, sample_embedding, sample_image):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        embs = [sample_embedding.copy() for _ in range(2)]
        names = ["批量1", "批量2"]
        imgs = [sample_image.copy(), sample_image.copy()]
        stats = repo.add_batch(embs, names, images=imgs)
        assert stats.succeeded == 2
        # Verify images saved
        all_faces, total = repo.list_faces(page=1, page_size=10)
        for face in all_faces:
            path = repo.get_image_path(face.face_id)
            assert path is not None
        repo.close()

    def test_add_batch_empty(self, gallery_dir):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        stats = repo.add_batch([], [])
        assert stats.succeeded == 0
        repo.close()

    def test_persistence_across_reload(self, gallery_dir, sample_embedding):
        db_name = "persist_test"
        repo = GalleryRepository(gallery_dir, db_name)
        repo.add(sample_embedding, "持久化用户")
        repo.close()
        # Reload from same directory
        repo2 = GalleryRepository(gallery_dir, db_name)
        assert repo2.get_count() == 1
        items, total = repo2.list_faces()
        assert items[0].name == "持久化用户"
        repo2.close()

    def test_top_k_clamped(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        repo.add(sample_embedding, "用户1")
        # top_k = 0 should be clamped to 1
        results = repo.search(sample_embedding, top_k=0)
        assert len(results) == 1
        repo.close()

    def test_person_auto_created(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        repo.add(sample_embedding, "张三", employee_id="E001")
        assert repo.get_person_count() == 1
        repo.close()

    def test_person_reused_on_same_name(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        repo.add(sample_embedding, "张三", employee_id="E001")
        repo.add(sample_embedding, "张三", employee_id="E001")
        assert repo.get_person_count() == 1
        assert repo.get_count() == 2
        repo.close()

    def test_list_persons(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        repo.add(sample_embedding, "张三", employee_id="E001")
        repo.add(sample_embedding, "李四", employee_id="E002")
        repo.add(sample_embedding, "张三", employee_id="E001")
        items, total = repo.list_persons()
        assert total == 2
        for p in items:
            if p.name == "张三":
                assert p.face_count == 2
            elif p.name == "李四":
                assert p.face_count == 1
        repo.close()

    def test_delete_person_cascade(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        repo.add(sample_embedding, "张三", employee_id="E001")
        repo.add(sample_embedding, "张三", employee_id="E001")
        persons, _ = repo.list_persons()
        person_id = persons[0].person_id
        repo.delete_person(person_id)
        assert repo.get_count() == 0
        assert repo.get_person_count() == 0
        repo.close()

    def test_search_aggregates_by_person(self, gallery_dir):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        rng = np.random.default_rng(1)
        id_embs = []
        for _ in range(5):
            emb = rng.random(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            id_embs.append(emb)
        repo.add(id_embs[0], "张三", employee_id="E001")
        repo.add(id_embs[1], "张三", employee_id="E001")
        repo.add(id_embs[2], "李四", employee_id="E002")
        results = repo.search(id_embs[0], top_k=5)
        names = [r["name"] for r in results]
        assert names.count("张三") == 1
        assert names.count("李四") == 1
        repo.close()

    def test_get_stats_includes_persons(self, gallery_dir, sample_embedding):
        repo = GalleryRepository(gallery_dir, "test_gallery")
        repo.add(sample_embedding, "张三")
        repo.add(sample_embedding, "李四")
        stats = repo.get_stats()
        assert stats["total_persons"] == 2
        assert stats["total_faces"] == 2
        repo.close()
