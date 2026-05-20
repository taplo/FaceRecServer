from facerecserver.api.schemas import EmbeddingRequest, ApiResponse
from facerecserver.gallery.schemas import GalleryAddRequest, RecognizeRequest


class TestApiSchemas:
    def test_embedding_request_defaults(self):
        req = EmbeddingRequest()
        assert req.image is None
        assert req.image_url is None

    def test_embedding_request_with_image(self):
        req = EmbeddingRequest(image="base64data")
        assert req.image == "base64data"
        assert req.image_url is None

    def test_embedding_request_with_url(self):
        req = EmbeddingRequest(image_url="https://example.com/face.jpg")
        assert req.image_url == "https://example.com/face.jpg"

    def test_api_response_success(self):
        resp = ApiResponse(code=0, message="success", data={"key": "value"})
        assert resp.code == 0
        assert resp.data is not None

    def test_api_response_error(self):
        resp = ApiResponse(code=1001, message="未检测到人脸", data=None)
        assert resp.code == 1001
        assert resp.data is None


class TestGallerySchemas:
    def test_gallery_add_request(self):
        req = GalleryAddRequest(image="base64data", name="张三")
        assert req.image == "base64data"
        assert req.name == "张三"

    def test_gallery_add_request_defaults(self):
        req = GalleryAddRequest()
        assert req.image is None
        assert req.name is None

    def test_recognize_request(self):
        req = RecognizeRequest(image_url="https://example.com/face.jpg")
        assert req.image_url == "https://example.com/face.jpg"
