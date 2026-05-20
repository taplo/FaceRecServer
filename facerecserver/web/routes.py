import os
from fastapi.responses import FileResponse, JSONResponse


def mount_frontend(app):
    """Mount built Vue frontend static files and SPA catch-all."""
    dist_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "frontend", "dist"
    )
    if not os.path.isdir(dist_dir):
        print(f"[Web] 前端构建目录不存在: {dist_dir}")
        print("[Web] 请先执行: cd frontend && npm run build")
        return

    from fastapi.staticfiles import StaticFiles
    assets_dir = os.path.join(dist_dir, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"code": 404, "message": "Not found"})
        index_path = os.path.join(dist_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path, media_type="text/html")
        return JSONResponse(status_code=404, content={"code": 404, "message": "Not found"})
