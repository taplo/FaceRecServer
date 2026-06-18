import logging
import os
from facerecserver.app import create_app
from facerecserver.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    config = load_config()
    app = create_app(config)
    import uvicorn
    port = int(os.environ.get("UVICORN_PORT", config.server.port))
    uvicorn.run(app, host=config.server.host, port=port)


if __name__ == "__main__":
    main()
