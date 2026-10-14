"""python -m nepkev.service: serve the API with uvicorn (host and port from NEPKEV_HOST / NEPKEV_PORT)."""

import logging
import os

import uvicorn


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # one line per Hub request while the model downloads
    uvicorn.run("nepkev.service.app:create_app", factory=True, host=os.environ.get("NEPKEV_HOST", "127.0.0.1"),
                port=int(os.environ.get("NEPKEV_PORT", "8000")), proxy_headers=True)


if __name__ == "__main__":
    main()
