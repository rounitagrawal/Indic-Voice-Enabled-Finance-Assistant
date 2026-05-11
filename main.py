"""
main.py
───────
Application entry point.

Run locally:
    python main.py

Run via Docker:
    docker-compose up

The host/port/debug settings are all controlled by environment variables.
No hardcoded IP addresses. No hardcoded debug flags.
"""
from src.api.app import create_app
from src.config import load_config

if __name__ == "__main__":
    config = load_config()
    app = create_app(config)

    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
    )
