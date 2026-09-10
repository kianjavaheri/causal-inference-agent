"""Runtime configuration, read from the environment."""

import os


class Settings:
    """App settings. All values are optional so the demo runs with an empty env."""

    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
    # Reports get a cheaper/faster model by default; planning is the hard reasoning step.
    anthropic_report_model: str = os.getenv("ANTHROPIC_REPORT_MODEL", "claude-sonnet-5")

    # Comma-separated list, or "*" for any origin.
    # Trailing slashes are stripped: browsers send Origin without one, so a pasted
    # "https://app.vercel.app/" would otherwise silently fail every CORS check.
    cors_origins: list[str] = [
        o.strip().rstrip("/")
        for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]

    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
    max_rows: int = int(os.getenv("MAX_ROWS", 250_000))
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", 60 * 60 * 4))

    @property
    def llm_enabled(self) -> bool:
        return self.anthropic_api_key is not None


settings = Settings()
