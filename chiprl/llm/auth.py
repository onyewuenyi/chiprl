"""Credential handling for the Claude API. The rules:

1. No key ever lives in the repo, a config file, a log, or a trajectory. `redact()` runs on every
   string that gets persisted.
2. Preferred: `ant auth login` (OAuth profile, short-lived tokens, nothing static on disk to leak).
   The SDK's zero-arg `Anthropic()` picks it up automatically.
3. If you use an API key: create it in a dedicated `chiprl` Console workspace with a monthly spend
   limit, store it in the macOS Keychain, and export it only into the shell that needs it
   (see docs/SECURITY.md). Never put it in .env files that a tool might commit.
4. A stale exported ANTHROPIC_API_KEY silently outranks a profile; we warn about that.
5. Hard per-run spend cap in code (costs.Budget), on top of the Console workspace limit.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

_KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")


class CredentialError(RuntimeError):
    pass


def redact(text: str) -> str:
    """Strip anything shaped like an Anthropic key. Applied to every persisted string."""
    return _KEY_RE.sub("sk-ant-***REDACTED***", text)


def _profile_exists() -> bool:
    cfg = Path(os.environ.get("ANTHROPIC_CONFIG_DIR", Path.home() / ".config/anthropic"))
    return (cfg / "credentials").is_dir() and any((cfg / "credentials").glob("*.json"))


def credential_source() -> str:
    """Which credential the SDK will use, without ever reading the secret value into a log."""
    if "ANTHROPIC_API_KEY" in os.environ:
        if not os.environ["ANTHROPIC_API_KEY"].strip():
            raise CredentialError("ANTHROPIC_API_KEY is set but empty; unset it to use your ant profile.")
        if _profile_exists():
            warnings.warn(
                "ANTHROPIC_API_KEY is exported and outranks your `ant auth login` profile.",
                stacklevel=2,
            )
        return "api_key_env"
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "auth_token_env"
    if _profile_exists() or os.environ.get("ANTHROPIC_PROFILE"):
        return "ant_profile"
    if os.environ.get("ANTHROPIC_FEDERATION_RULE_ID"):
        return "workload_identity_federation"
    raise CredentialError(
        "No Claude credentials found. Run `ant auth login` (preferred), or load a key from the "
        "Keychain: see docs/SECURITY.md."
    )


def make_client(max_retries: int = 4, timeout_s: float = 600.0):
    """Anthropic client from the ambient credential chain. Never accepts a key argument."""
    import anthropic

    credential_source()  # fail fast with a useful message
    return anthropic.Anthropic(max_retries=max_retries, timeout=timeout_s)
