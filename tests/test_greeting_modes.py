"""Greeting per session mode: offline keeps today's line, online gets its own.

The online greeting is spoken on a cloud cascade over the public documents
only, so it must never claim offline or private operation.
"""

import bot_browser
import config
from bot_browser import OFFLINE_MODE
from pipeline import ONLINE_MODE


def test_online_greeting_differs_from_offline(monkeypatch):
    """Online names the online/cloud session; it never says "offline"."""
    monkeypatch.delenv("LOFI_GREETING", raising=False)
    monkeypatch.delenv("LOFI_ONLINE_GREETING", raising=False)

    online = config.online_greeting()

    assert online != config.greeting()
    assert "offline" not in online.lower()
    assert "online" in online.lower()


def test_online_greeting_env_override(monkeypatch):
    """`LOFI_ONLINE_GREETING` wins over the tunables default."""
    monkeypatch.setenv("LOFI_ONLINE_GREETING", "hi from the cloud")

    assert config.online_greeting() == "hi from the cloud"


def test_greeting_for_mode(monkeypatch):
    """`_greeting(mode)` picks the mode's line; anything else is offline."""
    monkeypatch.delenv("LOFI_GREETING", raising=False)
    monkeypatch.delenv("LOFI_ONLINE_GREETING", raising=False)

    assert bot_browser._greeting(ONLINE_MODE) == config.online_greeting()
    assert bot_browser._greeting(OFFLINE_MODE) == config.greeting()
    assert bot_browser._greeting("garbage") == config.greeting()
