import json

from bot_moq import CONFIG_MESSAGE_TYPE, _lofi_config_message


def test_lofi_config_message_shape():
    message = _lofi_config_message()
    json.dumps(message)
    assert message["type"] == CONFIG_MESSAGE_TYPE == "lofi-config"
    assert [m["role"] for m in message["models"]] == [
        "STT",
        "LLM",
        "TTS",
        "EMBED",
    ]
    for entry in message["models"]:
        assert entry["model"]
        assert entry["path"].lstrip().startswith("→")
    assert message["ollama_host"].startswith("http")
    assert isinstance(message["ollama_started_by_lofi"], bool)
    assert message["rag"].startswith(("index:", "no index"))
