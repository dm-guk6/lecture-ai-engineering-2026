from types import SimpleNamespace

import pytest

from app import ask


@pytest.mark.parametrize(
    ("usage_metadata", "expected"),
    [
        (SimpleNamespace(prompt_token_count=12, candidates_token_count=4), "input=12 output=4"),
        (None, "input=N/A output=N/A"),
        (
            SimpleNamespace(prompt_token_count=None, candidates_token_count=4),
            "input=N/A output=4",
        ),
    ],
)
def test_ask_prints_token_counts(monkeypatch, capsys, usage_metadata, expected):
    response = SimpleNamespace(text="answer", usage_metadata=usage_metadata)

    class FakeModels:
        def generate_content(self, **_):
            return response

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(ask.genai, "Client", lambda **_: FakeClient())
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("question") == "answer"
    assert f"[tokens] {expected}" in capsys.readouterr().out
