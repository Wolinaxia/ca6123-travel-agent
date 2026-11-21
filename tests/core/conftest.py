import pytest


@pytest.fixture(autouse=True)
def block_real_http(monkeypatch):
    """Keep the suite offline; source adapters are patched or use replay fixtures."""
    import travel_agent.core.http

    def blocked(*args, **kwargs):
        raise AssertionError("real outbound HTTP disabled in tests")

    monkeypatch.setattr(travel_agent.core.http._client, "request", blocked)
    import travel_agent.core.sources.news

    monkeypatch.setattr(travel_agent.core.sources.news.time, "sleep", lambda seconds: None)
