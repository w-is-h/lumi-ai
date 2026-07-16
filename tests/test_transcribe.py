"""Tests for the transcription backends."""

from unittest.mock import MagicMock, patch

from lumi import transcribe as t


def test_missing_file():
    result = t.transcribe("does_not_exist.wav")
    assert result == "[Transcription error: File not found]"


def test_empty_file(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.touch()
    result = t.transcribe(str(empty))
    assert result == "[Transcription error: File is empty]"


def test_unknown_service(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"data")
    result = t.transcribe(str(audio), service="groq")
    assert result.startswith("[Transcription error:")


def test_mlx_dispatch(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"data")
    with patch.object(t, "_transcribe_mlx", return_value="hello") as mock_mlx:
        result = t.transcribe(str(audio), service="mlx", model="some-model")
    assert result == "hello"
    mock_mlx.assert_called_once_with(str(audio), "some-model")


def test_remote(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"data")
    response = MagicMock()
    response.json.return_value = {"text": "hello from server"}
    with patch.object(t.requests, "post", return_value=response) as mock_post:
        result = t.transcribe(str(audio), service="remote")
    assert result == "hello from server"
    url = mock_post.call_args.args[0]
    assert url.endswith("/transcribe")


def test_remote_error_is_wrapped(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"data")
    with patch.object(t.requests, "post", side_effect=ConnectionError("server down")):
        result = t.transcribe(str(audio), service="remote")
    assert result.startswith("[Transcription error:")
