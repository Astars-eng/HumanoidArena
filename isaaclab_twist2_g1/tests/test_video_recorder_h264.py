from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from isaaclab_twist2_g1.utils import video_recorder as video_recorder_module
from isaaclab_twist2_g1.utils.video_recorder import SimpleVideoRecorder


FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _record_test_video(path: Path, *, transcode_h264: bool | None) -> SimpleVideoRecorder:
    recorder = SimpleVideoRecorder(str(path), fps=12, transcode_h264=transcode_h264)
    for frame_index in range(7):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = frame_index * 20
        frame[8:32, 8 + frame_index:32 + frame_index, 1] = 220
        recorder.add_frame(frame)
    return recorder


def _probe_codec(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,codec_tag_string,pix_fmt,nb_frames,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)["streams"][0]


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg and ffprobe are required")
def test_h264_conversion_is_verified_before_source_is_removed(tmp_path: Path):
    source_path = tmp_path / "episode__tmp.mp4"
    final_path = tmp_path / "videos" / "episode__success.mp4"
    recorder = _record_test_video(source_path, transcode_h264=True)

    saved_path = recorder.save(output_path=final_path)

    assert saved_path == final_path
    assert final_path.is_file()
    assert not source_path.exists()
    stream = _probe_codec(final_path)
    assert stream["codec_name"] == "h264"
    assert stream["codec_tag_string"] == "avc1"
    assert stream["pix_fmt"] == "yuv420p"
    assert int(stream["nb_frames"]) == 7
    assert (int(stream["width"]), int(stream["height"])) == (64, 48)


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg and ffprobe are required")
def test_transcoding_is_on_by_default_without_environment_flag(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HUMANOIDARENA_EVAL_VIDEO_H264", raising=False)
    source_path = tmp_path / "episode__tmp.mp4"
    final_path = tmp_path / "episode__success.mp4"
    recorder = _record_test_video(source_path, transcode_h264=None)

    saved_path = recorder.save(output_path=final_path)

    assert saved_path == final_path
    assert not source_path.exists()
    assert _probe_codec(final_path)["codec_name"] == "h264"


def test_conversion_failure_preserves_original_video(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HUMANOIDARENA_FFMPEG", str(tmp_path / "missing-ffmpeg"))
    source_path = tmp_path / "episode__tmp.mp4"
    final_path = tmp_path / "videos" / "episode__success.mp4"
    recorder = _record_test_video(source_path, transcode_h264=True)

    saved_path = recorder.save(output_path=final_path)

    assert saved_path == source_path
    assert source_path.is_file()
    assert source_path.stat().st_size > 0
    assert not final_path.exists()


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg and ffprobe are required")
def test_validation_failure_preserves_source_and_removes_partial_output(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "episode__tmp.mp4"
    final_path = tmp_path / "videos" / "episode__success.mp4"
    recorder = _record_test_video(source_path, transcode_h264=True)

    def fail_validation(*_args, **_kwargs):
        raise video_recorder_module.VideoTranscodeError("injected validation failure")

    monkeypatch.setattr(video_recorder_module, "_verify_h264_video", fail_validation)
    saved_path = recorder.save(output_path=final_path)

    assert saved_path == source_path
    assert source_path.is_file()
    assert not final_path.exists()
    assert not list(final_path.parent.glob("*.h264.tmp.mp4"))


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg and ffprobe are required")
def test_source_cleanup_failure_keeps_both_valid_files(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "episode__tmp.mp4"
    final_path = tmp_path / "videos" / "episode__success.mp4"
    recorder = _record_test_video(source_path, transcode_h264=True)
    original_unlink = Path.unlink

    def fail_source_cleanup(path: Path, *args, **kwargs):
        if path == source_path:
            raise PermissionError("injected cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_source_cleanup)
    saved_path = recorder.save(output_path=final_path)

    assert saved_path == final_path
    assert source_path.is_file()
    assert final_path.is_file()
    assert _probe_codec(final_path)["codec_name"] == "h264"
