"""
Video Recorder for multi-view recording during OpenPI verification.

Records and combines:
- First-person view (from robot camera)
- Third-person view (world camera)
- SMPL visualization
"""

import json
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class VideoTranscodeError(RuntimeError):
    """Raised when a recorded video cannot be safely converted to H.264."""


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_rate(value: object) -> float:
    if value in (None, "", "0/0"):
        return 0.0
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _parse_number(value: object) -> Optional[float]:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_video_program(name: str, env_name: str) -> str:
    configured = os.getenv(env_name, "").strip()
    resolved = configured or shutil.which(name)
    if not resolved:
        raise VideoTranscodeError(
            f"{name} was not found; set {env_name} to an executable path"
        )
    return resolved


def _run_video_program(args: list[str], *, timeout: float, operation: str) -> subprocess.CompletedProcess:
    try:
        completed = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VideoTranscodeError(f"{operation} failed: {exc}") from exc
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        if len(details) > 4000:
            details = details[-4000:]
        raise VideoTranscodeError(
            f"{operation} exited with code {completed.returncode}"
            + (f": {details}" if details else "")
        )
    return completed


def _probe_video(video_path: Path, ffprobe_path: str) -> dict:
    completed = _run_video_program(
        [
            ffprobe_path,
            "-v",
            "error",
            "-threads",
            "1",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=codec_name,codec_tag_string,pix_fmt,width,height,"
                "r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration:"
                "format=duration"
            ),
            "-of",
            "json",
            str(video_path),
        ],
        timeout=120.0,
        operation=f"probing {video_path}",
    )
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise VideoTranscodeError(f"No readable video stream found in {video_path}") from exc

    duration = _parse_number(stream.get("duration"))
    if duration is None:
        duration = _parse_number(payload.get("format", {}).get("duration"))
    frame_count = _parse_number(stream.get("nb_read_frames"))
    if frame_count is None:
        frame_count = _parse_number(stream.get("nb_frames"))
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    return {
        "codec_name": stream.get("codec_name", ""),
        "codec_tag_string": stream.get("codec_tag_string", ""),
        "pix_fmt": stream.get("pix_fmt", ""),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": fps,
        "frame_count": int(frame_count) if frame_count is not None else None,
        "duration": duration,
    }


def _verify_h264_video(source_info: dict, output_path: Path, ffmpeg_path: str, ffprobe_path: str) -> None:
    output_info = _probe_video(output_path, ffprobe_path)
    problems = []
    if output_info["codec_name"] != "h264":
        problems.append(f"codec={output_info['codec_name']!r}")
    if output_info["codec_tag_string"] != "avc1":
        problems.append(f"codec_tag={output_info['codec_tag_string']!r}")
    if output_info["pix_fmt"] != "yuv420p":
        problems.append(f"pix_fmt={output_info['pix_fmt']!r}")

    expected_width = source_info["width"] + source_info["width"] % 2
    expected_height = source_info["height"] + source_info["height"] % 2
    if (output_info["width"], output_info["height"]) != (expected_width, expected_height):
        problems.append(
            f"size={output_info['width']}x{output_info['height']} "
            f"expected={expected_width}x{expected_height}"
        )

    source_fps = source_info["fps"]
    output_fps = output_info["fps"]
    if source_fps > 0 and (output_fps <= 0 or abs(source_fps - output_fps) > 0.01):
        problems.append(f"fps={output_fps:g} expected={source_fps:g}")

    source_frames = source_info["frame_count"]
    output_frames = output_info["frame_count"]
    if source_frames is not None and output_frames != source_frames:
        problems.append(f"frames={output_frames} expected={source_frames}")

    source_duration = source_info["duration"]
    output_duration = output_info["duration"]
    if source_duration is not None:
        duration_tolerance = max(0.1, 1.5 / source_fps) if source_fps > 0 else 0.1
        if output_duration is None or abs(source_duration - output_duration) > duration_tolerance:
            problems.append(f"duration={output_duration} expected={source_duration}")

    if problems:
        raise VideoTranscodeError(
            f"H.264 validation failed for {output_path}: " + ", ".join(problems)
        )

    duration = output_duration or source_duration or 0.0
    _run_video_program(
        [
            ffmpeg_path,
            "-v",
            "error",
            "-xerror",
            "-threads",
            "1",
            "-i",
            str(output_path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        timeout=max(120.0, duration * 5.0),
        operation=f"decoding {output_path}",
    )


def transcode_mp4_to_h264(source_path: str | Path, output_path: str | Path) -> Path:
    """Convert MP4 to browser-compatible H.264 and delete the source only after validation."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise VideoTranscodeError(f"Source video is missing or empty: {source_path}")

    ffmpeg_path = _find_video_program("ffmpeg", "HUMANOIDARENA_FFMPEG")
    ffprobe_path = _find_video_program("ffprobe", "HUMANOIDARENA_FFPROBE")
    source_info = _probe_video(source_path, ffprobe_path)
    if source_info["width"] <= 0 or source_info["height"] <= 0:
        raise VideoTranscodeError(f"Source video has invalid dimensions: {source_path}")

    try:
        crf = int(os.getenv("HUMANOIDARENA_H264_CRF", "20"))
    except ValueError as exc:
        raise VideoTranscodeError("HUMANOIDARENA_H264_CRF must be an integer") from exc
    if not 0 <= crf <= 51:
        raise VideoTranscodeError("HUMANOIDARENA_H264_CRF must be between 0 and 51")
    preset = os.getenv("HUMANOIDARENA_H264_PRESET", "veryfast").strip() or "veryfast"
    try:
        threads = int(os.getenv("HUMANOIDARENA_H264_THREADS", "2"))
    except ValueError as exc:
        raise VideoTranscodeError("HUMANOIDARENA_H264_THREADS must be an integer") from exc
    if not 1 <= threads <= 64:
        raise VideoTranscodeError("HUMANOIDARENA_H264_THREADS must be between 1 and 64")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.",
        suffix=".h264.tmp.mp4",
        dir=output_path.parent,
        delete=False,
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()
    temp_path.unlink()

    try:
        duration = source_info["duration"] or 0.0
        _run_video_program(
            [
                ffmpeg_path,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-threads",
                "1",
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-threads",
                str(threads),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temp_path),
            ],
            timeout=max(120.0, duration * 5.0),
            operation=f"transcoding {source_path} to H.264",
        )
        if not temp_path.is_file() or temp_path.stat().st_size == 0:
            raise VideoTranscodeError(f"H.264 output is missing or empty: {temp_path}")
        _verify_h264_video(source_info, temp_path, ffmpeg_path, ffprobe_path)

        validated_size = temp_path.stat().st_size
        os.replace(temp_path, output_path)
        if not output_path.is_file() or output_path.stat().st_size != validated_size:
            raise VideoTranscodeError(f"Validated H.264 video was not published correctly: {output_path}")

        if source_path.resolve() != output_path.resolve():
            try:
                source_path.unlink()
            except OSError as exc:
                print(
                    f"[video_recorder] H.264 video is valid, but source cleanup failed; "
                    f"keeping both files: source={source_path} output={output_path} error={exc}"
                )
        return output_path
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


class VideoRecorder:
    """
    Multi-view video recorder that combines multiple video streams.
    """

    def __init__(self, save_dir: str, fps: int = 30, enable_smpl_vis: bool = True,
                 smpl_visualizer=None):
        """
        Initialize video recorder.

        Args:
            save_dir: directory to save videos
            fps: frames per second
            enable_smpl_vis: if True, include SMPL visualization in output
            smpl_visualizer: SMPLVisualizer instance (optional, created if None and enable_smpl_vis=True)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.enable_smpl_vis = enable_smpl_vis

        # Frame storage
        self.first_person_frames = []
        self.third_person_frames = []
        self.smpl_states = []  # Store SMPL states instead of pre-rendered frames

        # SMPL visualizer
        self.smpl_visualizer = smpl_visualizer
        if enable_smpl_vis and smpl_visualizer is None:
            from .smpl_visualizer import create_smpl_visualizer
            self.smpl_visualizer = create_smpl_visualizer(use_simple=True, resolution=(640, 480))

        # Video writer (created when first frame is added)
        self.video_writer = None
        self.canvas_size = None

    def add_frame(self, first_person_img: Optional[np.ndarray] = None,
                  third_person_img: Optional[np.ndarray] = None,
                  smpl_state: Optional[np.ndarray] = None):
        """
        Add a frame to the recording.

        Args:
            first_person_img: (H, W, 3) RGB image from robot camera
            third_person_img: (H, W, 3) RGB image from world camera
            smpl_state: (75,) SMPL state array
        """
        # Store frames
        if first_person_img is not None:
            # Ensure uint8 format
            if first_person_img.dtype != np.uint8:
                first_person_img = (first_person_img * 255).astype(np.uint8) if first_person_img.max() <= 1.0 else first_person_img.astype(np.uint8)
            self.first_person_frames.append(first_person_img.copy())
        else:
            # Create placeholder if not provided
            if len(self.first_person_frames) > 0:
                placeholder = np.zeros_like(self.first_person_frames[0])
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            self.first_person_frames.append(placeholder)

        if third_person_img is not None:
            if third_person_img.dtype != np.uint8:
                third_person_img = (third_person_img * 255).astype(np.uint8) if third_person_img.max() <= 1.0 else third_person_img.astype(np.uint8)
            self.third_person_frames.append(third_person_img.copy())
        else:
            if len(self.third_person_frames) > 0:
                placeholder = np.zeros_like(self.third_person_frames[0])
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            self.third_person_frames.append(placeholder)

        if smpl_state is not None and self.enable_smpl_vis:
            self.smpl_states.append(smpl_state.copy())
        elif self.enable_smpl_vis:
            # Placeholder SMPL state
            self.smpl_states.append(np.zeros(75, dtype=np.float32))

    def save(self, output_path: Optional[str] = None, name: str = "output"):
        """
        Render SMPL frames and save combined video.

        Args:
            output_path: full path to output video file (optional)
            name: base name for video if output_path not specified
        """
        if len(self.first_person_frames) == 0:
            print("No frames to save!")
            return

        # Determine output path
        if output_path is None:
            output_path = self.save_dir / f"{name}.mp4"
        else:
            output_path = Path(output_path)

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Rendering and saving video to {output_path}...")

        # Get frame dimensions
        h, w = self.first_person_frames[0].shape[:2]

        # Resize all frames to same height if needed
        target_height = h
        resized_first_person = self._resize_frames(self.first_person_frames, target_height)
        resized_third_person = self._resize_frames(self.third_person_frames, target_height)

        # Render SMPL frames if enabled
        smpl_frames = []
        if self.enable_smpl_vis and self.smpl_visualizer is not None:
            print(f"Rendering {len(self.smpl_states)} SMPL frames...")
            for i, smpl_state in enumerate(self.smpl_states):
                smpl_img = self.smpl_visualizer.render(smpl_state)
                # Resize to target height
                smpl_h, smpl_w = smpl_img.shape[:2]
                if smpl_h != target_height:
                    scale = target_height / smpl_h
                    smpl_img = cv2.resize(smpl_img, (int(smpl_w * scale), target_height))
                smpl_frames.append(smpl_img)
            print("SMPL rendering complete.")

        # Calculate canvas size
        first_w = resized_first_person[0].shape[1]
        third_w = resized_third_person[0].shape[1]
        if len(smpl_frames) > 0:
            smpl_w = smpl_frames[0].shape[1]
            canvas_width = first_w + third_w + smpl_w
        else:
            canvas_width = first_w + third_w

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (canvas_width, target_height))

        # Combine and write frames
        num_frames = len(resized_first_person)
        print(f"Writing {num_frames} frames...")
        for i in range(num_frames):
            canvas = np.zeros((target_height, canvas_width, 3), dtype=np.uint8)

            # Place first-person view
            fp_img = resized_first_person[i]
            canvas[:, :first_w] = cv2.cvtColor(fp_img, cv2.COLOR_RGB2BGR)

            # Place third-person view
            tp_img = resized_third_person[i]
            canvas[:, first_w:first_w+third_w] = cv2.cvtColor(tp_img, cv2.COLOR_RGB2BGR)

            # Place SMPL visualization if enabled
            if len(smpl_frames) > 0 and i < len(smpl_frames):
                smpl_img = smpl_frames[i]
                canvas[:, first_w+third_w:] = cv2.cvtColor(smpl_img, cv2.COLOR_RGB2BGR)

            writer.write(canvas)

        writer.release()
        print(f"Video saved successfully: {output_path}")
        print(f"  - Resolution: {canvas_width}x{target_height}")
        print(f"  - FPS: {self.fps}")
        print(f"  - Frames: {num_frames}")
        print(f"  - Duration: {num_frames/self.fps:.2f}s")

    def _resize_frames(self, frames, target_height):
        """Resize frames to target height while maintaining aspect ratio."""
        resized = []
        for frame in frames:
            h, w = frame.shape[:2]
            if h != target_height:
                scale = target_height / h
                new_w = int(w * scale)
                resized_frame = cv2.resize(frame, (new_w, target_height))
                resized.append(resized_frame)
            else:
                resized.append(frame)
        return resized

    def clear(self):
        """Clear all stored frames."""
        self.first_person_frames = []
        self.third_person_frames = []
        self.smpl_states = []
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def close(self):
        """Clean up resources."""
        if self.video_writer is not None:
            self.video_writer.release()
        if self.smpl_visualizer is not None:
            self.smpl_visualizer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SimpleVideoRecorder:
    """
    Simple single-stream recorder that writes frames incrementally to disk.
    This avoids holding an entire episode video in RAM during persistent eval.
    """

    def __init__(self, save_path: str, fps: int = 30, transcode_h264: Optional[bool] = None):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.transcode_h264 = (
            _env_flag("HUMANOIDARENA_EVAL_VIDEO_H264", default=True)
            if transcode_h264 is None
            else bool(transcode_h264)
        )
        self.frames = []  # compatibility sentinel for legacy callers
        self.writer = None
        self.frame_count = 0
        self._frame_size = None

    def _ensure_writer(self, img: np.ndarray):
        h, w = img.shape[:2]
        size = (w, h)
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(str(self.save_path), fourcc, self.fps, size)
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(f"Failed to open VideoWriter for: {self.save_path}")
            self._frame_size = size
        elif self._frame_size != size:
            img = cv2.resize(img, self._frame_size)
        return img

    def add_frame(self, img: np.ndarray):
        """Add a frame to the recording and write it immediately."""
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        img = self._ensure_writer(img)
        self.writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if self.frame_count == 0:
            self.frames = [True]
        self.frame_count += 1

    def save(self, output_path: Optional[str] = None) -> Optional[Path]:
        """Finalize the writer and optionally move the temp video to a final path."""
        if self.frame_count == 0:
            print("No frames to save!")
            return None
        self.close()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.transcode_h264:
                source_path = self.save_path
                try:
                    self.save_path = transcode_mp4_to_h264(source_path, output_path)
                    print(
                        f"[video_recorder] H.264 conversion verified: "
                        f"{self.save_path} (frames={self.frame_count})"
                    )
                    return self.save_path
                except VideoTranscodeError as exc:
                    print(
                        f"[video_recorder] H.264 conversion failed; preserving original video "
                        f"at {source_path}: {exc}"
                    )
                    return source_path
            if output_path != self.save_path:
                if output_path.exists():
                    output_path.unlink()
                os.replace(self.save_path, output_path)
                self.save_path = output_path
        print(f"Video saved: {self.save_path} (frames={self.frame_count})")
        return self.save_path

    def clear(self):
        self.frames = []
        self.frame_count = 0
        self._frame_size = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


if __name__ == "__main__":
    # Quick test
    print("Testing VideoRecorder...")

    # Create test frames
    num_frames = 10
    h, w = 480, 640

    first_person_frames = []
    third_person_frames = []
    smpl_states = []

    for i in range(num_frames):
        # Create colored test frames
        fp_frame = np.ones((h, w, 3), dtype=np.uint8) * (255 * i // num_frames)
        fp_frame[:, :, 0] = 255  # Red channel
        first_person_frames.append(fp_frame)

        tp_frame = np.ones((h, w, 3), dtype=np.uint8) * (255 * i // num_frames)
        tp_frame[:, :, 1] = 255  # Green channel
        third_person_frames.append(tp_frame)

        # Create test SMPL state
        smpl_state = np.zeros(75, dtype=np.float32)
        smpl_state[74] = 0.9 + 0.1 * np.sin(2 * np.pi * i / num_frames)  # Varying height
        smpl_states.append(smpl_state)

    # Test recorder
    recorder = VideoRecorder(save_dir="./test_videos", fps=10, enable_smpl_vis=True)

    for i in range(num_frames):
        recorder.add_frame(
            first_person_img=first_person_frames[i],
            third_person_img=third_person_frames[i],
            smpl_state=smpl_states[i]
        )

    recorder.save(name="test_output")
    recorder.close()

    print("VideoRecorder test complete!")
