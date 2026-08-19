#!/usr/bin/env python3
"""Combine two videos into one side-by-side video with FFmpeg."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play two videos side by side in a single output video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("left_video", type=Path, help="video shown on the left")
    parser.add_argument("right_video", type=Path, help="video shown on the right")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("side_by_side.mp4"),
        help="output video path",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="height of each video; by default, use the left video's height",
    )
    parser.add_argument(
        "--audio",
        choices=("auto", "left", "right", "none"),
        default="auto",
        help="audio track to keep; auto prefers the left video",
    )
    parser.add_argument(
        "--longest",
        action="store_true",
        help="continue until the longer video ends (the last frame is repeated)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="output frame rate",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality (lower is better; normally 18-28)",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ),
        help="H.264 encoding speed/size trade-off",
    )
    parser.add_argument("-y", "--overwrite", action="store_true", help="overwrite output")
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg executable name or path",
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="FFprobe executable name or path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the FFmpeg command without running it",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for label, path in (("left", args.left_video), ("right", args.right_video)):
        if not path.is_file():
            raise ValueError(f"{label} video does not exist or is not a file: {path}")

    input_paths = {args.left_video.resolve(), args.right_video.resolve()}
    if args.output.resolve() in input_paths:
        raise ValueError("output path must be different from both input paths")
    if args.output.exists() and not args.overwrite:
        raise ValueError(f"output already exists (pass --overwrite to replace it): {args.output}")
    if args.height is not None and (args.height <= 0 or args.height % 2 != 0):
        raise ValueError("--height must be a positive even number")
    if not 0 < args.fps <= 240:
        raise ValueError("--fps must be greater than 0 and at most 240")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51")
    if shutil.which(args.ffmpeg) is None:
        raise ValueError(f"FFmpeg executable not found: {args.ffmpeg}")
    if shutil.which(args.ffprobe) is None:
        raise ValueError(f"FFprobe executable not found: {args.ffprobe}")


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    has_audio: bool


def probe_video(ffprobe: str, path: Path) -> VideoInfo:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    metadata = json.loads(result.stdout)
    streams = metadata.get("streams", [])
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), None
    )
    if video_stream is None:
        raise ValueError(f"input contains no video stream: {path}")

    duration_text = video_stream.get("duration") or metadata.get("format", {}).get("duration")
    try:
        duration = float(duration_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"could not determine video duration: {path}") from exc
    if duration <= 0:
        raise ValueError(f"video duration must be positive: {path}")

    return VideoInfo(
        duration=duration,
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def build_video_filter(height: int | None, longest: bool, fps: float) -> str:
    shortest = 0 if longest else 1
    # Extend the last frame past the source's final timestamp; -t later trims it to the
    # probed duration. This avoids a one- or two-frame visual gap at the end.
    stack_filter = (
        f"hstack=inputs=2:shortest={shortest},"
        f"fps={fps:g},tpad=stop_mode=clone:stop_duration=1[video]"
    )
    if height is not None:
        # Use display aspect ratio (dar), then square pixels, to preserve anamorphic videos.
        return (
            f"[0:v:0]setpts=PTS-STARTPTS,"
            f"scale=w=trunc({height}*dar/2)*2:h={height},setsar=1[left];"
            f"[1:v:0]setpts=PTS-STARTPTS,"
            f"scale=w=trunc({height}*dar/2)*2:h={height},setsar=1[right];"
            f"[left][right]{stack_filter}"
        )

    # Normalize the left input to an even height, then scale the right input to match it.
    return (
        "[0:v:0]setpts=PTS-STARTPTS,"
        "scale=w=trunc(oh*dar/2)*2:h=trunc(ih/2)*2,setsar=1[left];"
        "[1:v:0]setpts=PTS-STARTPTS[right_source];"
        "[right_source][left]scale2ref="
        "w=trunc(oh*mdar/2)*2:h=ih[right][left_ref];"
        "[right]setsar=1[right_square];"
        f"[left_ref][right_square]{stack_filter}"
    )


def choose_audio_input(
    requested_audio: str, left_info: VideoInfo, right_info: VideoInfo
) -> int | None:
    if requested_audio == "none":
        return None
    if requested_audio == "auto":
        if left_info.has_audio:
            return 0
        if right_info.has_audio:
            return 1
        return None

    selected_input = 0 if requested_audio == "left" else 1
    selected_info = left_info if selected_input == 0 else right_info
    if not selected_info.has_audio:
        side = "left" if selected_input == 0 else "right"
        print(f"warning: {side} video has no audio; creating a silent output", file=sys.stderr)
        return None
    return selected_input


def build_command(
    args: argparse.Namespace, output_duration: float, audio_input: int | None
) -> list[str]:
    command = [
        args.ffmpeg,
        "-hide_banner",
        "-y" if args.overwrite else "-n",
        "-i",
        str(args.left_video),
        "-i",
        str(args.right_video),
        "-filter_complex",
        build_video_filter(args.height, args.longest, args.fps),
        "-map",
        "[video]",
    ]

    if audio_input is not None:
        command.extend(
            [
                "-map",
                f"{audio_input}:a:0",
                # Padding prevents a short audio track from ending the video early.
                "-af",
                "asetpts=PTS-STARTPTS,apad",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        )

    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-t",
            f"{output_duration:.6f}",
            str(args.output),
        ]
    )
    return command


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        left_info = probe_video(args.ffprobe, args.left_video)
        right_info = probe_video(args.ffprobe, args.right_video)
        durations = (left_info.duration, right_info.duration)
        output_duration = max(durations) if args.longest else min(durations)
        audio_input = choose_audio_input(args.audio, left_info, right_info)
        command = build_command(args, output_duration, audio_input)
        print("Running:", shlex.join(command), flush=True)
        if args.dry_run:
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Created: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
