#!/usr/bin/env python3
"""
AI-Powered Video & Audio Upscaler

A local-first CLI tool for upscaling video using Real-ESRGAN and audio using AudioSR,
optimized for Apple Silicon (M4) with Metal/MPS acceleration.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import tempfile
from pathlib import Path

# Must be set before importing torch so MPS fallback ops work
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import click
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def probe_file(input_path: str) -> dict:
    """Run ffprobe and return parsed metadata."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        input_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        console.print("[bold red]Error:[/] ffprobe not found. Install FFmpeg first (brew install ffmpeg).")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        console.print(f"[bold red]Error:[/] ffprobe failed: {exc.stderr}")
        sys.exit(1)

    data = json.loads(result.stdout)

    video_stream = None
    audio_stream = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and video_stream is None:
            video_stream = s
        elif s.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = s

    if video_stream is None:
        console.print("[bold red]Error:[/] No video stream found in input file.")
        sys.exit(1)

    # Parse FPS from fraction string like "30000/1001"
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    duration = float(data.get("format", {}).get("duration", 0))

    info = {
        "width": width,
        "height": height,
        "fps": fps,
        "fps_str": fps_str,
        "duration": duration,
        "video_codec": video_stream.get("codec_name", "unknown"),
        "audio_sample_rate": int(audio_stream.get("sample_rate", 44100)) if audio_stream else None,
        "audio_codec": audio_stream.get("codec_name", "unknown") if audio_stream else None,
        "has_audio": audio_stream is not None,
    }
    return info


def print_probe_summary(input_path: str, info: dict) -> None:
    """Print a Rich table summarising the input file."""
    table = Table(title="Input File Analysis", show_header=False, border_style="cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("File", str(input_path))
    table.add_row("Resolution", f"{info['width']}x{info['height']}")
    table.add_row("FPS", f"{info['fps']:.3f} ({info['fps_str']})")
    table.add_row("Duration", f"{info['duration']:.2f}s")
    table.add_row("Video Codec", info["video_codec"])
    if info["has_audio"]:
        table.add_row("Audio Codec", info["audio_codec"])
        table.add_row("Audio Sample Rate", f"{info['audio_sample_rate']} Hz")
    else:
        table.add_row("Audio", "None detected")

    console.print(table)


def extract_audio(input_path: str, temp_dir: str, sample_rate: int, duration: float = None) -> str:
    """Extract audio track to WAV."""
    output_wav = os.path.join(temp_dir, "audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "2",
    ]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd.append(output_wav)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[yellow]Warning:[/] Audio extraction failed: {result.stderr.splitlines()[-1] if result.stderr else 'unknown error'}")
        return ""
    return output_wav


def upscale_audio_ai(input_wav: str, temp_dir: str) -> str:
    """Upscale audio using AudioSR."""
    output_wav = os.path.join(temp_dir, "audio_enhanced.wav")
    try:
        import audiosr
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        console.print(f"[yellow]Warning:[/] {exc}. Falling back to resample mode.")
        console.print("[dim]Install with: pip install audiosr soundfile[/]")
        return ""

    with console.status("[bold cyan]Upscaling audio with AudioSR (this may take a while)...", spinner="dots"):
        try:
            model = audiosr.build_model(model_name="basic", device="auto")
            waveform = audiosr.super_resolution(
                model,
                input_wav,
                seed=42,
                guidance_scale=3.5,
                ddim_steps=50,
                latent_t_per_second=12.8,
            )
            # waveform is a numpy array — squeeze batch/channel dims
            if hasattr(waveform, "numpy"):
                waveform = waveform.numpy()
            waveform = np.squeeze(waveform)
            if waveform.ndim > 1:
                waveform = waveform.T  # soundfile expects (samples, channels)
            sf.write(output_wav, waveform, 48000)
        except Exception as exc:
            console.print(f"[yellow]Warning:[/] AudioSR processing failed: {exc}")
            console.print("[yellow]Falling back to resample mode.[/]")
            return ""

    if os.path.exists(output_wav):
        console.print("[green]Audio upscaled with AudioSR to 48 kHz.[/]")
        return output_wav
    return ""


def upscale_audio_resample(input_wav: str, temp_dir: str) -> str:
    """Upscale audio via SoX (preferred) or FFmpeg resampler."""
    output_wav = os.path.join(temp_dir, "audio_enhanced.wav")

    # Try SoX first
    sox_path = shutil.which("sox")
    if sox_path:
        cmd = [sox_path, input_wav, "-r", "48000", output_wav, "rate", "-v", "-L"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            console.print("[green]Audio resampled to 48 kHz using SoX.[/]")
            return output_wav
        console.print(f"[yellow]Warning:[/] SoX failed ({result.stderr.strip()}), trying FFmpeg.")

    # Fall back to FFmpeg
    cmd = [
        "ffmpeg", "-y", "-i", input_wav,
        "-af", "aresample=resampler=soxr",
        "-ar", "48000",
        output_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        console.print("[green]Audio resampled to 48 kHz using FFmpeg.[/]")
        return output_wav

    console.print("[yellow]Warning:[/] Audio resampling failed. The original audio will be used.")
    return input_wav  # Return original as last resort


def build_upsampler(scale: int, face_enhance: bool, model_mode: str):
    """Build Real-ESRGAN upsampler (and optional GFPGAN face enhancer)."""
    try:
        import torch
        from realesrgan import RealESRGANer
    except ImportError as exc:
        console.print(f"[bold red]Error:[/] Missing dependency: {exc}")
        console.print("Run: pip install realesrgan basicsr torch torchvision")
        sys.exit(1)

    # Model selection
    if model_mode == "fast":
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=16, upscale=4, act_type="prelu",
        )
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        netscale = 4
        console.print("[cyan]Model: SRVGGNetCompact (fast — optimised for video)[/]")
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        if scale == 2:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2,
            )
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            netscale = 2
        else:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            )
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            netscale = 4
        console.print("[cyan]Model: RRDBNet (quality — best detail, slower)[/]")

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        console.print("[green]Using Metal/MPS GPU acceleration.[/]")
    else:
        device = torch.device("cpu")
        console.print("[yellow]MPS not available — falling back to CPU (this will be slow).[/]")

    # Half precision is NOT supported on MPS
    use_half = False

    console.print("[dim]Model weights will be downloaded automatically on first run.[/]")

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_url,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=device,
    )

    face_enhancer = None
    if face_enhance:
        try:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                upscale=scale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )
            console.print("[green]GFPGAN face enhancement enabled.[/]")
        except ImportError:
            console.print("[yellow]Warning:[/] gfpgan not installed. Face enhancement disabled.")
            console.print("Install with: pip install gfpgan")
        except Exception as exc:
            console.print(f"[yellow]Warning:[/] GFPGAN setup failed: {exc}")

    return upsampler, face_enhancer, netscale


def warmup_model(upsampler, face_enhancer, scale: int) -> None:
    """Run a tiny dummy frame through the model to trigger MPS shader compilation."""
    import cv2
    import numpy as np

    console.print("[dim]Warming up model (MPS shader compilation — this is a one-time cost)...[/]")
    dummy = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    t0 = time.time()
    if face_enhancer is not None:
        face_enhancer.enhance(dummy, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        upsampler.enhance(dummy, outscale=scale)
    elapsed = time.time() - t0
    console.print(f"[dim]Warmup complete in {elapsed:.1f}s.[/]")


def upscale_video_frames(
    input_path: str,
    temp_dir: str,
    scale: int,
    upsampler,
    face_enhancer,
    netscale: int,
    denoise: float,
    info: dict,
    max_frames: int = None,
):
    """Read every frame, upscale it, and write to temp PNGs."""
    import cv2
    import signal

    # Allow Ctrl+C to work even during MPS operations
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(1))

    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        console.print("[bold red]Error:[/] Could not open video with OpenCV.")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = int(info["duration"] * info["fps"])
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    orig_w, orig_h = info["width"], info["height"]
    target_w = orig_w * scale
    target_h = orig_h * scale
    needs_resize = (scale == 3)  # Upsampled at 4x, need to resize to 3x

    # Warmup: compile MPS shaders on a tiny image before starting the real work
    warmup_model(upsampler, face_enhancer, scale)

    console.print(f"[bold]Upscaling {total_frames} frames: {orig_w}x{orig_h} -> {target_w}x{target_h}[/]")

    # Process first frame separately and report timing for ETA estimate
    ret, frame = cap.read()
    if not ret:
        console.print("[bold red]Error:[/] Could not read first frame.")
        cap.release()
        sys.exit(1)

    console.print("[dim]Processing first frame...[/]")
    t0 = time.time()
    output = _enhance_frame(frame, upsampler, face_enhancer, scale, 0)
    first_frame_time = time.time() - t0

    if needs_resize:
        output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(os.path.join(frames_dir, "frame_00000000.png"), output)
    est_total = first_frame_time * total_frames
    est_h, est_m = divmod(int(est_total), 3600)
    est_m //= 60
    console.print(
        f"[green]First frame done in {first_frame_time:.1f}s.[/] "
        f"Estimated total: ~{est_h}h {est_m}m for {total_frames} frames."
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Upscaling video", total=total_frames, completed=1)

        frame_idx = 1
        while True:
            if frame_idx >= total_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            try:
                output = _enhance_frame(frame, upsampler, face_enhancer, scale, frame_idx)
            except RuntimeError as exc:
                error_msg = str(exc).lower()
                if "out of memory" in error_msg or "mps" in error_msg:
                    console.print(
                        "\n[bold red]Error:[/] Out of memory during upscaling. "
                        "Try reducing the tile size or using a smaller scale factor."
                    )
                else:
                    console.print(f"\n[bold red]Error:[/] Frame {frame_idx} failed: {exc}")
                cap.release()
                sys.exit(1)
            except Exception as exc:
                console.print(f"\n[bold red]Error:[/] Frame {frame_idx} failed: {exc}")
                cap.release()
                sys.exit(1)

            if needs_resize:
                output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            out_path = os.path.join(frames_dir, f"frame_{frame_idx:08d}.png")
            cv2.imwrite(out_path, output)

            frame_idx += 1
            progress.update(task, completed=frame_idx)

    cap.release()
    console.print(f"[green]Upscaled {frame_idx} frames.[/]")
    return frames_dir, target_w, target_h


def _enhance_frame(frame, upsampler, face_enhancer, scale: int, frame_idx: int):
    """Enhance a single frame with Real-ESRGAN or GFPGAN."""
    if face_enhancer is not None:
        _, _, output = face_enhancer.enhance(
            frame, has_aligned=False, only_center_face=False, paste_back=True,
        )
    else:
        output, _ = upsampler.enhance(frame, outscale=scale)
    return output


def reassemble_video(
    frames_dir: str,
    audio_path: str,
    output_path: str,
    fps: float,
    codec: str,
):
    """Mux upscaled frames + enhanced audio into the final video file."""

    # Determine encoder settings
    hw_encoder = None
    sw_encoder = None
    if codec == "h265":
        hw_encoder = "hevc_videotoolbox"
        sw_encoder = "libx265"
    else:
        hw_encoder = "h264_videotoolbox"
        sw_encoder = "libx264"

    def _build_cmd(encoder: str, is_hw: bool) -> list:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%08d.png"),
        ]
        if audio_path and os.path.exists(audio_path):
            cmd += ["-i", audio_path]

        cmd += ["-c:v", encoder]

        if is_hw:
            cmd += ["-q:v", "65"]
        else:
            cmd += ["-crf", "18", "-preset", "medium"]

        cmd += ["-pix_fmt", "yuv420p"]

        if audio_path and os.path.exists(audio_path):
            cmd += ["-c:a", "aac", "-b:a", "192k"]

        cmd += ["-movflags", "+faststart", output_path]
        return cmd

    # Use software encoder for maximum compatibility (QuickTime, etc.)
    console.print(f"[dim]Encoding with {sw_encoder}...[/]")
    cmd = _build_cmd(sw_encoder, is_hw=False)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        console.print(f"[green]Video encoded with {sw_encoder}.[/]")
        return

    # Fall back to hardware encoder if software fails
    console.print(f"[yellow]Software encoder failed, trying hardware encoder ({hw_encoder})...[/]")
    cmd = _build_cmd(hw_encoder, is_hw=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[bold red]Error:[/] FFmpeg encoding failed:\n{result.stderr}")
        sys.exit(1)
    console.print(f"[green]Video encoded with hardware encoder ({hw_encoder}).[/]")


def format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Human-readable duration."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output", "output_path", type=click.Path(), default=None,
    help="Output file path [default: {input}_upscaled.mp4]",
)
@click.option(
    "-s", "--scale", type=click.Choice(["2", "3", "4"]), default="2",
    help="Upscale factor [default: 2]",
)
@click.option(
    "--audio-mode", type=click.Choice(["ai", "resample"]), default="ai",
    help="Audio upscaling method [default: ai]",
)
@click.option(
    "--codec", type=click.Choice(["h265", "h264"]), default="h265",
    help="Output video codec [default: h265]",
)
@click.option(
    "--model", type=click.Choice(["fast", "quality"]), default="fast",
    help="Model: fast (~1h) or quality (~9h for 11min video) [default: fast]",
)
@click.option(
    "--face-enhance", is_flag=True, default=False,
    help="Enable GFPGAN face restoration",
)
@click.option(
    "--denoise", type=float, default=0.5,
    help="Denoise strength 0.0-1.0 [default: 0.5]",
)
@click.option(
    "--duration", type=float, default=None,
    help="Only process the first N seconds (useful for testing)",
)
def main(input_path, output_path, scale, audio_mode, codec, model, face_enhance, denoise, duration):
    """AI-powered video and audio upscaler.

    Upscales INPUT_PATH using Real-ESRGAN (video) and AudioSR (audio),
    optimised for Apple Silicon with Metal/MPS acceleration.
    """
    scale = int(scale)
    start_time = time.time()

    console.print()
    console.rule("[bold cyan]AI Video Upscaler[/]")
    console.print()

    # -----------------------------------------------------------------------
    # Step 1: Probe input
    # -----------------------------------------------------------------------
    console.print("[bold]Step 1/7:[/] Analysing input file...")
    info = probe_file(input_path)
    print_probe_summary(input_path, info)

    max_frames = int(duration * info["fps"]) if duration else None
    if duration:
        console.print(f"[cyan]Duration limit: {duration}s ({max_frames} frames)[/]")

    # Resolve output path
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_stem(p.stem + "_upscaled").with_suffix(".mp4"))
    console.print(f"[dim]Output: {output_path}[/]")
    console.print()

    # -----------------------------------------------------------------------
    # Step 2: Create temp directory
    # -----------------------------------------------------------------------
    temp_dir = tempfile.mkdtemp(prefix="upscaler_")
    console.print(f"[dim]Temp directory: {temp_dir}[/]")

    try:
        # -------------------------------------------------------------------
        # Step 3: Extract audio
        # -------------------------------------------------------------------
        audio_wav = ""
        if info["has_audio"]:
            console.print("[bold]Step 3/7:[/] Extracting audio...")
            audio_wav = extract_audio(input_path, temp_dir, info["audio_sample_rate"], duration)
            if audio_wav:
                console.print("[green]Audio extracted.[/]")
            else:
                console.print("[yellow]Skipping audio processing.[/]")
        else:
            console.print("[bold]Step 3/7:[/] No audio track — skipping extraction.")

        # -------------------------------------------------------------------
        # Step 4: Upscale audio
        # -------------------------------------------------------------------
        enhanced_audio = ""
        if audio_wav:
            console.print(f"[bold]Step 4/7:[/] Upscaling audio (mode: {audio_mode})...")
            if audio_mode == "ai":
                enhanced_audio = upscale_audio_ai(audio_wav, temp_dir)
                if not enhanced_audio:
                    console.print("[dim]AI upscale unavailable — resampling instead.[/]")
                    enhanced_audio = upscale_audio_resample(audio_wav, temp_dir)
            else:
                enhanced_audio = upscale_audio_resample(audio_wav, temp_dir)
        else:
            console.print("[bold]Step 4/7:[/] No audio to upscale — skipping.")

        console.print()

        # -------------------------------------------------------------------
        # Step 5: Upscale video
        # -------------------------------------------------------------------
        console.print("[bold]Step 5/7:[/] Building upscaling model...")
        upsampler, face_enhancer_model, netscale = build_upsampler(scale, face_enhance, model)

        console.print("[bold]Step 5/7:[/] Upscaling video frames...")
        frames_dir, target_w, target_h = upscale_video_frames(
            input_path, temp_dir, scale,
            upsampler, face_enhancer_model, netscale,
            denoise, info, max_frames,
        )
        console.print()

        # -------------------------------------------------------------------
        # Step 6: Reassemble
        # -------------------------------------------------------------------
        console.print("[bold]Step 6/7:[/] Encoding final video...")
        reassemble_video(
            frames_dir,
            enhanced_audio,
            output_path,
            info["fps"],
            codec,
        )
        console.print()

    finally:
        # -------------------------------------------------------------------
        # Step 7: Cleanup
        # -------------------------------------------------------------------
        console.print("[bold]Step 7/7:[/] Cleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
            console.print("[green]Temp files removed.[/]")
        except Exception as exc:
            console.print(f"[yellow]Warning:[/] Could not remove temp dir: {exc}")

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    console.print()
    console.rule("[bold green]Done[/]")

    result_table = Table(title="Upscale Complete", show_header=False, border_style="green")
    result_table.add_column("Property", style="bold")
    result_table.add_column("Value")

    result_table.add_row("Output File", str(output_path))
    result_table.add_row("Output Resolution", f"{target_w}x{target_h}")

    if os.path.exists(output_path):
        result_table.add_row("File Size", format_size(os.path.getsize(output_path)))
    else:
        result_table.add_row("File Size", "[red]Output file not found[/]")

    result_table.add_row("Processing Time", format_duration(elapsed))

    console.print(result_table)
    console.print()


if __name__ == "__main__":
    main()
