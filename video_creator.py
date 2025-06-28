"""
video_creator.py
Handles video creation, editing, and speed modification.
"""
import os
import random
import subprocess
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip

from utils import get_audio_duration
from video_utils import get_all_video_files, pick_non_repeating_videos


def create_video_with_audio(output_folder: str):
    """Create a video with random background footage matching the audio duration, then randomly speed it up."""
    videos_root = os.path.join(os.path.dirname(__file__), "Videos")
    audio_path = os.path.join(output_folder, "gene_audio.wav")
    
    # Check if audio file exists before proceeding
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        print("Make sure the audio generation step completed successfully.")
        return
    
    try:
        audio_duration = get_audio_duration(audio_path)
        print(f"Creating video to match audio duration: {audio_duration:.2f}s")
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return
    
    # Get all videos
    folder_to_videos = get_all_video_files(videos_root)
    if not folder_to_videos:
        print(f"Error: No video files found in {videos_root}")
        return
        
    # Decide how many clips to use
    count = int(audio_duration // 5) + 1  # e.g. 1 clip per 5 seconds
    try:
        selected_videos = pick_non_repeating_videos(folder_to_videos, count)
    except ValueError as e:
        print(e)
        return
    
    # Load video clips and trim if needed
    video_clips = []
    total_duration = 0
    for video_path in selected_videos:
        try:
            clip = VideoFileClip(video_path)
            video_clips.append(clip)
            total_duration += clip.duration
            if total_duration >= audio_duration:
                break
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            continue
    
    if not video_clips:
        print("Error: No video clips could be loaded")
        return
    
    try:
        # Concatenate and trim to audio duration
        final_clip = concatenate_videoclips(video_clips).subclip(0, audio_duration)
        
        # Load the original audio with MoviePy
        audio_clip = AudioFileClip(audio_path)
        
        # Combine video and audio
        final_clip = final_clip.set_audio(audio_clip)
        
        # Save the normal speed video first at 60 fps
        normal_output = os.path.join(output_folder, "normal_speed.mp4")
        final_clip.write_videofile(normal_output, codec="libx264", audio_codec="aac", 
                                   fps=60, verbose=False, logger=None)
        
        # Apply random speed
        _apply_random_speed(normal_output, output_folder)
        
        # Close all clips
        for clip in video_clips:
            clip.close()
        audio_clip.close()
        final_clip.close()
        
    except Exception as e:
        print(f"Error during video creation: {e}")
        # Clean up any clips that were opened
        for clip in video_clips:
            try:
                clip.close()
            except:
                pass


def _apply_random_speed(normal_output: str, output_folder: str):
    """Apply random speed between 1.5x and 1.75x to the video with proper timeout handling."""
    # Generate random speed factor between 1.5x and 1.75x
    speed_factor = round(random.uniform(1.5, 1.75), 2)
    print(f"Applying {speed_factor}x speed to video...")
    
    # Use ffmpeg to speed up the video
    sped_up_output = os.path.join(output_folder, "final_output.mp4")
    setpts_value = round(1.0 / speed_factor, 3)  # For video speed
    atempo_value = speed_factor  # For audio speed
    
    # Optimized FFmpeg command for AMD RX 6600 XT hardware acceleration
    ffmpeg_command = [
        'ffmpeg', '-i', normal_output,
        '-filter:v', f'setpts={setpts_value}*PTS',
        '-filter:a', f'atempo={atempo_value}',
        '-r', '60',
        '-c:v', 'h264_amf',  # Use AMD hardware encoder
        '-quality', 'quality',  # High quality mode for AMF
        '-rc', 'cqp',  # Constant quantization parameter
        '-qp_i', '20',  # I-frame quality (lower = better)
        '-qp_p', '23',  # P-frame quality
        '-threads', '12',  # Still use CPU threads for non-encoding tasks
        '-y', sped_up_output
    ]
    
    try:
        print("Processing video speed change... This may take a few minutes for large files.")
        
        # Run ffmpeg with hardware acceleration (should be very fast with RX 6600 XT)
        result = subprocess.run(
            ffmpeg_command,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes timeout (hardware encoding is much faster)
            check=False
        )
        
        if result.returncode == 0:
            print(f"✓ Video created with {speed_factor}x speed: {sped_up_output}")
            # Remove the normal speed video to save space
            if os.path.exists(normal_output):
                os.remove(normal_output)
        else:
            print("✗ Hardware encoding failed, trying CPU fallback...")
            if result.stderr:
                print(f"AMF error: {result.stderr}")
                
            # Try CPU fallback
            if _apply_random_speed_cpu_fallback(normal_output, output_folder, speed_factor):
                return
                
            # If both fail, rename normal speed as final output
            if os.path.exists(normal_output):
                os.rename(normal_output, sped_up_output)
                
    except subprocess.TimeoutExpired:
        print("✗ Hardware encoding timed out after 2 minutes.")
        print("Falling back to CPU encoding...")
        # Try CPU fallback
        if _apply_random_speed_cpu_fallback(normal_output, output_folder, speed_factor):
            return
        print("Keeping normal speed video as final output.")
        if os.path.exists(normal_output):
            os.rename(normal_output, sped_up_output)
            
    except FileNotFoundError:
        print("✗ FFmpeg not found. Please make sure FFmpeg is installed and in your PATH.")
        print("Keeping normal speed video as final output.")
        if os.path.exists(normal_output):
            os.rename(normal_output, sped_up_output)
            
    except Exception as e:
        print(f"✗ Unexpected error during hardware encoding: {e}")
        print("Trying CPU fallback...")
        if _apply_random_speed_cpu_fallback(normal_output, output_folder, speed_factor):
            return
        print("Keeping normal speed video as final output.")
        if os.path.exists(normal_output):
            os.rename(normal_output, sped_up_output)
    
    print(f"Final video: {sped_up_output}")


def _apply_random_speed_cpu_fallback(normal_output: str, output_folder: str, speed_factor: float):
    """CPU fallback for when hardware encoding fails."""
    print("Using CPU encoding as fallback...")
    
    sped_up_output = os.path.join(output_folder, "final_output.mp4")
    setpts_value = round(1.0 / speed_factor, 3)
    atempo_value = speed_factor
    
    # CPU-based FFmpeg command
    ffmpeg_command = [
        'ffmpeg', '-i', normal_output,
        '-filter:v', f'setpts={setpts_value}*PTS',
        '-filter:a', f'atempo={atempo_value}',
        '-r', '60',
        '-c:v', 'libx264',  # Use CPU encoder
        '-preset', 'fast',
        '-crf', '23',
        '-threads', '12',
        '-y', sped_up_output
    ]
    
    try:
        result = subprocess.run(
            ffmpeg_command,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes for CPU encoding
            check=False
        )
        
        if result.returncode == 0:
            print(f"✓ CPU fallback successful: {speed_factor}x speed applied")
            if os.path.exists(normal_output):
                os.remove(normal_output)
            return True
        else:
            print("✗ CPU fallback also failed")
            if result.stderr:
                print(f"CPU encoding error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ CPU fallback error: {e}")
        return False
    
    print(f"Final video: {sped_up_output}")


def _apply_random_speed_moviepy(normal_output: str, output_folder: str):
    """Alternative method using MoviePy for speed changes (slower but more reliable)."""
    speed_factor = round(random.uniform(1.5, 1.75), 2)
    print(f"Applying {speed_factor}x speed using MoviePy...")
    
    sped_up_output = os.path.join(output_folder, "final_output.mp4")
    
    try:
        # Load the video and apply speed change
        clip = VideoFileClip(normal_output)
        sped_clip = clip.fx(lambda gf: gf.speedx(speed_factor))
        
        # Write the sped up video
        sped_clip.write_videofile(
            sped_up_output,
            codec="libx264",
            audio_codec="aac",
            fps=60,
            verbose=False,
            logger=None
        )
        
        # Clean up
        clip.close()
        sped_clip.close()
        
        # Remove normal speed video
        if os.path.exists(normal_output):
            os.remove(normal_output)
            
        print(f"✓ Video created with {speed_factor}x speed using MoviePy: {sped_up_output}")
        
    except Exception as e:
        print(f"✗ MoviePy speed change failed: {e}")
        print("Keeping normal speed video as final output.")
        if os.path.exists(normal_output):
            os.rename(normal_output, sped_up_output)