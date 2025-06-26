"""
FRBV_Fixed.py
A professional script to generate creative titles and stories using an LLM via lmstudio.
Completely rewritten with robust connection handling and proper cleanup.
"""
import os
import re
import random
import lmstudio as lms
from typing import List, Optional, Tuple
import shutil
from tqdm import tqdm
import sys
import time
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
import requests
import json
from dotenv import load_dotenv
import traceback
import subprocess
import psutil
import socket
import signal
import threading
from contextlib import contextmanager

# Load environment variables
load_dotenv()

def get_random_question() -> str:
    """Return a random question word."""
    questions = ["Who", "What", "When", "Where", "Why", "How", "Which"]
    return random.choice(questions)

def remove_think_sections(text: str) -> str:
    """Remove <think> sections and extra whitespace from the LLM output."""
    start_tag = "</think>"
    start_index = text.find(start_tag)
    if start_index != -1:
        text = text[start_index + len(start_tag):]
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def read_file(filepath: str) -> str:
    """Read and return the contents of a file."""
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

def create_output_folder(base_path: str, folder_name: str) -> str:
    """Create a new folder for each run, named after the title, and return its path."""
    # Sanitize folder name to remove invalid characters
    safe_folder_name = re.sub(r'[^\w\-_ ]', '', folder_name).strip().replace(' ', '_')
    folder_path = os.path.join(base_path, safe_folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def write_text_file(folder: str, filename: str, content: str) -> None:
    """Write content to a text file inside the specified folder."""
    file_path = os.path.join(folder, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def generate_audio_from_story(story: str, output_folder: str) -> None:
    """Generate audio narration for the story using f5-tts_infer-cli and save as gene_audio.wav."""
    import glob
    ref_audio = os.path.join(os.path.dirname(__file__), "ref_audio.mp3")
    ref_text = os.path.join(os.path.dirname(__file__), "ref_txt.txt")
    original_cwd = os.getcwd()
    try:
        os.chdir(output_folder)
        command = (
            f"f5-tts_infer-cli --model F5TTS_v1_Base "
            f"--ref_audio '{ref_audio}' "
            f"--ref_text \"$(cat '{ref_text}')\" "
            f"--gen_text \"{story.replace('\\', ' ').replace('"', '')}\" "
        )
        os.system(command)
        # Recursively find any .wav file in output_folder or subfolders
        gene_audio_path = os.path.join(output_folder, 'gene_audio.wav')
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                if file.endswith('.wav'):
                    src = os.path.join(root, file)
                    if src != gene_audio_path:
                        shutil.move(src, gene_audio_path)
                    break
        # Remove tests folder if created
        tests_path = os.path.join(output_folder, "tests")
        if os.path.exists(tests_path) and os.path.isdir(tests_path):
            shutil.rmtree(tests_path)
        print("Audio generated at normal speed")
    finally:
        os.chdir(original_cwd)

def get_random_temperature(min_temp: float = 0.3, max_temp: float = 1.2) -> float:
    """Return a random temperature value for model response."""
    return round(random.uniform(min_temp, max_temp), 2)

def silent_system(command: str) -> int:
    """Run a shell command silently (suppress output)."""
    return os.system(f"{command} > /dev/null 2>&1")

def get_audio_duration(audio_path: str) -> float:
    """Return the duration of the audio file in seconds using ffprobe or MoviePy as fallback."""
    import subprocess
    
    # Check if audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Try ffprobe first
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", audio_path
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        duration_str = result.stdout.strip()
        if duration_str and result.returncode == 0:
            return float(duration_str)
        else:
            print(f"ffprobe failed or returned empty result. Error: {result.stderr}")
            raise ValueError("ffprobe failed")
            
    except (subprocess.SubprocessError, ValueError, FileNotFoundError) as e:
        print(f"ffprobe method failed: {e}")
        print("Falling back to MoviePy for duration detection...")
        
        # Fallback to MoviePy
        try:
            from moviepy.editor import AudioFileClip
            with AudioFileClip(audio_path) as audio_clip:
                duration = audio_clip.duration
            return duration
        except Exception as moviepy_error:
            print(f"MoviePy fallback also failed: {moviepy_error}")
            raise RuntimeError(f"Could not determine duration for {audio_path}")

def get_all_video_files(videos_root: str):
    """Return a dict mapping folder names to lists of video file paths."""
    folder_to_videos = {}
    for folder in os.listdir(videos_root):
        folder_path = os.path.join(videos_root, folder)
        if os.path.isdir(folder_path):
            videos = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
            if videos:
                folder_to_videos[folder] = videos
    return folder_to_videos

def pick_non_repeating_videos(folder_to_videos, count):
    """Pick 'count' videos, no repeats, and no consecutive from same folder."""
    all_videos = [(folder, video) for folder, videos in folder_to_videos.items() for video in videos]
    if count > len(all_videos):
        raise ValueError(f"Cannot select {count} unique videos. Maximum is {len(all_videos)}.")
    # Shuffle for randomness
    random.shuffle(all_videos)
    result = []
    used_videos = set()
    last_folder = None
    for _ in range(count):
        for i, (folder, video) in enumerate(all_videos):
            if video not in used_videos and folder != last_folder:
                result.append(video)
                used_videos.add(video)
                last_folder = folder
                all_videos.pop(i)
                break
        else:
            # If not possible, just pick any unused
            for i, (folder, video) in enumerate(all_videos):
                if video not in used_videos:
                    result.append(video)
                    used_videos.add(video)
                    last_folder = folder
                    all_videos.pop(i)
                    break
    return result

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
        final_clip.write_videofile(normal_output, codec="libx264", audio_codec="aac", fps=60, verbose=False, logger=None)
        
        # Generate random speed factor between 1.5x and 1.75x
        speed_factor = round(random.uniform(1.5, 1.75), 2)
        print(f"Applying {speed_factor}x speed to video...")
        
        # Use ffmpeg to speed up the video
        sped_up_output = os.path.join(output_folder, "final_output.mp4")
        setpts_value = round(1.0 / speed_factor, 3)  # For video speed
        atempo_value = speed_factor  # For audio speed
        
        ffmpeg_command = (
            f'ffmpeg -i "{normal_output}" '
            f'-filter:v "setpts={setpts_value}*PTS" '
            f'-filter:a "atempo={atempo_value}" '
            f'-r 60 '
            f'-y "{sped_up_output}"'
        )
        
        # Run ffmpeg command silently
        result = silent_system(ffmpeg_command)
        
        if result == 0:
            print(f"Video created with {speed_factor}x speed: {sped_up_output}")
            # Remove the normal speed video to save space
            if os.path.exists(normal_output):
                os.remove(normal_output)
        else:
            print("Error applying speed change, keeping normal speed video")
            # If ffmpeg fails, rename normal speed as final output
            if os.path.exists(normal_output):
                os.rename(normal_output, sped_up_output)
        
        # Close all clips
        for clip in video_clips:
            clip.close()
        audio_clip.close()
        final_clip.close()
        print(f"Final video: {sped_up_output}")
        
    except Exception as e:
        print(f"Error during video creation: {e}")
        # Clean up any clips that were opened
        for clip in video_clips:
            try:
                clip.close()
            except:
                pass

def force_kill_processes():
    """Aggressively kill all LMStudio processes."""
    try:
        # Kill by process name patterns
        kill_patterns = ["lms", "lmstudio", "LMStudio"]
        
        for pattern in kill_patterns:
            # Try different kill methods
            silent_system(f"pkill -9 -f {pattern}")
            silent_system(f"killall -9 {pattern}")
        
        # Kill by PID if we can find any
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                proc_info = proc.info
                if proc_info['name']:
                    name_lower = proc_info['name'].lower()
                    if any(pattern in name_lower for pattern in ['lms', 'lmstudio']):
                        print(f"Force killing process: {proc_info['name']} (PID: {proc_info['pid']})")
                        proc.kill()
                        proc.wait(timeout=3)
                
                if proc_info['cmdline']:
                    cmdline_str = ' '.join(proc_info['cmdline']).lower()
                    if any(pattern in cmdline_str for pattern in ['lms', 'lmstudio']):
                        print(f"Force killing LMStudio process (PID: {proc_info['pid']})")
                        proc.kill()
                        proc.wait(timeout=3)
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, psutil.TimeoutExpired):
                pass
        
        # Additional cleanup
        time.sleep(2)
        
    except Exception as e:
        print(f"Error in force kill: {e}")

def wait_for_port_free(port: int = 1234, max_wait: int = 30) -> bool:
    """Wait for port to be completely free."""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                if result != 0:  # Port is free
                    return True
        except Exception:
            return True
        time.sleep(0.5)
    return False

def wait_for_lmstudio_ready(max_wait: int = 60) -> bool:
    """Wait for LMStudio server to be ready."""
    start_time = time.time()
    print("Waiting for LMStudio server to be ready...")
    
    while time.time() - start_time < max_wait:
        try:
            # Test the connection with a simple request
            response = requests.get("http://localhost:1234/v1/models", timeout=3)
            if response.status_code == 200:
                print("LMStudio server is ready!")
                time.sleep(1)  # Small additional wait
                return True
        except Exception:
            pass
        
        elapsed = time.time() - start_time
        print(f"Waiting for server... ({elapsed:.1f}s elapsed)")
        time.sleep(2)
    
    print(f"Timeout: LMStudio server not ready after {max_wait}s")
    return False

def start_lmstudio_fresh():
    """Start LMStudio server with complete cleanup first."""
    print("=== Starting fresh LMStudio instance ===")
    
    # Step 1: Complete cleanup
    print("Step 1: Cleaning up any existing instances...")
    try:
        subprocess.run(["lms", "server", "stop"], capture_output=True, timeout=10)
    except:
        pass
    
    force_kill_processes()
    
    # Step 2: Wait for port to be free
    print("Step 2: Waiting for port to be free...")
    if not wait_for_port_free(1234, 30):
        print("Warning: Port may still be in use")
    
    # Step 3: Start server
    print("Step 3: Starting LMStudio server...")
    try:
        # Start server in background
        process = subprocess.Popen(
            ["lms", "server", "start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it time to start
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("LMStudio server process started")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"LMStudio failed to start: {stderr}")
            return False
            
    except Exception as e:
        print(f"Error starting LMStudio: {e}")
        return False

def stop_lmstudio_complete():
    """Complete shutdown of LMStudio."""
    print("=== Stopping LMStudio ===")
    
    # Graceful shutdown first
    try:
        subprocess.run(["lms", "server", "stop"], capture_output=True, timeout=10)
        time.sleep(2)
    except:
        pass
    
    # Force cleanup
    force_kill_processes()
    
    # Wait for port to be free
    wait_for_port_free(1234, 15)
    print("LMStudio shutdown complete")

def generate_with_lmstudio(title_prompt: str, story_prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Generate title and story using LMStudio with isolated session."""
    
    # Start fresh LMStudio instance
    if not start_lmstudio_fresh():
        print("Failed to start LMStudio server")
        return None, None
    
    # Wait for server to be ready
    if not wait_for_lmstudio_ready(60):
        print("LMStudio server not ready")
        stop_lmstudio_complete()
        return None, None
    
    try:
        print("Connecting to LMStudio...")
        
        # Try to connect with retries
        model = None
        for attempt in range(3):
            try:
                print(f"Connection attempt {attempt + 1}/3")
                model = lms.llm("google/gemma-3-4b")
                print("Model loaded successfully!")
                break
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    raise RuntimeError("Failed to connect after 3 attempts")
        
        # Generate title
        print("Generating title...")
        title_messages = [
            {"role": "system", "content": title_prompt},
            {"role": "user", "content": get_random_question()}
        ]
        
        title_response = model.respond({
            "messages": title_messages,
            "temperature": get_random_temperature(),
            "stream": False
        })
        
        if not title_response or not title_response.content:
            raise RuntimeError("Empty title response")
        
        title = remove_think_sections(title_response.content)
        print(f"Title generated: {title[:50]}...")
        
        # Generate story
        print("Generating story...")
        story_content = (
            "Always keep the shared title as the first line of the story & do not make several paragraphs, "
            "give the entire story in one single paragraph\n" + title
        )
        
        story_messages = [
            {"role": "system", "content": story_prompt},
            {"role": "user", "content": story_content}
        ]
        
        story_response = model.respond({
            "messages": story_messages,
            "temperature": get_random_temperature(),
            "stream": False
        })
        
        if not story_response or not story_response.content:
            raise RuntimeError("Empty story response")
        
        story = remove_think_sections(story_response.content)
        print("Story generated successfully!")
        
        return title, story
        
    except Exception as e:
        print(f"Error during generation: {e}")
        traceback.print_exc()
        return None, None
    
    finally:
        # Always cleanup
        stop_lmstudio_complete()

def generate_text_content() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Generate title and story content. Returns (title, story, output_folder)."""
    
    try:
        # Read prompts
        title_prompt_path = "System_Title_Prompt.txt"
        story_prompt_path = "System_Story_Prompt.txt"
        
        print("Reading system prompts...")
        system_prompt_title = read_file(title_prompt_path)
        system_prompt_story = read_file(story_prompt_path)
        
        # Generate content
        title, story = generate_with_lmstudio(system_prompt_title, system_prompt_story)
        
        if not title or not story:
            print("Failed to generate content")
            return None, None, None
        
        # Create output folder
        output_base = "/media/lord/New Volume"
        output_folder = create_output_folder(output_base, title)
        
        # Save files
        write_text_file(output_folder, "title.txt", title)
        write_text_file(output_folder, "storie.txt", story)
        
        print("Content saved successfully!")
        return title, story, output_folder
        
    except Exception as e:
        print(f"Error in text generation: {e}")
        traceback.print_exc()
        return None, None, None

def run_once():
    """Run the complete pipeline once."""
    steps = [
        "Generating title and story with LLM",
        "Generating audio narration", 
        "Creating video with audio and applying random speed"
    ]
    
    with tqdm(total=len(steps), file=sys.stdout, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as pbar:
        try:
            # Phase 1: Generate text content
            pbar.set_postfix_str(steps[0])
            title, story, output_folder = generate_text_content()
            
            if not title or not story or not output_folder:
                print("Failed to generate text content. Aborting this run.")
                return False
            
            pbar.update(1)

            # Phase 2: Audio generation
            pbar.set_postfix_str(steps[1])
            generate_audio_from_story(story, output_folder)
            pbar.update(1)

            # Phase 3: Video creation
            pbar.set_postfix_str(steps[2])
            create_video_with_audio(output_folder)
            pbar.update(1)
            
            print(f"Run completed successfully! Output saved to: {output_folder}")
            return True

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    try:
        num_runs = int(input("How many times should the program run? (Enter a number): "))
    except Exception:
        num_runs = 1
    
    successful_runs = 0
    
    for i in range(num_runs):
        print(f"\n{'='*50}")
        print(f"Run {i+1} of {num_runs}")
        print(f"{'='*50}")
        
        # Ensure clean state before each run
        if i > 0:
            print("Ensuring clean state between runs...")
            stop_lmstudio_complete()
            time.sleep(5)
        
        success = run_once()
        if success:
            successful_runs += 1
        
        # Pause between runs (except after the last one)
        if i < num_runs - 1:
            print(f"\nCompleted run {i+1}. Preparing for next run...")
            time.sleep(3)
    
    print(f"\n{'='*50}")
    print(f"All runs completed!")
    print(f"Successful runs: {successful_runs}/{num_runs}")
    print(f"{'='*50}")
    
    # Final cleanup
    stop_lmstudio_complete()