"""
FRBV.py
A professional script to generate creative titles and stories using an LLM via lmstudio.
Updated to include random video speed between 1.5x and 1.75x and SRT subtitle generation.
Fixed: Unload model and close LMStudio immediately after text generation.
"""
import os
import re
import random
import lmstudio as lms
from typing import List
import shutil
from tqdm import tqdm
import sys
import time
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
import requests
import json
from dotenv import load_dotenv
import assemblyai as aai
import traceback

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

def generate_srt_from_video(output_folder: str) -> None:
    """Generate SRT subtitle file from the final video using AssemblyAI SDK."""
    api_key = os.getenv('ASSEMBLY_AI_API_KEY')
    if not api_key:
        print("Warning: ASSEMBLY_AI_API_KEY not found in environment variables. Skipping SRT generation.")
        return

    video_path = os.path.join(output_folder, "final_output.mp4")
    if not os.path.exists(video_path):
        print(f"Error: Final video not found at {video_path}")
        return

    print("Generating SRT subtitles using AssemblyAI SDK...")

    # Extract audio from video for transcription
    temp_audio_path = os.path.join(output_folder, "temp_for_transcription.wav")
    try:
        with VideoFileClip(video_path) as video_clip:
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(temp_audio_path, verbose=False, logger=None)
            audio_clip.close()
    except Exception as e:
        print(f"Error extracting audio for transcription: {e}")
        return

    srt_path = os.path.join(output_folder, "subtitles.srt")
    try:
        aai.settings.api_key = api_key
        transcriber = aai.Transcriber()
        print("Uploading and transcribing audio with AssemblyAI SDK...")
        transcript = transcriber.transcribe(temp_audio_path)
        
        # Fixed: Use polling instead of wait_till_complete
        print("Waiting for transcription to complete...")
        while transcript.status not in ['completed', 'error']:
            time.sleep(5)  # Wait 5 seconds before checking again
            transcript = transcriber.get_transcript(transcript.id)
        
        if transcript.status != 'completed':
            print(f"Transcription failed: {transcript.status}")
            if hasattr(transcript, 'error'):
                print(f"Error details: {transcript.error}")
            return

        # Export SRT subtitles
        srt_content = transcript.export_subtitles_srt()
        if not srt_content or not isinstance(srt_content, str) or len(srt_content.strip()) == 0:
            print("Warning: AssemblyAI SDK did not return SRT content. No subtitles will be saved.")
        else:
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            print(f"SRT file generated: {srt_path}")
            if not os.path.exists(srt_path):
                print("Warning: SRT file was not created as expected.")
                
    except Exception as e:
        print(f"Error during SRT generation: {e}")
        # Try alternative approach if the SDK method fails
        try:
            print("Attempting alternative transcription method...")
            # Use the transcript data directly if available
            if hasattr(transcript, 'words') and transcript.words:
                generate_srt_from_words(transcript.words, srt_path)
                print(f"SRT file generated using alternative method: {srt_path}")
            else:
                print("No word-level data available for SRT generation")
        except Exception as alt_e:
            print(f"Alternative SRT generation also failed: {alt_e}")

    finally:
        # Clean up temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def generate_srt_from_words(words, srt_path: str) -> None:
    """Generate SRT file from word-level transcript data."""
    if not words:
        print("No word-level timestamps available")
        return
    
    srt_content = []
    subtitle_index = 1
    
    # Group words into subtitles (approximately 5-8 words per subtitle)
    words_per_subtitle = 6
    
    for i in range(0, len(words), words_per_subtitle):
        word_group = words[i:i + words_per_subtitle]
        
        start_time = word_group[0].start / 1000.0  # Convert ms to seconds
        end_time = word_group[-1].end / 1000.0
        
        # Format timestamps for SRT (HH:MM:SS,mmm)
        start_srt = format_timestamp_for_srt(start_time)
        end_srt = format_timestamp_for_srt(end_time)
        
        # Combine words into text
        text = ' '.join([word.text for word in word_group])
        
        # Add to SRT content
        srt_content.append(f"{subtitle_index}")
        srt_content.append(f"{start_srt} --> {end_srt}")
        srt_content.append(text)
        srt_content.append("")  # Empty line between subtitles
        
        subtitle_index += 1
    
    # Write SRT file
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_content))

def format_timestamp_for_srt(seconds: float) -> str:
    """Format timestamp in seconds to SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

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

def safe_model_operations(model, operation_name: str, operation_func):
    """Safely perform model operations with error handling."""
    try:
        return operation_func()
    except Exception as e:
        print(f"Error during {operation_name}: {e}")
        # Try to gracefully handle the error
        if "websocket" in str(e).lower() or "connection" in str(e).lower():
            print("Connection issue detected. Attempting to continue...")
            # Don't raise the error, just log it
            return None
        else:
            raise e

def wait_for_lmstudio_server(max_wait=30, interval=2):
    """Wait for the lmstudio server to be ready before proceeding."""
    import socket
    start = time.time()
    while time.time() - start < max_wait:
        try:
            # Try to connect to the default lmstudio websocket port
            with socket.create_connection(("localhost", 1234), timeout=2):
                return True
        except Exception:
            time.sleep(interval)
    return False

def cleanup_lmstudio(model):
    """Properly cleanup LMStudio model and server."""
    try:
        if model:
            print("Unloading model...")
            model.unload()
    except Exception as e:
        print(f"Error unloading model: {e}")
    
    try:
        print("Stopping LMStudio server...")
        silent_system("lms server stop")
        # Give it a moment to properly shut down
        time.sleep(2)
    except Exception as e:
        print(f"Error stopping server: {e}")

def generate_text_content():
    """Generate title and story content using LMStudio. Returns (title, story, output_folder)."""
    model = None
    
    try:
        print("Starting LMStudio server...")
        silent_system("lms server start")

        # Wait for server to be ready before loading model
        if not wait_for_lmstudio_server():
            raise RuntimeError("Timed out waiting for lmstudio server to be ready.")

        print("Loading model...")
        model = lms.llm("google/gemma-3-4b")

        # File paths
        title_prompt_path = "System_Title_Prompt.txt"
        story_prompt_path = "System_Story_Prompt.txt"
        
        print("Reading system prompts...")
        system_prompt_title = read_file(title_prompt_path)
        system_prompt_story = read_file(story_prompt_path)

        # Generate random input for title
        random_question = get_random_question()
        title_prompt = f"{random_question}"
        messages_for_title = [
            {"role": "system", "content": system_prompt_title},
            {"role": "user", "content": title_prompt}
        ]
        title_temp = get_random_temperature()
        
        print("Generating title...")
        # Safe title generation
        def generate_title():
            return model.respond({
                "messages": messages_for_title,
                "temperature": title_temp,
                "stream": False
            }).content
        
        response_title = safe_model_operations(model, "title generation", generate_title)
        if response_title is None:
            print("Failed to generate title. Using fallback.")
            response_title = f"Generated Story - {random_question}"
        
        cleaned_title = remove_think_sections(response_title)

        # Create output folder for this run
        output_base = "/media/lord/New Volume"
        output_folder = create_output_folder(output_base, cleaned_title)
        write_text_file(output_folder, "title.txt", cleaned_title)
        print("Title saved successfully")

        # Prepare messages for story generation
        story_user_content = (
            "Always keep the shared title as the first line of the story & do not make several paragraphs, "
            "give the entire story in one single paragraph\n" + cleaned_title
        )
        messages_for_story = [
            {"role": "system", "content": system_prompt_story},
            {"role": "user", "content": story_user_content}
        ]
        story_temp = get_random_temperature()
        
        print("Generating story...")
        # Safe story generation
        def generate_story():
            return model.respond({
                "messages": messages_for_story,
                "temperature": story_temp,
                "stream": False
            }).content
        
        response_story = safe_model_operations(model, "story generation", generate_story)
        if response_story is None:
            print("Failed to generate story. Using fallback.")
            response_story = f"{cleaned_title}\n\nThis is a generated story about the topic above."
        
        cleaned_story = remove_think_sections(response_story)
        write_text_file(output_folder, "storie.txt", cleaned_story)
        print("Story saved successfully")
        
        return cleaned_title, cleaned_story, output_folder
        
    except Exception as e:
        print(f"Error during text generation: {e}")
        traceback.print_exc()
        return None, None, None
    finally:
        # Always cleanup LMStudio
        cleanup_lmstudio(model)

def run_once():
    steps = [
        "Generating title and story with LLM",
        "Generating audio narration", 
        "Creating video with audio and applying random speed",
        "Generating SRT subtitles"
    ]
    
    with tqdm(total=len(steps), file=sys.stdout, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as pbar:
        try:
            # Phase 1: Generate text content and immediately cleanup LMStudio
            pbar.set_postfix_str(steps[0])
            title, story, output_folder = generate_text_content()
            
            if not title or not story or not output_folder:
                print("Failed to generate text content. Aborting this run.")
                return
            
            pbar.update(1)

            # Phase 2: Audio/Video processing (no LMStudio needed)
            pbar.set_postfix_str(steps[1])
            generate_audio_from_story(story, output_folder)
            pbar.update(1)

            pbar.set_postfix_str(steps[2])
            create_video_with_audio(output_folder)
            pbar.update(1)

            pbar.set_postfix_str(steps[3])
            generate_srt_from_video(output_folder)
            pbar.update(1)
            
            print(f"Run completed successfully! Output saved to: {output_folder}")

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        num_runs = int(input("How many times should the program run? (Enter a number): "))
    except Exception:
        num_runs = 1
    for i in range(num_runs):
        print(f"\n--- Run {i+1} of {num_runs} ---\n")
        run_once()