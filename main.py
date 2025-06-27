"""
main.py
Main execution script for the FRBV content generation pipeline.
"""
import time
import os
import sys
from tqdm import tqdm
import traceback

from text_generator import generate_text_content
from audio_generator import generate_audio_from_story
from video_creator import create_video_with_audio


def run_once():
    """Execute one complete run of the content generation pipeline."""
    steps = [
        "Generating title and story with LLM",
        "Generating audio narration", 
        "Creating video with audio and applying random speed",
        "Generating SRT subtitles"
    ]
    
    with tqdm(total=len(steps), file=sys.stdout, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as pbar:
        try:
            # Phase 1: Generate text content
            pbar.set_postfix_str(steps[0])
            title, story, output_folder = generate_text_content()
            
            if not title or not story or not output_folder:
                print("✗ Failed to generate text content. Aborting this run.")
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
            
            print(f"✓ Run completed successfully! Output saved to: {output_folder}")
            return True

        except Exception as e:
            print(f"✗ An error occurred during processing: {e}")
            traceback.print_exc()
            return False


if __name__ == "__main__":

    num_runs = int(input("How many times should the program run? (Enter a number): "))
    
    successful_runs = 0

    print("✓ Starting LMStudio server...")
    os.system("lms server start")
    
    for i in range(num_runs):
        print(f"\n{'='*50}")
        print(f"Run {i+1} of {num_runs}")
        print(f"{'='*50}")
        
        if run_once():
            successful_runs += 1
        
        if i < num_runs - 1:  # Not the last run
            print("Completed run {}. Preparing for next run...".format(i+1))
            time.sleep(3)  # Brief pause between runs
    
    print("✓ Stopping LMStudio server...")
    os.system("lms server stop")

    print(f"\n{'='*50}")
    print("All runs completed!")
    print(f"Successful runs: {successful_runs}/{num_runs}")
    print(f"{'='*50}")