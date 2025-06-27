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


def get_custom_titles(num_runs):
    """Get custom titles from user if they want to provide them."""
    custom_titles = []
    
    use_custom = input("Do you have specific titles in mind for the stories? (y/n): ").lower().strip()
    
    if use_custom in ['y', 'yes']:
        print(f"\nPlease enter {num_runs} title(s):")
        for i in range(num_runs):
            while True:
                title = input(f"Title {i+1}: ").strip()
                if title:
                    custom_titles.append(title)
                    break
                else:
                    print("Please enter a valid title (cannot be empty)")
        
        print(f"\n✓ Got {len(custom_titles)} custom titles!")
        for i, title in enumerate(custom_titles, 1):
            print(f"  {i}. {title}")
    
    return custom_titles


def run_once(custom_title=None, run_number=1):
    """Execute one complete run of the content generation pipeline."""
    steps = [
        "Generating title and story with LLM" if not custom_title else "Generating story with custom title",
        "Generating audio narration", 
        "Creating video with audio and applying random speed",
        "Generating SRT subtitles"
    ]
    
    with tqdm(total=len(steps), file=sys.stdout, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as pbar:
        try:
            # Phase 1: Generate text content
            pbar.set_postfix_str(steps[0])
            if custom_title:
                print(f"Using custom title: {custom_title}")
            
            title, story, output_folder = generate_text_content(custom_title)
            
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
    num_runs = int(input("How many stories would you like to create? (Enter a number): "))
    
    # Get custom titles if user wants to provide them
    custom_titles = get_custom_titles(num_runs)
    
    successful_runs = 0

    print("\n✓ Starting LMStudio server...")
    os.system("lms server start")
    
    for i in range(num_runs):
        print(f"\n{'='*50}")
        print(f"Run {i+1} of {num_runs}")
        print(f"{'='*50}")
        
        # Use custom title if available, otherwise None for auto-generation
        current_title = custom_titles[i] if custom_titles else None
        
        if run_once(current_title, i+1):
            successful_runs += 1
        
        if i < num_runs - 1:  # Not the last run
            print("Completed run {}. Preparing for next run...".format(i+1))
            time.sleep(3)  # Brief pause between runs
    
    print("\n✓ Stopping LMStudio server...")
    os.system("lms server stop")

    print(f"\n{'='*50}")
    print("All runs completed!")
    print(f"Successful runs: {successful_runs}/{num_runs}")
    print(f"{'='*50}")