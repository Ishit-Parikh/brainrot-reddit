"""
text_generator.py
Handles text content generation using LMStudio.
"""
import os
import time
import lmstudio as lms
import traceback

from utils import (
    read_file, 
    create_output_folder, 
    write_text_file
)


def generate_text_content(custom_title=None):
    """
    Generate title and story content using LMStudio. 
    
    Args:
        custom_title (str, optional): Custom title to use instead of generating one
        
    Returns:
        tuple: (title, story, output_folder)
    """
    # Give the server some time to start
    time.sleep(5)

    model_name = "google/gemma-3-4b"
    try:
        # Simple approach: Just start the server (like in llms.py)
        print("✓ Starting LMStudio server...")
        
        print("✓ Loading model...")
        model = lms.llm(model_name)

        # File paths
        title_prompt_path = "System_Title_Prompt.txt"
        story_prompt_path = "System_Story_Prompt.txt"
        
        print("✓ Reading system prompts...")
        system_prompt_title = read_file(title_prompt_path)
        system_prompt_story = read_file(story_prompt_path)

        # Handle title generation or use custom title
        if custom_title:
            print(f"✓ Using custom title: {custom_title}")
            response_title = custom_title
        else:
            title_prompt = "Generate a creative title for a story based on either a Doctor or a Navy Officer" 
            
            print("✓ Generating title...")
            
            # Use the simpler approach from llms.py
            full_title_prompt = f"{system_prompt_title}\n\nUser: {title_prompt}"
            response_title = model.respond(full_title_prompt)
            response_title = response_title.content
        
        # Create output folder for this run
        output_base = "/media/lord/Local Disk"
        output_folder = create_output_folder(output_base, response_title)
        write_text_file(output_folder, "title.txt", response_title)
        print("✓ Title saved successfully")

        # Prepare story generation prompt
        story_user_content = (
            "Always keep the shared title as the first line of the story & do not make several paragraphs, "
            "give the entire story in as specified in the system prompt\n" + response_title 
        )
        
        print("✓ Generating story...")
        full_story_prompt = f"{system_prompt_story}\n\nUser: {story_user_content}"
        response_story = model.respond(full_story_prompt)
        response_story = response_story.content
        
        write_text_file(output_folder, "storie.txt", response_story)
        print("✓ Story saved successfully")
        
        model.unload()  

        return response_title, response_story, output_folder
          
    except Exception as e:
        print(f"Error {e}")
        return None, None, None