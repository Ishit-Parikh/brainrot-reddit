"""
video_utils.py
Video utility functions for handling video files and selections.
"""
import os
import random


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