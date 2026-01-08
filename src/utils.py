"""
Utility functions for self-driving car project
Basic file operations and data handling
"""

import os
import csv

def setup_project_folders():
    """Ensure all required folders exist"""
    folders = [
        'data/training/IMG',
        'data/sample_data/IMG', 
        'models',
        'notebooks',
        'report'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created/verified: {folder}")

def get_file_count(folder_path):
    """Count files in a folder"""
    if os.path.exists(folder_path):
        return len([f for f in os.listdir(folder_path) 
                   if os.path.isfile(os.path.join(folder_path, f))])
    return 0

if __name__ == "__main__":
    setup_project_folders()
    print("Project folders set up successfully")
