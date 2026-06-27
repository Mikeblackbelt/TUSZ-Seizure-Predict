import os
import logging 
from pathlib import Path

def verify_data_path(data_path):
    """Verify that the data path exists."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    return True


def verify_data_integrity(data_path):
    """Verify that data files are valid and accessible."""
    path = Path(data_path)
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {data_path}")
    
    files = list(path.glob("*"))
    if not files:
        raise ValueError(f"No files found in directory: {data_path}")
    
    return True

def list_files_glob(data_path):
    path = Path(data_path)
    for file in path.glob("*"):
        print(file)

def validate_input(data_path):
    """Main validation function to check path and data validity."""
    try:
        verify_data_path(data_path)
        verify_data_integrity(data_path)
        list_files_glob(data_path)
        is_valid = input('Does the input data look correct? (y/n): ')
        if is_valid.lower() != 'y':
            return False, "Data validation failed."
        return True, "Data validation successful"
    except (FileNotFoundError, ValueError) as e:
        return False, str(e)