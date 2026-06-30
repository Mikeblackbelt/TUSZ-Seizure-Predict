import os
import logging 
from pathlib import Path

def verify_data_path(data_path):
    """
    Verify that the data path exists.
    
    Returns:
    	bool: `True` if the path exists.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    return True


def verify_data_integrity(data_path):
    """
    Verify that a data directory exists and contains at least one entry.
    
    Parameters:
    	data_path: Path to the directory to check.
    
    Returns:
    	bool: `True` if the path is a directory and contains at least one file or subdirectory.
    """
    path = Path(data_path)
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {data_path}")
    
    files = list(path.glob("*"))
    if not files:
        raise ValueError(f"No files found in directory: {data_path}")
    
    return True

def list_files_glob(data_path):
    """
    Print all entries in the given directory.
    
    Parameters:
    	data_path: Directory path to scan.
    """
    path = Path(data_path)
    for file in path.glob("*"):
        print(file)

def validate_input(data_path):
    """
    Validate the input data path and confirm the contents interactively.
    
    Returns:
    	result (tuple[bool, str]): A success flag and message. Returns `(True, "Data validation successful")` when the path exists, is a populated directory, and the user confirms the data; otherwise returns `(False, message)` with an error or failure message.
    """
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