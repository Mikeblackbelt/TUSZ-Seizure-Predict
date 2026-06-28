import argparse
import subprocess
import sys
from util import handle_logs, verify_data
from testing import read_annotated_data, test_

parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=str, help="Path to the input dataset.")
parser.add_argument("--log_path", type=str, default="logs\\app.log", help="Path to the log file. Default is logs\\app.log")
args = parser.parse_args()

DATASET_PATH = args.input_path
LOG_PATH = args.log_path

LOGGER = handle_logs.get_logger(args.log_path)

def main(): 
    result = subprocess.run([
        sys.executable, "-m", "pytest", "test_myfile.py",
        "-v",        # test names
        "--tb=short", # traceback on failure
        "--no-header",
    ])
    if result.returncode != 0:
        LOGGER.error("Unit Tests Failed. Exiting.")
        return

if __name__ == "__main__":
    main()
  
    
