from util import handle_logs, verify_data

def read_annotated_data(file_path):
    """
    Reads annotated .edf data from the TUSZ corpus and returns a structured representation of the data.
    Each .edf file in a folder in the TUSZ accompined by the following:  
       (a) a .csv file containing the seizure start and end times, the seizure type, and the channel(s) recording the seizure
       (b) a .csvbi file containing seizure start and end times, though not the seizure types"""