import os
import glob
import scipy.io as sio
import pandas as pd

def load_segment(filepath):
    """
    Load a single .mat file and extract its segment structure fields.
    
    Assumes that the .mat file contains one main MATLAB structure with the following fields:
      - data: EEG data matrix (electrode x time)
      - data_length_sec: duration (in seconds) of each data row
      - sampling_frequency: number of samples per second
      - channels: list of electrode names (corresponding to rows in data)
      - sequence: the segment's sequence number in the series
    """
    # Load the .mat file with squeeze_me=True to simplify arrays
    mat_contents = sio.loadmat(filepath, squeeze_me=True)
    
    # Find the first key that doesn't start with '__'
    keys = [key for key in mat_contents.keys() if not key.startswith('__')]
    if not keys:
        raise ValueError(f"No valid data structure found in {filepath}")
    
    segment_struct = mat_contents[keys[0]]
    
    # Extract fields from the MATLAB structure
    # (this assumes the MATLAB struct fields are available as attributes)
    segment = {
        'data': segment_struct.data,
        'data_length_sec': segment_struct.data_length_sec,
        'sampling_frequency': segment_struct.sampling_frequency,
        'channels': segment_struct.channels,
        'sequence': segment_struct.sequence
    }
    return segment

def load_segments_from_folder(folder_path, segment_type):
    """
    Load all segments of a given type from a folder.
    
    Parameters:
      folder_path (str): Path to the folder (e.g., for preictal or interictal segments)
      segment_type (str): One of 'preictal', 'interictal', or 'test'
    
    Returns:
      segments (dict): Dictionary mapping file names to loaded segment data.
    """
    # Construct file pattern based on expected naming convention
    pattern = os.path.join(folder_path, f"{segment_type}_segment_*.mat")
    file_list = glob.glob(pattern)
    
    segments = {}
    for filepath in file_list:
        filename = os.path.basename(filepath)
        try:
            segments[filename] = load_segment(filepath)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return segments

if __name__ == "__main__":
    # Define the paths to your data folders
    training_folder_preictal = "path/to/training/preictal"
    training_folder_interictal = "path/to/training/interictal"
    testing_folder = "path/to/testing"

    # Load data from each folder
    preictal_segments = load_segments_from_folder(training_folder_preictal, "preictal")
    interictal_segments = load_segments_from_folder(training_folder_interictal, "interictal")
    test_segments = load_segments_from_folder(testing_folder, "test")
    
    print(f"Loaded {len(preictal_segments)} preictal segments.")
    print(f"Loaded {len(interictal_segments)} interictal segments.")
    print(f"Loaded {len(test_segments)} test segments.")
    
    # Create a summary DataFrame for the preictal segments
    summary_list = []
    for filename, seg in preictal_segments.items():
        summary_list.append({
            'filename': filename,
            'sequence': seg['sequence'],
            'data_length_sec': seg['data_length_sec'],
            'sampling_frequency': seg['sampling_frequency'],
            'num_channels': seg['data'].shape[0] if seg['data'].ndim > 1 else 1
        })
    df_preictal = pd.DataFrame(summary_list)
    print("\nPreictal segments summary:")
    print(df_preictal.head())
