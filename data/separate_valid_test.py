import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def copy_file(file_path, target_directory):
    """
    Copies a single file to the target directory.
    """
    shutil.copy(file_path, target_directory)

def split_and_copy_h5_files(source_directory, validation_directory, test_directory, validation_size):
    """
    Splits h5 files found in the source directory and its subdirectories into validation and test sets,
    then copies them into the specified directories using multiprocessing for faster execution.
    """
    # Find all h5 files in the source directory and its subdirectories
    h5_files = glob.glob(os.path.join(source_directory, '**', '*.h5'), recursive=True)

    # Split the files into validation and test sets based on the validation_size
    validation_files, test_files = train_test_split(h5_files, train_size=validation_size,
                                                    shuffle=True, random_state=42)

    # Ensure the target directories exist
    os.makedirs(validation_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    # Use ProcessPoolExecutor to copy files in parallel
    with ProcessPoolExecutor(max_workers=32) as executor:
        # Copy validation files
        list(tqdm(executor.map(copy_file, validation_files, [validation_directory] * len(validation_files)),
                  total=len(validation_files), desc="Copying validation files"))
        # Copy test files
        list(tqdm(executor.map(copy_file, test_files, [test_directory] * len(test_files)),
                  total=len(test_files), desc="Copying test files"))

    print(f"Copied {len(validation_files)} files to the validation directory.")
    print(f"Copied {len(test_files)} files to the test directory.")


if __name__ == '__main__':
    source_directory = '/mnt/hdd8/mehdi/datasets/vqvae/test_case_a_1024_h5/'
    validation_directory = '/mnt/hdd8/mehdi/datasets/vqvae/validation_set_1024/'
    test_directory = '/mnt/hdd8/mehdi/datasets/vqvae/test_set_a_1024/'
    validation_size = 2000  # This could be an integer for exact number or a float for a proportion

    split_and_copy_h5_files(source_directory, validation_directory, test_directory, validation_size)