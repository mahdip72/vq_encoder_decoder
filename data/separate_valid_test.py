import os
import glob
import shutil
from sklearn.model_selection import train_test_split


def split_and_copy_h5_files(source_directory, validation_directory, test_directory, validation_size):
    """
    Splits h5 files found in the source directory and its subdirectories into validation and test sets,
    then copies them into the specified directories.

    :param source_directory: The directory to search for h5 files.
    :param validation_directory: The directory to copy the validation set h5 files.
    :param test_directory: The directory to copy the test set h5 files.
    :param validation_size: The number of samples to be included in the validation set.
    """
    # Find all h5 files in the source directory and its subdirectories
    h5_files = glob.glob(os.path.join(source_directory, '**', '*.h5'), recursive=True)

    # Split the files into validation and test sets based on the validation_size
    validation_files, test_files = train_test_split(h5_files, train_size=validation_size,
                                                    shuffle=True, random_state=42)

    # Ensure the target directories exist
    os.makedirs(validation_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    # Copy the validation files to the validation directory
    for file in validation_files:
        shutil.copy(file, validation_directory)

    # Copy the test files to the test directory
    for file in test_files:
        shutil.copy(file, test_directory)

    print(f"Copied {len(validation_files)} files to the validation directory.")
    print(f"Copied {len(test_files)} files to the test directory.")


if __name__ == '__main__':
    source_directory = '/path/to/your/source/directory'
    validation_directory = '/path/to/your/validation/directory'
    test_directory = '/path/to/your/test/directory'
    validation_size = 1000  # This could be an integer for exact number or a float for a proportion

    split_and_copy_h5_files(source_directory, validation_directory, test_directory, validation_size)
