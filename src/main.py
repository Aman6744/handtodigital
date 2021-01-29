import argparse
import os

from Create_hdf5 import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transform", action="store_true", default=False)

    args = parser.parse_args()

    raw_path = os.path.join("..", "data", "raw", "IAM")
    source_path = os.path.join("..", "data", "dataset_hdf5", "iam_words.hdf5")
    
    target_image_shape = (96, 32, 1)
    batch_size = 32
    maxTextLength = 24

    if args.transform:
        if os.path.isfile(source_path): 
            print("Dataset file already exists")
        else:
            print("Transforming the IAM dataset..")
            ds = Dataset(raw_path)
            ds.save_partitions(source_path, target_image_shape, maxTextLength)

