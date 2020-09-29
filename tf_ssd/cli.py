import argparse, os
from tf_ssd import tfrecord_creator
from tf_ssd.helpers import download_chess_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="the directory containing images", type=str, required=True)
    parser.add_argument("--xml_dir", help="the directory containing annotations in pascal-voc xml format", type=str, required=True)

    args = parser.parse_args()

    tfrecord_creator.create_tfrecords(
        image_dir=args.image_dir, 
        xml_dir=args.xml_dir)

def dowmload_chess_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", help="the directory where dataset is to be extracted", type=str, default=os.getcwd())

    args = parser.parse_args()

    download_chess_dataset(os.getcwd())

