# -*- coding: utf-8 -*-
import logging

from src.data.DataHandler import DataHandler


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    handler = DataHandler()
    logger.info('resizing images and saving as interim data...')
    handler.create_interim_data()
    logger.info('merging channels into one image...')
    handler.create_processed_data()
    logger.info('data processing completed.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()