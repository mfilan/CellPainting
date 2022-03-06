import glob
import os
import re
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tifffile import imread, imsave

from src.config.files import (
    data_raw as raw_path,
    data_interim as interim_path,
    data_processed as processed_path,
    data_annotations as df_path
)


class DataHandler:
    def __init__(self):
        if not df_path.is_file():
            self.create_csv(df_path)
        self.dataframe = pd.read_csv(df_path, dtype={'folder': 'category'})

    def get_base_df(self):
        paths = [i.split(os.sep)[-2:] for i in glob.iglob(str(raw_path) + '/**/*.tiff', recursive=True)]
        df = pd.DataFrame(paths, columns=['folder', 'filename'])
        # r01c01f01p01-ch1sk1fk1fl1.tiff
        # r01   f01   -ch1   fk1   .tiff
        #    c01   p01-   sk1   fl1.tiff
        # 012345678901234567890123456789
        # A different order of these lines breaks everything lol (enjoy saving 30 min)
        df['well_id'] = df['folder'] + df['filename'].apply(lambda x: x[0:6])
        df['compound'] = df['filename'].apply(lambda x: int(x[1:3]))
        df['concentration'] = df['filename'].apply(lambda x: int(x[4:6]))
        df['Channel'] = df['filename'].apply(lambda x: x[15:16])
        df['ids'] = df['folder'] + df['filename'].astype(str) \
            .apply(lambda x: "".join([x[1:3], x[4:6], x[7:9], x[10:12], x[18:19], x[21:22], x[24:25]]))
        df = df.pivot(index='ids', columns='Channel').T.drop_duplicates().T
        df = pd.DataFrame(df[:].values,
                          columns=['folder', 'filename1', 'filename2', 'filename3', 'filename4',
                                   'well_id', 'compound', 'concentration'])
        return df

    @staticmethod
    def map_values(df):
        compound_mapping = {1: 'Berberine Chloride', 2: 'Brefeldin A', 3: 'Fluphenazine', 4: 'Latrunculin B',
                            5: 'Nocodazole', 6: 'Rapamycin', 7: 'Rotenone', 8: 'Tetrandrine', 9: 'Berberine Chloride',
                            10: 'Brefeldin A', 11: 'Fluphenazine', 12: 'Latrunculin B', 13: 'Nocodazole',
                            14: 'Rapamycin', 15: 'Rotenone', 16: 'Tetrandrine'}
        concentration_mapping = {1: 50, 2: 25, 3: 12.5, 4: 6.25, 5: 3.166666667, 6: 1.583333333, 7: 0.75,
                                 8: 0.416666667, 9: 0.166666667, 10: 0.083333333, 11: 0.041666667, 12: 50, 13: 25,
                                 14: 12.5, 15: 6.25, 16: 3.166666667, 17: 1.583333333, 18: 0.75, 19: 0.416666667,
                                 20: 0.166666667, 21: 0.083333333, 22: 0.041666667, 23: 0, 24: 0}
        df.loc[:, 'concentration_name'] = df['concentration'].map(concentration_mapping)
        df.loc[:, 'compound_name'] = df['compound'].map(compound_mapping)
        return df

    def create_csv(self, path):
        df = self.get_base_df()
        df = self.map_values(df)
        df['folder'] = df['folder'].astype('category')
        # df['dolek'] = df/caskjhfa.apply(lambda x: )
        new_filenames = df['filename1'].str[:-5] + "_" + df.folder.cat.codes.astype(str) + ".tiff"
        df.insert(0, "filename", new_filenames)
        df.filename = df.filename.apply(lambda x: re.sub(r"ch\d", "", x))
        df.loc[df['concentration_name'] == 0, 'compound_name'] = "DFSO"
        df['compound_label'] = LabelEncoder().fit_transform(df['compound_name'])
        df.to_csv(path, index=False)

    def get_raw_paths(self):
        filenames = self.dataframe.iloc[:, 1:6].melt(id_vars="folder")[['folder', 'value']]
        filenames['new_filenames'] = filenames['value'].str[:-5] + "_" + filenames.folder.cat.codes.astype(
            str) + ".tiff"
        paths = filenames[['folder', 'value']].agg(f'{os.sep}'.join, axis=1).to_numpy()
        new_filenames = filenames['new_filenames'].to_numpy()
        return np.stack((paths, new_filenames), axis=1)

    def resize(self, paths):
        img = imread(raw_path / paths[0])
        resized_img = cv2.resize(img, (224, 224), cv2.INTER_LANCZOS4)
        imsave(interim_path / paths[1], resized_img)

    def create_interim_data(self):
        paths = self.get_raw_paths()
        pool = ThreadPool(cpu_count())
        pool.map(self.resize, paths)
        return None

    def get_interim_paths(self):
        df = self.dataframe.iloc[:, 1:6]
        df = df.iloc[:, 1:5].apply(
            lambda x: str(interim_path) + os.sep + x.str[:-5] + "_" + df.folder.cat.codes.astype(str) + ".tiff")
        return df.to_numpy()

    def merge_images(self, filenames):
        new_filename = re.sub(r"ch\d", "", filenames[0].split(os.sep)[-1])
        image = np.stack(
            (imread(filenames[0]), imread(filenames[1]), imread(filenames[2]), imread(filenames[3]))).transpose(
            (1, 2, 0))
        imsave(processed_path / new_filename, image)
        return None

    def create_processed_data(self):
        filenames = self.get_interim_paths()
        pool = ThreadPool(cpu_count())
        pool.map(self.merge_images, filenames)