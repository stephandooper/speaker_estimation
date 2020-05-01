# !/usr/bin/env python3 .
# -*- coding: utf-8 -*-.
"""
Created on Tue Apr 21 23:05:36 2020

@author: stephan
"""

import sys
import os
from pydub import AudioSegment
from pydub import effects
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from data_utils.mixer import BaseMixer
from data_utils.dataconfig import DataConfig


class LibriSpeechDataConfig(DataConfig):
    """Data configuration for the LibriSpeech dataset

    See DataConfig for more detailed information
    """

    # write speaker information from SPEAKER.txt (sex, ID) to a json files
    WRITE_META = True

    # path to the SPEAKER.TXT file, only relevant if WRITE_META is True
    SPEAKER_TXT_PATH = '../../data/LibriSpeech/SPEAKERS.TXT'

    # number of samples
    NUM_SAMPLES = 3000

    # maximum number of concurrent speakers ([1,..., MAX_NUM_SPEAKER])
    MAX_NUM_SPEAKER = 10


class LibriSpeechMixer(BaseMixer):
    """Generates synthetic mixtures for LibriSpeech

    most of the audio manipulation functionality is in BaseMixer
    """

    def __init__(self, config, audio_files=[], subset='train-clean-100'):
        """Build class constructor

        Parameters
        ----------
        config : config class instance
            The config file, has to be an instance of DataConfig
        audio_files : LIST
            A list with the audio file locations. The default is [].
        subset : STRING, optional
            Which LibriSpeech data folder to look for.
            The default is 'train-clean-100'.

        Raises
        ------
        ValueError
            if the target path in IN_PATH + subset is not found

        """
        super().__init__(config)
        self.audio_files = audio_files
        assert subset in ['dev-clean', 'dev-other',
                          'test-clean', 'test-other',
                          'train-clean-100', 'train-clean-360',
                          'train-other-500']

        self.subset = subset
        self._SPEAKER_TXT_PATH = config.SPEAKER_TXT_PATH
        self._WRITE_META_INFO = config.WRITE_META

        # The full input directory
        self._DATA_PATH = Path(self._PATH_IN) / Path(self.subset)
        if not self._DATA_PATH.exists():
            raise ValueError("Path not found")

        # append with subset to create original dir names
        self._PATH_OUT = str(Path(self._PATH_OUT) / Path(self.subset))

    @staticmethod
    def get_speaker_file(audio_files, path):
        """Get meta information from SPEAKER.txt for audio files

        SPEAKER.TXT contains relevant meta information
        about speakers (ID, sex, clip length)

        Parameters
        ----------
        audio_files : list
            A list with the paths to the flac files
        path : string
            Path to the SPEAKER.txt file

        Raises
        ------
        ValueError
            Check if audio_files is not empty

        Returns
        -------
        path_speaker_info : pandas dataframe
            a pandas dataframe with meta information
            for the speakers in audio_files
        """
        if audio_files is None:
            raise ValueError("audio files cannot be empty")

        try:
            speaker_meta = pd.read_csv(path, sep="|",
                                       error_bad_lines=False,
                                       skiprows=range(0, 11),
                                       skipinitialspace=True,
                                       usecols=[0, 1, 2, 3])

            # Strip annoying whitespaces from column headers
            speaker_meta.columns = speaker_meta.columns.str.strip()

            # Match speaker IDS
            ids = [int(Path(x).name.split('-')[0]) for x in audio_files]

            speaker_meta.rename(columns={';ID': 'ID'}, inplace=True)

            # merge speaker ID and the Path information
            file_id = pd.DataFrame(list(zip(ids, audio_files)),
                                   columns=['ID', 'PATH'])

            file_id = pd.merge(file_id, speaker_meta, on='ID')

            # Remove white spaces everywhere else
            df_obj = file_id.select_dtypes(['object'])
            file_id[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

            return file_id
        except FileNotFoundError:
            print("Error: Could not find or open the SPEAKER.TXT file")

    @classmethod
    def get_files_from_path(cls, config, subset='train'):
        """Descript

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.
        subset : TYPE, optional
            DESCRIPTION. The default is 'train'.

        Raises
        ------
        ValueError
            DESCRIPTION.
        Exception
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        assert subset in ['dev-clean', 'dev-other',
                          'test-clean', 'test-other',
                          'train-clean-100', 'train-clean-360',
                          'train-other-500']

        data_path = config.IN_PATH

        try:
            if not (Path(data_path) / Path(subset)).exists():
                raise ValueError("Path not found")

            audio_files = []

            print("crawling for files recursively in {}".format(data_path))
            for path in Path(data_path).rglob('*.flac'):
                audio_files.append(str(path))
            print("A total of {} files were found".format(len(audio_files)))

            return cls(config, audio_files, subset)

        except ValueError as e:
            raise Exception('Could not find the directory or files specified in {}'
                            .format(data_path)) from e

    def process_speakers(self, speakers):
        """Descript

        Parameters
        ----------
        speakers : TYPE
            DESCRIPTION.

        Returns
        -------
        audio : TYPE
            DESCRIPTION.
        label : TYPE
            DESCRIPTION.

        """
        # Create pydub AudioSegment instances
        audio = [AudioSegment.from_file(x, format=self._FORMAT_IN) for x in speakers]

        # remove silence at the beginning and end of the file
        audio = [super(LibriSpeechMixer, self).remove_silence(x) for x in audio]

        # repeat audio to at least 10 second duration
        audio = [super(LibriSpeechMixer, self).trim_or_extend_audio(x) for x in audio]

        # detect speech in each file
        speech_detections = [super(LibriSpeechMixer, self).detect_speech(x) for x in audio]

        # get the maximum number of concurrent speakers of the file to be mixed
        label = np.max(np.sum(speech_detections, axis=0))

        # overlay the mixtures into a single audio file
        audio = super().overlay(audio)

        # peak normalization by dividing by maximum
        audio = effects.normalize(audio)

        return audio, label

    def write_records(self):
        """Descript

        Returns
        -------
        None.

        """
        # create output folder if needed
        print("checking or creating path {}".format(self._PATH_OUT))
        Path(self._PATH_OUT).mkdir(parents=True, exist_ok=True)

        # total number of files, for generating filenames, at least 3 digits
        total_digits = np.max((len(str(self._NUM_SAMPLES)), 3))

        speaker_meta = LibriSpeechMixer.get_speaker_file(self.audio_files,
                                                         path=self._SPEAKER_TXT_PATH)

        # loop over the  desired range of simultaneously occurring speakers
        for num_speakers in tqdm(range(1, self._MAX_NUM_SPEAKER + 1), desc='Total progress'):

            # Generate a self._NUM_SAMPLES x num_speakers list of random speakers
            random_speakers = super().select_random_speakers(num_speakers, self._NUM_SAMPLES)

            # Loop over each sample of random speakers (self._NUM_SAMPLES is a sample)
            for i, speakers in enumerate(tqdm(random_speakers, desc='Current samples')):

                mixture, label = self.process_speakers(speakers)
                base_fname = str(label) + '_' + self._PREFIX + '_' + str(i).zfill(total_digits)

                # Write the mixture to the output folder
                mixture.export(Path(self._PATH_OUT) / (base_fname + '.' + self._FORMAT_OUT),
                               format=self._FORMAT_OUT)

                # Write json information
                records = speaker_meta.loc[speaker_meta['PATH'].isin(speakers)]
                records[['ID', 'SUBSET', 'SEX']].to_json(Path(self._PATH_OUT) /
                                                         (base_fname + '.json'),
                                                         orient='records')


if __name__ == '__main__':
    config = LibriSpeechDataConfig()

    # generate training dataset
    training = LibriSpeechMixer.get_files_from_path(config,
                                                    subset='train-clean-100')
    training.write_records()

    # Generate validation set
    config.NUM_SAMPLES = 300
    valid = LibriSpeechMixer.get_files_from_path(config, subset='dev-clean')
    valid.write_records()

    # Generate test set
    config.NUM_SAMPLES = 300
    test = LibriSpeechMixer.get_files_from_path(config, subset='test-clean')
    test.write_records()
