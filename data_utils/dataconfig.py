#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 00:09:31 2020

@author: stephan
"""

class DataConfig(object):
    
    # Path to the dataset dir (must have training, dev, and test directories)
    IN_PATH = '../../data/LibriSpeech'
    
    # Path where the mixtures will be saved
    OUT_PATH = '../../data/LibriSpeech/mixtures'
    
    # The format the original files are in (.wav, .mp3, .flac, ...)
    FORMAT_IN = 'flac'
    
    # The desired output format
    FORMAT_OUT = 'wav'
    
    # A filename prefix for each individual synthetic mixture
    PREFIX = 'mixture'
    
    # Note: the total number of files generated = NUM_SAMPLES * NUM_SPEAKERS
    # The number of samples per speaker mixtures
    NUM_SAMPLES = 3
    
    # The maximum number of concurrent speakers
    # The program will generated [1, ..., MAX_NUM_SPEAKER] concurrent speakers
    MAX_NUM_SPEAKER = 3
    
    # duration of the mixture, in ms
    CLIP_DURATION = 10000
    
    