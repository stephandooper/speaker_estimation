#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:55:13 2020

@author: stephan
"""

# TODO: write documentation and comments

import numpy as np
import webrtcvad
import random

class BaseMixer(object):

    def __init__(self, config, audio_files=[]):
        
        # empty list 
        self.audio_files = audio_files
        
        # create voice acitvity detector with agressiveness of 3
        self.vad = webrtcvad.Vad(3)
        
        # config parameters, explained in its respective file
        print(config)
        self._PATH_IN = config.IN_PATH
        self._PATH_OUT = config.OUT_PATH
        self._FORMAT_IN = config.FORMAT_IN
        self._FORMAT_OUT=config.FORMAT_OUT
        self._PREFIX = config.PREFIX
        self._NUM_SAMPLES = config.NUM_SAMPLES
        self._MAX_NUM_SPEAKER = config.MAX_NUM_SPEAKER
        self._MIXTURE_DURATION = config.CLIP_DURATION

    def get_files(self):
        raise NotImplementedError("Error: method should be defined in a subclass")

    def select_random_speakers(self, num_speakers, num_files=None):
        if num_files is None:
            num_files = self._NUM_SAMPLES
        
        mixture_cand = np.random.choice(self.audio_files, num_speakers * num_files, replace=True)
        mixture_cand = mixture_cand.reshape([num_files, num_speakers])
        return mixture_cand

    def remove_silence(self, audio):

        def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
            '''
            https://stackoverflow.com/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
            sound is a pydub.AudioSegment
            silence_threshold in dB
            chunk_size in ms

            iterate over chunks until you find the first one with sound
            '''
            trim_ms = 0 # ms

            assert chunk_size > 0 # to avoid infinite loop
            while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
                trim_ms += chunk_size

            return trim_ms

        start_trim = detect_leading_silence(audio)
        end_trim = detect_leading_silence(audio.reverse())

        return audio[start_trim:len(audio)-end_trim]

    def detect_speech(self, audio, chunk_ms=30):
        speech_detections = []
        offset = 0
        while offset + chunk_ms< len(audio):
            chunk = audio[offset: offset + chunk_ms]
            speech_detections.append(self.vad.is_speech(chunk.raw_data, sample_rate=chunk.frame_rate) * 1)
            offset += chunk_ms
            
        return speech_detections

    def trim_or_extend_audio(self,audio, target_duration=None):
        if target_duration is None:
            target_duration = self._MIXTURE_DURATION

        # Duration of the current clip in ms
        duration = len(audio)        
        if duration < target_duration:

            # the amount of times to repeat the audio file
            num_repeat = int(np.ceil(target_duration / duration))

            # repeat the audio num_repeat times
            audio = audio * num_repeat

            # new duration can be > duration, so only select target_duration
            return audio[:target_duration]

        # For audio files (much) larger than the duration, select random excerpts
        else:
            # The clip can start at most here, or else the desired duration exceeds the clip duration
            max_start_time = duration - target_duration
            start_time = random.choice(range(0, max_start_time))
            return audio[start_time:start_time + target_duration]

    # TODO: write this without recursion
    def overlay(self, audio_list):

        if len(audio_list) == 1:
            return audio_list[0]

        audio_list[0] = audio_list[0].overlay(audio_list[-1:][0])
        return self.overlay(audio_list[:-1])
