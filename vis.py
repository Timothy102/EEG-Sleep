#!/usr/bin/env python
import os
import logging
import wfdb
import glob
import numpy as np
import pyedflib
import mne

from mne.io import read_raw_edf
from fourier import fourier
from config import * 


mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

def edfplot(psg_name, ann_name):
    raw_train = mne.io.read_raw_edf(psg_name)
    annot_train = mne.read_annotations(ann_name)

    raw_train.set_annotations(annot_train, emit_warning=False)
    raw_train.set_channel_types(mapping)

    # plot some data
    raw_train.plot(duration=60, scalings='auto')

    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                'Sleep stage 1': 2,
                                'Sleep stage 2': 3,
                                'Sleep stage 3': 4,
                                'Sleep stage 4': 4,
                                'Sleep stage R': 5}

    # keep last 30-min wake events before sleep and first 30-min wake events after
    # sleep and redefine annotations on raw data
    annot_train.crop(annot_train[1]['onset'] - 30 * 60,
                    annot_train[-2]['onset'] + 30 * 60)
    raw_train.set_annotations(annot_train, emit_warning=False)

    events_train, _ = mne.events_from_annotations(
        raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    # create a new event_id that unifies stages 3 and 4
    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3/4': 4,
                'Sleep stage R': 5}

    # plot events
    fig = mne.viz.plot_events(events_train, event_id=event_id,
                            sfreq=raw_train.info['sfreq'],
                            first_samp=events_train[0, 0])

    # keep the color-code for further plotting
    stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def ftt(data):
    ### BEFORE FOURIER
    plt.plot(data)
    plt.title('Before Fourier Transformation')
    plt.ylabel('sleep')
    plt.xlabel('time')
    plt.show()

    ### AFTER FOURIER
    plt.plot(fourier(data))
    plt.title('After Fourier Transformation')
    plt.ylabel('sleep')
    plt.xlabel('time')
    plt.legend(['before', 'after'], loc='upper right')
    plt.show()


def main(args = sys.argv[1:]):
if name == 'main':
    main()

