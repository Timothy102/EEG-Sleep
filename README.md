# DeepSleepNet: Sleep syndromes onset detection based on automatic sleep staging algorithm

# Abstract
Sleep specialists often conduct manual sleep stage scoring by visually inspecting the patient’s neurophysiological signals collected at sleep labs. This is, generally, a difﬁcult, tedious and time-consuming task. The limitations of manual sleep stage scoring have escalated the demand for developing Automatic Sleep Stage Classiﬁcation (ASSC) systems. Sleep stage classiﬁcation refers to identifying the various stages of sleep and is a critical step in an effort to assist physicians in the diagnosis and treatment of related sleep disorders.
In this paper, we propose a novel method and a practical approach to predicting early onsets of sleep syndroms, including restless leg syndrome, insomnia, based on an algorithm which is comprised of two modules. A Fast Fourier Transform is applied to 30 seconds long epochs of EEG recordings to provide localized time-frequency information, and a deep convolutional LSTM neural network is trained for sleep stage classification. Automating sleep stages detection from EEG data offers a great potential to tackling sleep irregularities on a daily basis. Thereby, a novel approach for sleep stage classification is proposed which combines the best of signal processing and statistics. In this study, we used the PhysioNet Sleep European Data Format (EDF) Database. The code evaluation showed impressive results, reaching accuracy of 90.43, precision of 77.76, recall of 93,32, F1-score of 89.12 with the final mean false error loss 0.09.

You can read the whole paper at : https://www.academia.edu/s/be1e1d7846

## The Model Architecture

<img src="https://github.com/Timothy102/EEG-sleep/blob/main/images/model.png" alt="drawing" width="750"/>

# Requirements

* pandas==1.1.4
* numpy==1.19.5
* mne==0.22.0
* matplotlib==3.3.3
* glob2==0.7
* datetime==4.3

`
pip3 install requirements.txt
`


## Data

We utilized the PhysioNet dataset called SleepEDF. The sleep-edf database contains 197 whole-night PolySomnoGraphic sleep recordings, containing EEG, EOG, chin EMG, and event markers. Some records also contain respiration and body temperature. Corresponding hypnograms (sleep patterns) were manually scored by well-trained technicians according to the Rechtschaffen and Kales manual, and are also available

```
cd data_2018
chmod +x download_physionet.sh
./download_physionet.sh
```

To extract the 30-second epochs, use the code below! :)
```
python prepare_physionet.py --data_dir data_2013 --output_dir data_2013/eeg_pz_oz --select_ch 'EEG Pz-Oz'
```

<img src="https://github.com/Timothy102/EEG-sleep/blob/main/images/alice.png" alt="drawing" width="750"/>



## References
 [github:akaraspt](https://github.com/akaraspt/deepsleepnet)  
 [deepschool.io](https://github.com/sachinruk/deepschool.io/blob/master/DL-Keras_Tensorflow)
 ```
@article{mousavi2019sleepEEGnet,
  title={SleepEEGNet: Automated Sleep Stage Scoring with Sequence to Sequence Deep Learning Approach},
  author={Sajad Mousavi, Fatemeh Afghah and U. Rajendra Acharya},
  journal={arXiv preprint arXiv:1903.02108},
  year={2019}
}
```
