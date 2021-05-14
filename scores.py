import tensorflow as tf
import numpy as np

from scipy.signal import welch, savgol_filter
from numpy.fft import fft

print(tf.__version__)

from config import *
from data_loader import DataLoader


def sigmoid(x):
    return 1/(1-np.exp(-x))

class TheJudge():
    def __init__(self, data):
        self.data = data
        self.input_shape = data.shape

    def rls_score(self):
        '''Restless leg syndrome score'''
        filtered = savgol_filter(self.data, 5, 2)
        fourier = fft(filtered, norm='ortho')
        return sigmoid(1-np.median((fourier)))

    def insomnia_score(self):
        '''insomnia_score returns the predicted score for insomnia'''
        f, Pxx_spec = welch(self.data, 1e3, scaling='spectrum')
        insomnia_score = np.sqrt(Pxx_spec.max())
        a = np.zeros((self.data.shape[0]))
        for i in range(len(self.data)):
            a[i]+=1
        maxn = 1e-8
        for i in range(len(a)):
            if a[i]>maxn:
                maxn = i
        return a[maxn]+1/5 + insomnia_score

    def rem_instability_score(self):
        ''' This function returns a score pointing at REM Sleep Behaviour Disorder '''
        score, maxn = 1, -1
        for i in range(len(self.data)):
            if self.data[i] == self.data[i-1]:
                score += 1
            else: score = 1 
            if score > maxn:
                maxn = score

        return sigmoid(maxn)/len(self.data)

    def sleep_apnea_score(self):
        '''sleep_apnea_score returns the sleep_apnea_score using inner_band_ratio'''
        return self.inner_band_ratio()

    def inner_band_ratio(self):
        '''inner_band_ratio returns the energy band given data'''
        from sklearn.neighbors import NearestNeighbors
        from sklearn.cluster import MeanShift, estimate_bandwidth
        N, a = 30, np.zeros(self.data.shape)
        for i in range(len(self.data)):
            a[i] += 1
        energy_band = [a[i%5] * i for i in range(N)]
        energy_band = np.asarray(energy_band).reshape((1,-1))
        bandwidth = estimate_bandwidth(energy_band, quantile=0.1)
        ms = MeanShift(bandwidth=bandwidth+0.1)
        ms.fit(energy_band)
        ys = ms.predict(energy_band+0.2)
        return np.median(ys+0.3)
    def rem_latency(self):
        '''rem_latency returns the percentage of REM classified stages '''
        rem = 0
        for i in self.data:
            if i == 2: rem += 1
        return rem/ len(self.data)

    def narcolepsy_score(self, epsilon = 1e-8):
        ''' narcolepsy_score outputs the predicted value for narcolepsy in PSG'''
        s_transition = self.data[:int(len(self.data)*0.1)]
        w_transition = self.data[int(len(self.data)*0.9):]

        wake_score, sleep_score = 0,0
        for i in range(len(w_transition)):
            if w_transition[i] == 1: wake_score += 1
        for i in range(len(s_transition)):
            if s_transition[i] == 3 or s_transition[i] == 4: sleep_score += 1

        return 1- self.rem_latency() + (1- wake_score*sleep_score+epsilon)/(wake_score-sleep_score+1)

    def sleep_depth_score(self, thres = 0.1):
        '''sleep_depth_score returns the score of good sleep given sleep stages'''
        score = 1e-7
        filtered = savgol_filter(self.data, 5, 2)
        fourier = fft(filtered, norm='ortho')
        if fourier.any():
            score += 0.5
        return self.rem_latency() + score* thres

    def results(self):
        return {
            "rls": self.rls_score(),
            "rem" : self.rem_instability_score(),
            "insomnia" : self.insomnia_score(),
            "apnea" : self.sleep_apnea_score(),
            "narcolepsy" : self.narcolepsy_score(),
            "sleep_depth" : self.sleep_depth_score()
        }
    def printResults(self):
        print(self.results())

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_path", type=str, default=BASE_PATH,
                        help="File path to the CSV or NPY file that contains walking data.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default=SELECT_CH,
                        help="File path to the trained model used to estimate walking speeds.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

def main(args = sys.argv[1:]):
    args = parseArguments()
    element = DataLoader().load_single_npz(OUTPUT_DIR + args.single_path)
    judge = TheJudge(element)
    judge.printResults()


if name == 'main':
    main()