# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:11:46 2021

@author: Boaz_Laptop
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from lempel_ziv_complexity import lempel_ziv_complexity
from tqdm import tqdm
from datetime import datetime


class flutterFeatureExtractor:
    def __init__(
        self,
        sensorName,
        windowSize,
        calibrationFactor,
        calibrationFlag,
        numArFeatures=2,
    ):
        """
        This is the regularity feature extractor class.
        implementation instructions:

        1. initialize class
        2. iterate over signal measurements
        3. use get_measurement to feed current measurement to class

        once the class has enough data (>= window size) regularity feature calculation will commence


        :param sensorName: string, name of the sensor
        :param windowSize: int, size of the sliding window
        :param calibrationFactor: float, data augmentation factor
        :param calibrationFlag: bool, True - calculate calibration factor for dataset
        :param numArFeatures: int, optional, autoregressive model order
        """

        # unpack parameters:
        # ------------------
        self.sensorName = sensorName
        self.windowSize = windowSize
        self.numArFeatures = numArFeatures
        self.calibrationFactor = calibrationFactor
        self.calibrationFlag = calibrationFlag

        # initialize:
        # ----------
        self.onlineSignal = []
        self.velocity = []
        self.signalLength = 0
        self.arCod = []
        self.spectralEntropy = []
        self.lzc = []
        self.uval = []
        self.arDataset = []
        self.arLabels = []
        self.flutterWarning = []

        # Default Thresholds
        # ------------------
        self.threshold_arcod = 0.48
        self.threshold_spectralEntropy = 0.6
        self.threshold_lzc = 1.15
        self.threshold_uval = 14

    def get_measurement(self, measurement):
        """
        :param measurement: current measurement value from sensor data
        :return: none
        """
        # add sensor measurement
        self.onlineSignal.append(
            measurement + np.random.normal() * self.calibrationFactor
        )
        self.signalLength += 1
        # calculate regularity features
        self.calcFeatures()

    def signal_to_dataset(self, signal):
        """
        Create autoregressive feature matrix and target vector for linear regression
        :param signal: list, sensor data for current window
        :return: none
        """
        i = 0
        dataset = []
        labels = []
        while i < (len(signal) - self.numArFeatures - 1):
            dataset.append(signal[i : (i + self.numArFeatures)])
            labels.append(signal[i + self.numArFeatures])
            i += 1
        self.arDataset = np.array(dataset)
        self.arLabels = np.array(labels)

    def calcSpectralEntropy(self, signal):
        """
        Calculate Spectral Entropy for current window
        :param signal: list, sensor data for current window
        :return:
        """
        fft_signal = np.abs(np.fft.fft(signal)) ** 2
        fft_signal = fft_signal / sum(fft_signal)
        if self.calibrationFlag:
            return -np.sum(fft_signal * np.log2(fft_signal) // np.log2(self.windowSize))
        else:
            self.spectralEntropy.append(
                -np.sum(fft_signal * np.log2(fft_signal)) / np.log2(self.windowSize)
            )

    def calcArCod(self, signal):
        """
        Calculate arCod feature for current window
        :param signal: list, sensor data for current window
        :return: none
        """
        self.signal_to_dataset(signal)
        clf = LinearRegression()
        clf.fit(self.arDataset, self.arLabels)
        y_pred = clf.predict(self.arDataset)
        if self.calibrationFlag:
            return r2_score(self.arLabels, y_pred)
        else:
            self.arCod.append(r2_score(self.arLabels, y_pred))

    def calcUval(self, signal):
        """
        Calculate U-val feature for current window
        :param signal: list, sensor data for current window
        :return: none
        """
        fft_signal = np.abs(np.fft.fft(signal))
        fft_signal = fft_signal / max(fft_signal)
        fft_signal_sum = np.sum(fft_signal)
        if self.calibrationFlag:
            return fft_signal_sum
        else:
            self.uval.append(fft_signal_sum)

    def calcLZC(self, signal):
        """
        Calculate Lempel-Ziv Complexity for current window
        :param signal: list, sensor data for current window
        :return: none
        """
        binary_signal = signal > np.median(signal)
        binary_signal = binary_signal.astype("int")
        binary_signal = binary_signal.astype("str")
        string_signal = ""
        for j in range(len(signal)):
            string_signal += binary_signal[j]
        N = len(string_signal)
        b = N / np.log(N)

        if self.calibrationFlag:
            return lempel_ziv_complexity(string_signal) / b
        else:
            self.lzc.append(lempel_ziv_complexity(string_signal) / b)

    def calibration(self, start, stop):
        """
        Find adequate augmentation factor for current dataset
        :param start: int, first index of nominal-flight signal measurements
        :param stop:  int, last index of nominal-flight signal measurements
        :return: none
        """

        if self.calibrationFlag:

            self.nominalSignal = np.array(self.onlineSignal[start:stop])
            self.calibrationRandomSignal = np.random.normal(
                size=len(self.nominalSignal)
            )
            self.targetValue = 0.16

            def median_nominal_arCod(factor):
                """
                Calculate median arCod value for signal + whiteNoise*factor
                :param factor: float, augmentation factor
                :return: medianArcod: float, median arCod for signal + whiteNoise*factor
                """
                factor = max([factor, 0])
                randSignal = self.calibrationRandomSignal * factor
                self.augmentedSignal = self.nominalSignal + randSignal
                statArcod = np.zeros([len(self.nominalSignal) - self.windowSize, 1])
                for j in range(len(self.augmentedSignal) - self.windowSize):
                    statArcod[j] = self.calcArCod(
                        self.augmentedSignal[j : (j + self.windowSize)]
                    )
                medianArcod = np.median(statArcod)
                return medianArcod

            tic = datetime.now()

            notOptimized = True
            optimalFactor = 0
            print("\nSearching for optimal factor:..\n")

            # Exhaustive search of augmentation factor:
            # ----------------------------------------
            while notOptimized:
                optimalFactor += np.std(self.nominalSignal) * 0.01
                tmp = median_nominal_arCod(optimalFactor)
                notOptimized = tmp > self.targetValue
                print("current median arCod: " + str(np.round(tmp, 3)) + "\n")

            toc = datetime.now()

            print("optimization duration: " + str(toc - tic) + "\n")
            print("Augmentation factor = " + str(optimalFactor))

            self.calibrationFactor = optimalFactor

            self.calibrationFlag = False

    def calcFeatures(self):
        if self.signalLength > self.windowSize:
            self.calcSpectralEntropy(
                self.onlineSignal[(self.signalLength - self.windowSize) : -1]
            )
            self.calcArCod(
                self.onlineSignal[(self.signalLength - self.windowSize) : -1]
            )
            self.calcUval(self.onlineSignal[(self.signalLength - self.windowSize) : -1])
            self.calcLZC(self.onlineSignal[(self.signalLength - self.windowSize) : -1])


#%% Implantation  Example:

if __name__ == "__main__":
    
    # Set sliding window size:
    windowSize = 128
    
    # Set desired tset signal length:
    signal_length = 10000
    
    # Time step duration:
    step_size = 1
    
    # Define time vector for test signal:
    time = np.arange(0, signal_length, step_size)
    
    # Base signal:
    signal = np.random.normal(size=signal_length)
    
    # increase signal regularity for final 1000 datapoints
    signal[-1001:-1] += np.sin(time[-1001:-1]) * 4 * np.arange(0, 1, 0.001)
    calibration_start = 0
    calibration_stop = 2000
    
    # Setting random seed for repeatability
    np.random.seed(42)
    
    example = flutterFeatureExtractor(
        "example", windowSize, 0, calibrationFlag=False, numArFeatures=2
    )
    
    # iterate over signal values
    for i in tqdm(range(len(signal))):
        # add current value to flutter precursor object
        example.get_measurement(signal[i])
    
        # When reaching end of calibration section, start calibration process
        if i == calibration_stop:
            example.calibrationFlag = True
            example.calibration(calibration_start, calibration_stop)
    
    
    #%%
    
    # Plot Regularity Features:
    # ========================
    
    
    f1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex="col", figsize=(15.5, 7.5))
    
    ax1.set_title("Dataset Regularity features vs. Time")
    ax1.plot(time[windowSize:], example.lzc, color="darkorange")
    ax1.plot(
        time[windowSize:], np.ones(np.shape(example.lzc)) * example.threshold_lzc, "--r"
    )
    ax1.grid("on")
    ax1.set_ylabel("LZC")
    
    ax2.plot(time[windowSize:], np.array(example.arCod), color="red")
    ax2.plot(
        time[windowSize:], np.ones(np.shape(example.arCod)) * example.threshold_arcod, "--r"
    )
    ax2.grid("on")
    ax2.set_ylabel("arCod")
    
    ax3.plot(time[windowSize:], example.uval, color="green")
    ax3.plot(
        time[windowSize:], np.ones(np.shape(example.uval)) * example.threshold_uval, "--r"
    )
    ax3.grid("on")
    ax3.set_ylabel("U-val")
    
    ax4.plot(time[windowSize:], example.spectralEntropy, color="blue")
    ax4.plot(
        time[windowSize:],
        np.ones(np.shape(example.spectralEntropy)) * example.threshold_spectralEntropy,
        "--r",
    )
    ax4.grid("on")
    ax4.set_ylabel("Spectral\nEntropy")
    
    ax5.plot(
        time[windowSize:],
        example.onlineSignal[windowSize:] / np.max(example.onlineSignal[windowSize:]),
        "k",
    )
    ax5.grid("on")
    ax5.set_ylabel("Signal \n[norm.]")
