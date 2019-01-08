#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# from  https://github.com/BlueCocoa/imguidedfilter-opencv


import cv2
import numpy as np


def imguidedfilter(A, G, filtSize, inversionEpsilon):
    """images.internal.algimguidedfilter @ matlab R2018a
    A: numpy ndaary image float [0, 1], [h,w,c]
    G: numpy ndaary image float [0, 1], [h,w,c]
    filtSize: 2-element tuple (h, w)
    """
    sizeA = A.shape
    if len(sizeA) == 2:
        sizeA = (sizeA[0], sizeA[1], 1)
    sizeG = G.shape
    if len(sizeG) == 2:
        sizeG = (sizeG[0], sizeG[1], 1)

    doMultiChannelCovProcessing = sizeG[2] > sizeA[2]
    isGuidanceChannelReuseNeeded = sizeG[2] < sizeA[2]

    approximateGuidedFilter = False
    subsampleFactor = 1
    useIntergralFiltering = False
    # NotImplemented for nargin >= 5

    originalClassA = A.dtype
    A = A.astype(np.double)
    G = G.astype(np.double)

    B = np.zeros(A.shape, dtype=A.dtype)

    if not doMultiChannelCovProcessing:
        # Iprime = I[0:subsampleFactor:-1, 0:subsampleFactor:-1]
        # since we did not implement subsampleFactor
        I = G[:, :, 1]

        for k in range(0, sizeA[2]):
            # P = A[0:subsampleFactor:-1, 0:subsampleFactor:-1,k]
            # since we did not implement subsampleFactor
            P = A[:, :, k]

            if not isGuidanceChannelReuseNeeded:
                I = G[:, :, k]
            Iprime = I

            meanI = cv2.boxFilter(Iprime, -1, filtSize)
            meanP = cv2.boxFilter(P, -1, filtSize)
            corrI = cv2.boxFilter(Iprime * Iprime, -1, filtSize)
            corrIP = cv2.boxFilter(Iprime * P, -1, filtSize)

            varI = corrI - meanI * meanI
            covIP = corrIP - meanI * meanP

            a = covIP / (varI + inversionEpsilon)
            b = meanP - a * meanI

            meana = cv2.boxFilter(a, -1, filtSize)
            meanb = cv2.boxFilter(b, -1, filtSize)

            B[:, :, k] = meana * I + meanb
    else:
        # Iprime = G[0:subsampleFactor:-1, 0:subsampleFactor:-1, :]
        # since we did not implement subsampleFactor
        Iprime = G
        meanIrgb = cv2.boxFilter(Iprime, -1, filtSize)

        meanIr = meanIrgb[:, :, 0]
        meanIg = meanIrgb[:, :, 1]
        meanIb = meanIrgb[:, :, 2]

        # P = A[0:subsampleFactor:-1, 0:subsampleFactor:-1,:]
        # since we did not implement subsampleFactor
        P = A

        meanP = cv2.boxFilter(P, -1, filtSize)

        IP = Iprime * P.reshape((sizeA[0], sizeA[1], 1))
        corrIrP = cv2.boxFilter(IP[:, :, 0], -1, filtSize)
        corrIgP = cv2.boxFilter(IP[:, :, 1], -1, filtSize)
        corrIbP = cv2.boxFilter(IP[:, :, 2], -1, filtSize)

        varIrr = cv2.boxFilter(Iprime[:, :, 0] * Iprime[:, :, 0], -1, filtSize) - meanIr * meanIr + inversionEpsilon
        varIrg = cv2.boxFilter(Iprime[:, :, 0] * Iprime[:, :, 1], -1, filtSize) - meanIr * meanIg
        varIrb = cv2.boxFilter(Iprime[:, :, 0] * Iprime[:, :, 2], -1, filtSize) - meanIr * meanIb
        varIgg = cv2.boxFilter(Iprime[:, :, 1] * Iprime[:, :, 1], -1, filtSize) - meanIg * meanIg + inversionEpsilon
        varIgb = cv2.boxFilter(Iprime[:, :, 1] * Iprime[:, :, 2], -1, filtSize) - meanIg * meanIb
        varIbb = cv2.boxFilter(Iprime[:, :, 2] * Iprime[:, :, 2], -1, filtSize) - meanIb * meanIb + inversionEpsilon

        covIrP = corrIrP - meanIr * meanP
        covIgP = corrIgP - meanIg * meanP
        covIbP = corrIbP - meanIb * meanP

        invMatEntry11 = varIgg * varIbb - varIgb * varIgb
        invMatEntry12 = varIgb * varIrb - varIrg * varIbb
        invMatEntry13 = varIrg * varIgb - varIgg * varIrb

        covDet = (invMatEntry11 * varIrr) + (invMatEntry12 * varIrg) + (invMatEntry13 * varIrb)

        a = np.zeros(sizeG, dtype=P.dtype)

        a[:, :, 0] = ((invMatEntry11 * covIrP) + \
                      ((varIrb * varIgb - varIrg * varIbb) * covIgP) + \
                      ((varIrg * varIgb - varIrb * varIgg) * covIbP)) / covDet

        a[:, :, 1] = ((invMatEntry12 * covIrP) + \
                      ((varIrr * varIbb - varIrb * varIrb) * covIgP) + \
                      ((varIrb * varIrg - varIrr * varIgb) * covIbP)) / covDet

        a[:, :, 2] = ((invMatEntry13 * covIrP) + \
                      ((varIrg * varIrb - varIrr * varIgb) * covIgP) + \
                      ((varIrr * varIgg - varIrg * varIrg) * covIbP)) / covDet

        b = meanP - (a[:, :, 0] * meanIr) - (a[:, :, 1] * meanIg) - (a[:, :, 2] * meanIb)

        a = cv2.boxFilter(a, -1, filtSize)
        b = cv2.boxFilter(b, -1, filtSize)

        B = np.sum(a * G, axis=2) + b
    return B
