# worker.py
import numpy as np
import math
from obspy.taup import TauPyModel

from subArrivalTimeForward import subArrivalTimeForward
from sunDeleteRefractedWave import subDeleteRefractedWave

# 注意：确保subArrivalTimeForward和subDeleteRefractedWave在主程序中可用
# 或者将它们复制到worker.py中

def _calc_one_depth(args):
    """计算单个深度的匹配结果"""
    (tmp_depth, velModel, recDisInDeg, recDepth, crustInterfaceDepths,
     arrTimeDiffTole, pickedPhaseTimeDiffZ, pickedPhaseTimeDiffR, 
     pickedPhaseTimeDiffT, pickedPhaseTimeZ, pickedPhaseTimeR, 
     pickedPhaseTimeT) = args
    
    try:
        # 直接S波
        phaList = ["s", "Sg"]
        arrivals, rays = subArrivalTimeForward(velModel, tmp_depth, recDisInDeg, phaList, recDepth)
        arrivals, rays = subDeleteRefractedWave(crustInterfaceDepths, tmp_depth, arrivals, rays)
        while len(arrivals) < 1:
            arrivals, rays = subArrivalTimeForward(velModel, tmp_depth, recDisInDeg, phaList, recDepth)
            arrivals, rays = subDeleteRefractedWave(crustInterfaceDepths, tmp_depth, arrivals, rays)
            tmp_depth += 0.1
        
        onsetCalS = arrivals[0].time
        
        # 所有深度相候选
        phaList = ["p", "Pg", "pPg", "sPg", "PvmP", "pPvmP", "sPvmP", 
                  "Sg", "sSg", "SvmS", "sSvmS"]
        arrivals, rays = subArrivalTimeForward(velModel, tmp_depth, recDisInDeg, phaList, recDepth)
        arrivals, rays = subDeleteRefractedWave(crustInterfaceDepths, tmp_depth, arrivals, rays)
        
        # 计算P波和S波的走时差
        taupOriginalName = []
        arrivalsTimeDiffP = []
        arrivalsTimeDiffS = []
        
        for i in range(len(arrivals)):
            taupOriginalName.append(arrivals[i].name)
            arrivalsTimeDiffP.append(arrivals[i].time - arrivals[0].time)
            arrivalsTimeDiffS.append(arrivals[i].time - onsetCalS)
            arrivals[i].name = i
        
        # Z分量匹配
        matchedTaupPhaseOrgNameZ, matchedTaupPhaseDigNameZ = [], []
        matchedTaupTimeDiffZ, matchedPickedTimeZ, matchedPickedTimeDiffZ = [], [], []
        
        tmpArrivalsTimeDiffP = arrivalsTimeDiffP.copy()
        tmpPickedPhaseTimeDiffZ = list(pickedPhaseTimeDiffZ) if pickedPhaseTimeDiffZ else []
        tmpPickedPhaseTimeZ = list(pickedPhaseTimeZ) if pickedPhaseTimeZ else []
        
        loopFlag = 1
        while loopFlag == 1:
            for i in range(len(tmpArrivalsTimeDiffP)):
                for x in range(len(tmpPickedPhaseTimeDiffZ)):
                    if tmpArrivalsTimeDiffP[i] > 0.0 and tmpPickedPhaseTimeDiffZ[x] > 0.0:
                        if math.fabs(tmpArrivalsTimeDiffP[i] - tmpPickedPhaseTimeDiffZ[x]) <= arrTimeDiffTole:
                            matchedPickedTimeDiffZ.append(tmpPickedPhaseTimeDiffZ[x])
                            matchedTaupTimeDiffZ.append(tmpArrivalsTimeDiffP[i])
                            matchedTaupPhaseDigNameZ.append(arrivals[i].name)
                            matchedTaupPhaseOrgNameZ.append(taupOriginalName[i])
                            matchedPickedTimeZ.append(tmpPickedPhaseTimeZ[x])
                            tmpPickedPhaseTimeDiffZ.pop(x)
                            tmpPickedPhaseTimeZ.pop(x)
                            break
            loopFlag = 0
        
        # R分量匹配
        matchedTaupPhaseOrgNameR, matchedTaupPhaseDigNameR = [], []
        matchedTaupTimeDiffR, matchedPickedTimeR, matchedPickedTimeDiffR = [], [], []
        
        tmpArrivalsTimeDiffP = arrivalsTimeDiffP.copy()
        tmpPickedPhaseTimeDiffR = list(pickedPhaseTimeDiffR) if pickedPhaseTimeDiffR else []
        tmpPickedPhaseTimeR = list(pickedPhaseTimeR) if pickedPhaseTimeR else []
        
        loopFlag = 1
        while loopFlag == 1:
            for i in range(len(tmpArrivalsTimeDiffP)):
                for x in range(len(tmpPickedPhaseTimeDiffR)):
                    if tmpArrivalsTimeDiffP[i] > 0.0 and tmpPickedPhaseTimeDiffR[x] > 0.0:
                        if math.fabs(tmpArrivalsTimeDiffP[i] - tmpPickedPhaseTimeDiffR[x]) <= arrTimeDiffTole:
                            matchedPickedTimeDiffR.append(tmpPickedPhaseTimeDiffR[x])
                            matchedTaupTimeDiffR.append(tmpArrivalsTimeDiffP[i])
                            matchedTaupPhaseDigNameR.append(arrivals[i].name)
                            matchedTaupPhaseOrgNameR.append(taupOriginalName[i])
                            matchedPickedTimeR.append(tmpPickedPhaseTimeR[x])
                            tmpPickedPhaseTimeDiffR.pop(x)
                            tmpPickedPhaseTimeR.pop(x)
                            break
            loopFlag = 0
        
        # T分量匹配
        matchedTaupPhaseOrgNameT, matchedTaupPhaseDigNameT = [], []
        matchedTaupTimeDiffT, matchedPickedTimeT, matchedPickedTimeDiffT = [], [], []
        
        tmpArrivalsTimeDiffS = arrivalsTimeDiffS.copy()
        tmpPickedPhaseTimeDiffT = list(pickedPhaseTimeDiffT) if pickedPhaseTimeDiffT else []
        tmpPickedPhaseTimeT = list(pickedPhaseTimeT) if pickedPhaseTimeT else []
        
        loopFlag = 1
        while loopFlag == 1:
            for i in range(len(tmpArrivalsTimeDiffS)):
                for x in range(len(tmpPickedPhaseTimeDiffT)):
                    if tmpArrivalsTimeDiffS[i] > 0.0 and tmpPickedPhaseTimeDiffT[x] > 0.0:
                        if math.fabs(tmpArrivalsTimeDiffS[i] - tmpPickedPhaseTimeDiffT[x]) <= arrTimeDiffTole:
                            matchedPickedTimeDiffT.append(tmpPickedPhaseTimeDiffT[x])
                            matchedTaupTimeDiffT.append(tmpArrivalsTimeDiffS[i])
                            matchedTaupPhaseDigNameT.append(arrivals[i].name)
                            matchedTaupPhaseOrgNameT.append(taupOriginalName[i])
                            matchedPickedTimeT.append(tmpPickedPhaseTimeT[x])
                            tmpPickedPhaseTimeDiffT.pop(x)
                            tmpPickedPhaseTimeT.pop(x)
                            break
            loopFlag = 0
        
        # 计算统计量
        totNumCalPha = 0
        totNumMatPha = 0
        percNumMatPha = 0.0
        
        if (len(matchedTaupTimeDiffZ) >= 1 and
            len(matchedTaupTimeDiffR) >= 1 and
            len(matchedTaupTimeDiffT) >= 1):
            totNumCalPha = len(arrivals) * 3
            totNumMatPha = (len(set(matchedTaupPhaseDigNameZ)) +
                           len(set(matchedTaupPhaseDigNameR)) +
                           len(set(matchedTaupPhaseDigNameT)))
            if totNumCalPha > 0:
                percNumMatPha = totNumMatPha / totNumCalPha
        
        # 计算RMS
        avgArrTimeDiffResZ = 9999.0
        avgArrTimeDiffResR = 9999.0
        avgArrTimeDiffResT = 9999.0
        avgArrTimeDiffResSum = 9999.0
        
        if len(matchedTaupTimeDiffZ) >= 1:
            avgArrTimeDiffResZ = np.sum(
                np.fabs(np.array(matchedPickedTimeDiffZ) - np.array(matchedTaupTimeDiffZ))
            ) / len(matchedTaupTimeDiffZ)
        
        if len(matchedTaupTimeDiffR) >= 1:
            avgArrTimeDiffResR = np.sum(
                np.fabs(np.array(matchedPickedTimeDiffR) - np.array(matchedTaupTimeDiffR))
            ) / len(matchedTaupTimeDiffR)
        
        if len(matchedTaupTimeDiffT) >= 1:
            avgArrTimeDiffResT = np.sum(
                np.fabs(np.array(matchedPickedTimeDiffT) - np.array(matchedTaupTimeDiffT))
            ) / len(matchedTaupTimeDiffT)
        
        if (len(matchedTaupTimeDiffZ) >= 1 and
            len(matchedTaupTimeDiffR) >= 1 and
            len(matchedTaupTimeDiffT) >= 1):
            avgArrTimeDiffResSum = (avgArrTimeDiffResZ + avgArrTimeDiffResR + avgArrTimeDiffResT) / 3.0
        
        return {
            'depth': tmp_depth,
            'totNumCalPha': totNumCalPha,
            'totNumMatPha': totNumMatPha,
            'percNumMatPha': percNumMatPha,
            'avgArrTimeDiffResZ': avgArrTimeDiffResZ,
            'avgArrTimeDiffResR': avgArrTimeDiffResR,
            'avgArrTimeDiffResT': avgArrTimeDiffResT,
            'avgArrTimeDiffResSum': avgArrTimeDiffResSum,
            'matchedTaupPhaseOrgNameZ': matchedTaupPhaseOrgNameZ,
            'matchedTaupPhaseDigNameZ': matchedTaupPhaseDigNameZ,
            'matchedTaupPhaseOrgNameR': matchedTaupPhaseOrgNameR,
            'matchedTaupPhaseDigNameR': matchedTaupPhaseDigNameR,
            'matchedTaupPhaseOrgNameT': matchedTaupPhaseOrgNameT,
            'matchedTaupPhaseDigNameT': matchedTaupPhaseDigNameT,
            'matchedPickedTimeDiffZ': matchedPickedTimeDiffZ,
            'matchedTaupTimeDiffZ': matchedTaupTimeDiffZ,
            'matchedPickedTimeZ': matchedPickedTimeZ,
            'matchedPickedTimeDiffR': matchedPickedTimeDiffR,
            'matchedTaupTimeDiffR': matchedTaupTimeDiffR,
            'matchedPickedTimeR': matchedPickedTimeR,
            'matchedPickedTimeDiffT': matchedPickedTimeDiffT,
            'matchedTaupTimeDiffT': matchedTaupTimeDiffT,
            'matchedPickedTimeT': matchedPickedTimeT,
            'has_valid_matches': (len(matchedTaupTimeDiffZ) >= 1 and
                                  len(matchedTaupTimeDiffR) >= 1 and
                                  len(matchedTaupTimeDiffT) >= 1)
        }
    
    except Exception as e:
        print(f"Error computing depth {tmp_depth}: {e}")
        # 返回一个默认的空结果
        return {
            'depth': tmp_depth,
            'totNumCalPha': 0,
            'totNumMatPha': 0,
            'percNumMatPha': 0.0,
            'avgArrTimeDiffResZ': 9999.0,
            'avgArrTimeDiffResR': 9999.0,
            'avgArrTimeDiffResT': 9999.0,
            'avgArrTimeDiffResSum': 9999.0,
            'matchedTaupPhaseOrgNameZ': [],
            'matchedTaupPhaseDigNameZ': [],
            'matchedTaupPhaseOrgNameR': [],
            'matchedTaupPhaseDigNameR': [],
            'matchedTaupPhaseOrgNameT': [],
            'matchedTaupPhaseDigNameT': [],
            'matchedPickedTimeDiffZ': [],
            'matchedTaupTimeDiffZ': [],
            'matchedPickedTimeZ': [],
            'matchedPickedTimeDiffR': [],
            'matchedTaupTimeDiffR': [],
            'matchedPickedTimeR': [],
            'matchedPickedTimeDiffT': [],
            'matchedTaupTimeDiffT': [],
            'matchedPickedTimeT': [],
            'has_valid_matches': False
        }