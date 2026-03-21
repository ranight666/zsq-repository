import math
from subArrivalTimeForward import subArrivalTimeForward
from sunDeleteRefractedWave import subDeleteRefractedWave

def compute_one_depth(srcDepth,
                      ist,
                      idx,
                      velModel,
                      epiDisEachStation,
                      recDepth,
                      crustInterfaceDepths,
                      depthCandidatePhaOrgNameGlobalZ,
                      depthCandidateArrDiffGlobalZ,
                      depthCandidatePhaOrgNameGlobalR,
                      depthCandidateArrDiffGlobalR,
                      depthCandidatePhaOrgNameGlobalT,
                      depthCandidateArrDiffGlobalT,
                      arrTimeDiffTole):

        """
        对单个 srcDepth 进行原来 for 循环里的全部计算，
        并返回该深度对应的 rmsZ, rmsR, rmsT, sumRmsRTZ, count3 等。
        """

        taupOriginalName = []
        arrivalsTimeDiffP = []
        arrivalsTimeDiffS = []

        # -------- 计算 direct S 到时（onsetCalS） --------
        phaList = ["s", "Sg"]
        arrivals, rays = subArrivalTimeForward(velModel, srcDepth,
                                               epiDisEachStation[ist],
                                               phaList, recDepth)
        arrivals, rays = subDeleteRefractedWave(crustInterfaceDepths,
                                                srcDepth,
                                                arrivals, rays)
        # 保证有direct S
        while len(arrivals) < 1:
            srcDepth += 0.1
            phaList = ["s", "Sg"]
            arrivals, rays = subArrivalTimeForward(velModel, srcDepth,
                                                   epiDisEachStation[ist],
                                                   phaList, recDepth)
            arrivals, rays = subDeleteRefractedWave(crustInterfaceDepths,
                                                    srcDepth,
                                                    arrivals, rays)
        onsetCalS = arrivals[0].time

        # -------- 计算所有候选相位的到时差 --------
        phaList = ["p", "Pg", "pPg", "sPg", "PvmP", "pPvmP", "sPvmP",
                   "Sg", "sSg", "SvmS", "sSvmS"]
        arrivals, rays = subArrivalTimeForward(velModel, srcDepth,
                                               epiDisEachStation[ist],
                                               phaList, recDepth)
        arrivals, rays = subDeleteRefractedWave(crustInterfaceDepths,
                                                srcDepth,
                                                arrivals, rays)

        for i in range(len(arrivals)):
            taupOriginalName.append(arrivals[i].name)
            arrivalsTimeDiffP.append(arrivals[i].time - arrivals[0].time)
            arrivalsTimeDiffS.append(arrivals[i].time - onsetCalS)
            arrivals[i].name = i  # 用编号区分相位

        # -------- 计算 rmsZ, rmsR, rmsT --------
        rmsZ = 1.0
        rmsR = 1.0
        rmsT = 1.0
        tmp3 = 0.0
        count3 = 0

        # Z 分量
        tmp = 0.0
        count = 0
        for i in range(len(depthCandidatePhaOrgNameGlobalZ[ist][idx])):
            for j in range(len(arrivals)):
                if (depthCandidatePhaOrgNameGlobalZ[ist][idx][i]
                    == taupOriginalName[j]) and \
                        (math.fabs(depthCandidateArrDiffGlobalZ[ist][idx][i]
                                   - arrivalsTimeDiffP[j]) <= arrTimeDiffTole):
                    tmp += (depthCandidateArrDiffGlobalZ[ist][idx][i]
                            - arrivalsTimeDiffP[j]) ** 2
                    count += 1
        if count > 0:
            tmp3 += tmp
            rmsZ = math.sqrt(tmp / count)
            count3 += count

        # R 分量
        tmp = 0.0
        count = 0
        for i in range(len(depthCandidatePhaOrgNameGlobalR[ist][idx])):
            for j in range(len(arrivals)):
                if (depthCandidatePhaOrgNameGlobalR[ist][idx][i]
                    == taupOriginalName[j]) and \
                        (math.fabs(depthCandidateArrDiffGlobalR[ist][idx][i]
                                   - arrivalsTimeDiffP[j]) <= arrTimeDiffTole):
                    tmp += (depthCandidateArrDiffGlobalR[ist][idx][i]
                            - arrivalsTimeDiffP[j]) ** 2
                    count += 1
        if count > 0:
            tmp3 += tmp
            rmsR = math.sqrt(tmp / count)
            count3 += count

        # T 分量
        tmp = 0.0
        count = 0
        for i in range(len(depthCandidatePhaOrgNameGlobalT[ist][idx])):
            for j in range(len(arrivals)):
                if (depthCandidatePhaOrgNameGlobalT[ist][idx][i]
                    == taupOriginalName[j]) and \
                        (math.fabs(depthCandidateArrDiffGlobalT[ist][idx][i]
                                   - arrivalsTimeDiffS[j]) <= arrTimeDiffTole):
                    tmp += (depthCandidateArrDiffGlobalT[ist][idx][i]
                            - arrivalsTimeDiffS[j]) ** 2
                    count += 1
        if count > 0:
            tmp3 += tmp
            rmsT = math.sqrt(tmp / count)
            count3 += count

        # 三个分量平均
        sumRmsRTZ = (rmsR + rmsT + rmsZ) / 3.0


        return {
            "srcDepth": srcDepth,
            "rmsZ": rmsZ,
            "rmsR": rmsR,
            "rmsT": rmsT,
            "sumRmsRTZ": sumRmsRTZ,
            "match_count": count3
        }
    
