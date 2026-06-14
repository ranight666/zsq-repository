#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 01 00:00:00 2020

@author: Jianlong Yuan (yuan_jianlong@126.com)
    
    Supervisors: Honn Kao & Jiashun Yu
myplot.png

Modified by:
    Hongyu Yu, Shaoqi Zhang(2420566673@qq.com), and Yanjiu Wu(40433155@qq.com)
    March 2026

Parallelization notes:
    Based on the WCSB-adapted version, this version further improves the  computational 
    efficiency of the DSA workflow. The main modification is the parallelization of the 
    station and depth loops in Step 3 (preliminary focal-depth  determination) and Step 4 
    (final solution based on travel-time residuals).These changes reduce runtime while 
    preserving the original depth-scanning logic and output format.
    
    
Algorithm name: 
        Depth-Scanning Algorithm (DSA)


Framework:
 1. Automatic generation of synthetic waveforms for all possible depth phases.    
 2. Match-filtering of all possible depth phases.
 3. Preliminary determination of the focal depth.
 4. Final solution based on travel time residuals.
    

Input:
  1. Three-component waveforms.
      Notice: SAC format. Header at least should has corrected 'dist' and 'baz'.
              header.b = 0
  2. Velocity model.
      Notice: TauP Toolkit format (see Section 5 in(1-10 Hz, cc = 0.7, dep: 2.8 km
0-10 km (vel model: crust1)

              https://www.seis.sc.edu/downloads/TauP/taup.pdf )

Output:
  Focal depth (median) 
  
  
More details see in our preprint submitted to JGR: Solid Earth entitled:
    “Depth-Scanning Algorithm: Accurate, Automatic, and Efficient Determination
     of Focal Depths for Local and Regional Earthquakes” by Jianlong Yuan,
     Honn Kao, and Jiashun Yu


Get a preprint or have questions? Please contact Jianlong Yuan at:
    yuan_jianlong@126.com
     
"""

from obspy.taup import TauPyModel, taup_create
import matplotlib.pyplot as plt
from obspy.geodetics.base import kilometer2degrees
from obspy.core import UTCDateTime
import matplotlib.pyplot as pltDebug
import numpy as np
from scipy.signal import hilbert, find_peaks
from obspy import read, read_inventory
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy.stats import kurtosis as kurt
import scipy.stats as stats
from scipy import signal
from scipy.signal import argrelextrema
from itertools import chain
import os, fnmatch, sys
import timeit
start = timeit.default_timer()
import shutil
import csv
plt.close("all")
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic('reset -sf')
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from worker import compute_one_depth
sys.path.append(os.getcwd())
from work import _calc_one_depth

#%%-- subroutine: load input parameters from 'DSA_SETTINGS.txt'
def load_settings():
    '''
     PARAMETER          DESCRIPTION
     
     par1    data directory, including wavefroms and velocity model 
     par2    velocity model name (this string should not include '.nd')    
     par3    tolerance between the observed and predicted differential travel times (second)
     par4    cross-correlation coefficient threshold
     par5    minimal frequency used for band-pass filter (Hz)
     par6    maximal frequency used for band-pass filter (Hz)
     par7    minimal scanning depth candidate (interger, km)
     par8    maximal scanning depth candidate (interger, km)
     par9    for monitoring: 1 -> active,  0 -> inactive
     par10   plot Steps 1 and 2 of DSA: 1 -> active,  0 -> inactive
    '''
    
    try:
        SETTINGS = pd.read_csv('/home/zsq/DSA/DSA_SETTINGS_WCSB_general.right', delim_whitespace=True, index_col='PARAMETER')
        #SETTINGS = pd.read_csv('./DSA_v1/DSA_SETTINGS_AB_WD.txt', delim_whitespace=True, index_col='PARAMETER')
        #SETTINGS = pd.read_csv('./DSA_v1/DSA_SETTINGS_RM_evt.txt', delim_whitespace=True, index_col='PARAMETER')
        #SETTINGS = pd.read_csv('D:/汾河地堑浅源地震/DSA/DSA_SETTINGS_Montney_Bei.txt', delim_whitespace=True, index_col='PARAMETER')
        par1 = SETTINGS.VALUE.loc['dataPath'] 
        par2 = SETTINGS.VALUE.loc['velModel']   
        par3 = float( SETTINGS.VALUE.loc['arrTimeDiffTole'] ) 
        par4 = float( SETTINGS.VALUE.loc['ccThreshold']  )
        par5 = float( SETTINGS.VALUE.loc['frequencyFrom']  )
        par6 = float( SETTINGS.VALUE.loc['frequencyTo']  )
        par7 =   int( SETTINGS.VALUE.loc['scanDepthFrom']  )
        par8 =   int( SETTINGS.VALUE.loc['scanDepthTo']  )
        par9 =   int( SETTINGS.VALUE.loc['verboseFlag']  )
        par10=   int( SETTINGS.VALUE.loc['plotSteps1n2Flag']  )
        
        return par1, par2, par3, par4, par5, par6, par7, par8, par9, par10    
    
    except:
        sys.exit("Errors in 'DSA_SETTINGS.txt' !\n")
        



#%%-- subroutine: Extract top depths of the crust interfaces from velocity model
def GetCrustInterfaceDepths( velModel ):
    crustInterfaceDepths = []
    
    # velMod = pd.read_csv(str(dataPath)+str(velModel)+'.nd',
    #                      delim_whitespace=True, header=None,
    #                      names=['TopDepth', 'Vp', 'Vs', 'Rho', 'Qp', 'Qs'])
    velMod = pd.read_csv('/home/zsq/DSA/test/vel_model/'+str(velModel)+'.nd',
                         delim_whitespace=True, header=None,
                         names=['TopDepth', 'Vp', 'Vs', 'Rho', 'Qp', 'Qs'])
    
    for irow in range( len(velMod) ):
        ival = velMod.loc[irow]['TopDepth']
    
        if ival == 'mantle':
            break  
        
        try:
            ival = np.float64( ival )
            if ival > 0.0:
                crustInterfaceDepths.append(ival)
        except:
            continue

    crustInterfaceDepths = sorted(set(crustInterfaceDepths))
    return crustInterfaceDepths


#%%-- subroutine: cross-correlation
def xcorrssl( scanTimeBeg, scanTimeEnd, tem, tra ):
    
    temLeng = len(tem)
    traLeng = len(tra)
    time_lags= traLeng - temLeng + 1
    
    #-- demean for the template
    b = tem - np.mean(tem)
    corr_norm_idx = []
    corr_norm_val = []
    
    for k in range( time_lags ):
        if ( k >= scanTimeBeg and k <=scanTimeEnd ):
            # demean for the trace
            a = tra[k:(k+temLeng)] - np.mean(tra[k:(k+temLeng)])
            stdev = (np.sum(a**2)) ** 0.5 * (np.sum(b**2)) ** 0.5
            if stdev != 0:
                corr = np.sum(a*b)/stdev
            else:
                corr = 0
            corr_norm_idx.append(k)
            corr_norm_val.append(corr)
        else:
            corr_norm_idx.append(k)
            corr_norm_val.append(0.)
            
    return corr_norm_val

#-- subroutine: arrival time forward modelling kernel
def subArrivalTimeForward( velModel, srcDepth, recDisInDeg, phaList, recDepth ):
    model = TauPyModel(model= velModel )
    
    try:
        arrivals = model.get_travel_times(source_depth_in_km=srcDepth,
                                           distance_in_degree=recDisInDeg,
                                           phase_list= phaList,
                                           receiver_depth_in_km=recDepth)
        
        rays = model.get_ray_paths(source_depth_in_km=srcDepth,
                                   distance_in_degree=recDisInDeg,
                                   phase_list= phaList,
                                   receiver_depth_in_km=recDepth)
    except: # avoid TauP error
        srcDepth += 0.1
        arrivals = model.get_travel_times(source_depth_in_km=srcDepth,
                                           distance_in_degree=recDisInDeg,
                                           phase_list= phaList,
                                           receiver_depth_in_km=recDepth)
        
        rays = model.get_ray_paths(source_depth_in_km=srcDepth,
                                   distance_in_degree=recDisInDeg,
                                   phase_list= phaList,
                                   receiver_depth_in_km=recDepth)
        
        
    # correct phase name (e.g., PvmP is code name, PmP is academic name)
    for i in range(len(arrivals)):
        if arrivals[i].name == 'PvmP':
            arrivals[i].name = 'PmP'
        if arrivals[i].name == 'pPvmP':
            arrivals[i].name = 'pPmP'
        if arrivals[i].name == 'sPvmP':
            arrivals[i].name = 'sPmP'
        if arrivals[i].name == 'SvmS':
            arrivals[i].name = 'SmS'
        if arrivals[i].name == 'sSvmS':
            arrivals[i].name = 'sSmS'    
    
    return  arrivals, rays

#-- subroutine: filter out some strange rays that do not arrive at the staion,
#   and delete some refracted waves.
def subDeleteRefractedWave( crustInterfaceDepths, srcDepth, arrivals, rays ):        
    removeIdx = []
    nRays = len(rays)
    
    ###################################################################
    #--  filter out some strange rays that do not arrive at the staion:
    # 1. get distance value of the last point of each ray
    # 2. get the median value of the distances
    # 3. find out the distance that is different with the median
    # 4. if the ratio (the different distance value / median) > 10 %, then
    #    delete this distance
    ###################################################################
    dist  = []
    for iRay in range( nRays ):
        nPointsRay = np.shape( rays[iRay].path )
        lastPtIdx = nPointsRay[0]
        dist.append( rays[iRay].path[ lastPtIdx-1 ][2] )
    if( len(dist) > 0 ): # avoid unstable situation
        median = np.median( dist )
        #print( "median=", median )
        
        for iRay in range( nRays ):
            disRatio = np.fabs( ( dist[ iRay ] - median ) / median )
            #print( "disRatio={0}".format( format(disRatio,".3f" )))
            if disRatio > 0.1:
                removeIdx.append(iRay)
    
    
    #%%-- delete some refracted waves, which are related to 
     # the interfaces by using:
     # 1) find out the maximum depth of each ray
     # 2) delete the ray whose maximum depth is not located at the interface    
    for iRay in range( nRays ):
        depthData = []
        nPointsRay = np.shape( rays[iRay].path )
        for i in np.arange( 1, nPointsRay[0], 1 ):
            depthData.append( rays[iRay].path[i][3] )
        maxRayDepth = np.max( depthData )
        #print(  maxRayDepth )
             
        if( arrivals[iRay].name != "p" and arrivals[iRay].name != "s" ):
            rayIsRefractedWave = 1
            for iDep in crustInterfaceDepths:
                if maxRayDepth == iDep  or maxRayDepth == srcDepth:
                    rayIsRefractedWave = 0
            if rayIsRefractedWave == 1:
                removeIdx.append( iRay )
        

        for i in range( nPointsRay[0] - 2):
            i1 = i
            i2 = i+1
            i3 = i+2
            # rays[iRay].path[i][3], in which 3 is the index of "depth"
            if rays[iRay].path[i1][3] == rays[iRay].path[i3][3]:
                if ((rays[iRay].path[i1][3] < rays[iRay].path[i2][3]) and 
                    (rays[iRay].path[i2][3] > rays[iRay].path[i3][3]) ):
                       err12 = rays[iRay].path[i1][3] - rays[iRay].path[i2][3]
                       err23 = rays[iRay].path[i2][3] - rays[iRay].path[i3][3]
                       if np.fabs(err12) < 0.5 or np.fabs(err23) < 0.5:
                           removeIdx.append(iRay)
                           break
                
    #-- deleting using reverse order
    if len(removeIdx) > 0:
        removeIdxInv = sorted( set( list(removeIdx) ), reverse=True )

        for i in removeIdxInv:
            #print(i)
            arrivals.remove( arrivals[i] )
            rays.remove( rays[i] )
            
    return  arrivals, rays






#%%
###################################################
# Input parameters
###################################################
'''
 PARAMETER          DESCRIPTION
 
 dataPath           data directory, including wavefroms and velocity model 
 velModel           velocity model name (this string should not include '.nd')    
 arrTimeDiffTole    tolerance between the observed and predicted differential travel times (second)
 ccThreshold        cross-correlation coefficient threshold
 frequencyFrom      minimal frequency used for band-pass filter (Hz)
 frequencyTo        maximal frequency used for band-pass filter (Hz)
 scanDepthFrom      minimal scanning depth candidate (interger, km)
 scanDepthTo        maximal scanning depth candidate (interger, km)
 verboseFlag        for monitoring: 1 -> active,  0 -> inactive
 plotSteps1n2Flag   plot Steps 1 and 2 of DSA: 1 -> active,  0 -> inactive
'''

#%%-- load input parameters
dataPath, velModel, arrTimeDiffTole, ccThreshold, frequencyFrom, frequencyTo,\
scanDepthFrom, scanDepthTo, verboseFlag, plotSteps1n2Flag = load_settings()
    
#%%-- get the number of waveform files (HH* components)
wfFiles = fnmatch.filter( sorted(os.listdir(dataPath)), '*.SAC')
numSt = int( len(wfFiles)/3 ) # 3 -> three components
for i in range( numSt):
    print( '\t SAC files in the directory:')
    print( wfFiles[i*3], wfFiles[i*3+1], wfFiles[i*3+2] )

#%%-- create output file directory
outfilePath = str(dataPath)+'results_cc'+str(ccThreshold)+'_'+str(frequencyFrom)+'_'+str(frequencyTo)+'hz_'+\
              str(velModel)+'_'+str(scanDepthFrom)+'_'+str(scanDepthTo)+'km/'
if not os.path.exists(str(outfilePath)):
    os.mkdir(str(outfilePath))
else:
    print( '"resluts" already exists!')
    shutil.rmtree(str(outfilePath))
    os.mkdir(str(outfilePath))

#%%-- output file, here to create newfile    
outPath = str(outfilePath)+'/LocatingResults.csv'
with open( '{0}'.format( outPath ), mode='w', newline=''  ) as resultsFile:
    writer = csv.writer( resultsFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow( [ 'StName', 'Az(deg)', 'EpDis(deg)', 'NumMatPha_step3','NumMatPha_step4'
                       'Loc(km)', 'SubDivDep(km)',' Rms(s)', 'MinRms(s)' ] )

    
#%%-- print key information
print( '\n==========- INPUT PARAMETERS -==========\n')
print( 'dataPath           =', dataPath)
print( 'velModel           =', velModel)
print( 'arrTimeDiffTole    =', arrTimeDiffTole)
print( 'ccThreshold        =', ccThreshold )
print( 'frequencyFrom      =', frequencyFrom )
print( 'frequencyTo        =', frequencyTo )
print( 'scanDepthFrom      =', scanDepthFrom )
print( 'scanDepthTo        =', scanDepthTo )
print( 'verboseFlag        =', verboseFlag )
print( 'plotSteps1n2Flag   =', plotSteps1n2Flag )
print( 'Number of stations =', numSt )
print( 'outfilePath        =', outfilePath)
print( '\n======================================\n')    
    
    

#%%-- velocity for tauP
# taup_create.build_taup_model( str(dataPath)+str(velModel)+'.nd' )
taup_create.build_taup_model( '/home/zsq/DSA/test/vel_model/'+''+str(velModel)+'.nd' )
    
#%%-- Allocate memory
numScanDepth = int(scanDepthTo-scanDepthFrom)
idxMaxNumPhaEachStation = np.zeros((numSt))
idxMaxNumPhaEachStation.fill(9999) # initial array with a high value
azimuthEachStation= np.zeros((numSt))
epiDisEachStation = np.zeros((numSt))
onsetPEachStation = np.zeros((numSt))
onsetSEachStation = np.zeros((numSt))
totNumMatPhaGlobal  = np.zeros((numSt, numScanDepth))
totNumCalPhaGlobal  = np.zeros((numSt, numScanDepth))
percNumMatPhaGlobal = np.zeros((numSt, numScanDepth))
avgArrTimeDiffResEachStationZ = np.zeros((numSt, numScanDepth))
avgArrTimeDiffResEachStationZ.fill(9999) # initial array with a high value
avgArrTimeDiffResEachStationR = np.zeros((numSt, numScanDepth))
avgArrTimeDiffResEachStationR.fill(9999) # initial array with a high value
avgArrTimeDiffResEachStationT = np.zeros((numSt, numScanDepth))
avgArrTimeDiffResEachStationT.fill(9999) # initial array with a high value
avgArrTimeDiffResEachStationSum = np.zeros((numSt, numScanDepth))
avgArrTimeDiffResEachStationSum.fill(9999) # initial array with a high value
sumAvgArrTimeDiffResGlobal = np.zeros((numScanDepth))
sumAvgArrTimeDiffResGlobal.fill(9999) # initial array with a high value
sumAvgArrTimeDiffResGlobal_2 = np.zeros((numScanDepth))
sumAvgArrTimeDiffResGlobal_2.fill(9999) # initial array with a high value
Sta_timediffR = []
Sta_timediffT = []
Sta_timediffZ = []
Sta_distkm = []
Sta_stanm = []
Sta_stalat = []
Sta_stalon = []
Sta_dt_final= np.zeros((numSt))
Sta_dt_final.fill(9999)
Sta_match = np.zeros((numSt))
Sta_dep_final = np.zeros((numSt))
Sta_best = np.zeros((numSt))

nameEachStation        = [[] for i in range(numSt)]
histogramGlobalZ       = [[] for i in range(numSt)]
histogramGlobalR       = [[] for i in range(numSt)]
histogramGlobalT       = [[] for i in range(numSt)]
template0GlobalZ       = [[] for i in range(numSt)]
template0GlobalR       = [[] for i in range(numSt)]
template0GlobalT       = [[] for i in range(numSt)]
template60GlobalZ      = [[] for i in range(numSt)]
template60GlobalR      = [[] for i in range(numSt)]
template60GlobalT      = [[] for i in range(numSt)]
template120GlobalZ     = [[] for i in range(numSt)]
template120GlobalR     = [[] for i in range(numSt)]
template120GlobalT     = [[] for i in range(numSt)]
template170GlobalZ     = [[] for i in range(numSt)]
template170GlobalR     = [[] for i in range(numSt)]
template170GlobalT     = [[] for i in range(numSt)]
waveformGlobalZ        = [[] for i in range(numSt)]
waveformGlobalR        = [[] for i in range(numSt)]
waveformGlobalT        = [[] for i in range(numSt)]
normWaveformGlobalZ    = [[] for i in range(numSt)]
normWaveformGlobalR    = [[] for i in range(numSt)]
normWaveformGlobalT    = [[] for i in range(numSt)]
bigAmpGlobalZ          = [[] for i in range(numSt)]
bigAmpGlobalR          = [[] for i in range(numSt)]
bigAmpGlobalT          = [[] for i in range(numSt)]
bigAmpTimeGlobalZ      = [[] for i in range(numSt)]
bigAmpTimeGlobalR      = [[] for i in range(numSt)]
bigAmpTimeGlobalT      = [[] for i in range(numSt)]
peaksCurveGlobalZ      = [[] for i in range(numSt)]
peaksCurveGlobalR      = [[] for i in range(numSt)]
peaksCurveGlobalT      = [[] for i in range(numSt)]
finalpeaksPtsGlobalZ   = [[] for i in range(numSt)]
finalpeaksPtsGlobalR   = [[] for i in range(numSt)]
finalpeaksPtsGlobalT   = [[] for i in range(numSt)]
finalCcGlobalZ         = [[] for i in range(numSt)]
finalCcGlobalR         = [[] for i in range(numSt)]
finalCcGlobalT         = [[] for i in range(numSt)]
phaShiftAngGlobalZ     = [[] for i in range(numSt)]
phaShiftAngGlobalR     = [[] for i in range(numSt)]
phaShiftAngGlobalT     = [[] for i in range(numSt)]
phaShiftAngTimeGlobalZ = [[] for i in range(numSt)]
phaShiftAngTimeGlobalR = [[] for i in range(numSt)]
phaShiftAngTimeGlobalT = [[] for i in range(numSt)]
leftBoundry1GlobalZ    = [[] for i in range(numSt)]
leftBoundry1GlobalR    = [[] for i in range(numSt)]
leftBoundry1GlobalT    = [[] for i in range(numSt)]
rightBoundry1GlobalZ   = [[] for i in range(numSt)]
rightBoundry1GlobalR   = [[] for i in range(numSt)]
rightBoundry1GlobalT   = [[] for i in range(numSt)]
DTGlobalZ              = [[] for i in range(numSt)]
DTGlobalR              = [[] for i in range(numSt)]
DTGlobalT              = [[] for i in range(numSt)]
st_begin_timeNorGlobalZ= [[] for i in range(numSt)]
st_begin_timeNorGlobalR= [[] for i in range(numSt)]
st_begin_timeNorGlobalT= [[] for i in range(numSt)]
wantedTimeLengGlobalZ  = [[] for i in range(numSt)]
wantedTimeLengGlobalR  = [[] for i in range(numSt)]
wantedTimeLengGlobalT  = [[] for i in range(numSt)]
temBegNorGlobalZ       = [[] for i in range(numSt)]
temBegNorGlobalR       = [[] for i in range(numSt)]
temBegNorGlobalT       = [[] for i in range(numSt)]
corrLengGlobalZ        = [[] for i in range(numSt)]
corrLengGlobalR        = [[] for i in range(numSt)]
corrLengGlobalT        = [[] for i in range(numSt)]

depthCandidateArrGlobalZ        = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrGlobalR        = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrGlobalT        = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrDiffGlobalZ    = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrDiffGlobalR    = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrDiffGlobalT    = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrTaupDiffGlobalZ    = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrTaupDiffGlobalR    = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrTaupDiffGlobalT    = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidatePhaDigNameGlobalZ = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidatePhaDigNameGlobalR = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidatePhaDigNameGlobalT = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidatePhaOrgNameGlobalZ = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidatePhaOrgNameGlobalR = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidatePhaOrgNameGlobalT = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrdtGlobalR = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrdtGlobalT = [[[] for i in range(numScanDepth)] for j in range(numSt)]
depthCandidateArrdtGlobalZ = [[[] for i in range(numScanDepth)] for j in range(numSt)]





def process_one_station(ist, dataPath, wfFiles, velModel, arrTimeDiffTole,
                        ccThreshold, frequencyFrom, frequencyTo, scanDepthFrom,
                        scanDepthTo, verboseFlag, crustInterfaceDepths,
                        recDepth, srcDepthScanInc,numScanDepth):

    infileE = open('{0}/{1}'.format( dataPath,wfFiles[ist*3+0] ) )
    infileN = open('{0}/{1}'.format( dataPath, wfFiles[ist*3+1] ) )
    infileZ = open('{0}/{1}'.format( dataPath, wfFiles[ist*3+2] ) )
    print( infileE.name )
    #%%--
    stRawE = read(infileE.name, debug_headers=True)
    stRawN = read(infileN.name, debug_headers=True)
    stRawZ = read(infileZ.name, debug_headers=True)
    
    #%%-- Get header information
    evLa = stRawZ[0].stats.sac.evla # event's latitude.  unit: degree
    evLo = stRawZ[0].stats.sac.evlo # event's longitude. unit: degree    
    stLa = stRawZ[0].stats.sac.stla # station's latitude.  unit: degree
    stLo = stRawZ[0].stats.sac.stlo # station's latitude.  unit: degree
    recDisInKm = stRawZ[0].stats.sac.dist
    recDisInDeg = kilometer2degrees(recDisInKm)
    nameEachStation[ist].append( stRawZ[0].stats.sac.kstnm.strip() )
    epiDisEachStation[ist] = recDisInDeg
    azimuthEachStation[ist] = stRawZ[0].stats.sac.az
    baz = stRawZ[0].stats.sac.baz
    DT = stRawZ[0].stats.sac.delta
    st_begin_timeNor = stRawZ[0].stats.sac.o
    st_begin_timeUTC = UTCDateTime(stRawZ[0].stats.starttime+st_begin_timeNor) #this is not accurate, because it assumes b=0
    # st_begin_timeUTC = UTCDateTime(stRawZ[0].stats.starttime-stRawZ[0].stats.sac.b+st_begin_timeNor)

    print("\n\n\n========================================================")
    print('Now processing:')
    print(infileE.name)
    print(infileN.name)
    print(infileZ.name)
    print("Station = ", stRawZ[0].stats.sac.kstnm )
    print("Scanning station id=", ist)
    print("Az, Baz = ", stRawZ[0].stats.sac.az, baz )
    print("evLa, evLo = ", evLa, evLo )
    print("stLa, stLo = ", stLa, stLo )
    print("recDisInKm  = ",recDisInKm)
    print("recDisInDeg = ", recDisInDeg)
    print("Dt = ", DT)
    print("stRawZ[0].stats.sac.dist = ", stRawZ[0].stats.sac.dist )
    print("stRawZ[0].stats.starttime", stRawZ[0].stats.starttime)
    print("st_begin_timeNor = ", st_begin_timeNor)
    print("st_begin_timeUTC = ", st_begin_timeUTC)
    print("--------------------------------------------------------\n")
    
    #%%-- Waveform scanning window used for DSA ([t.b, t.o + 40 or 70])
    if recDisInKm < 80:
        wantedTimeLengZ = 40
    elif recDisInKm < 150:
        wantedTimeLengZ = 70
    else:
        wantedTimeLengZ = 110

        
    
    #-- This commend is only for the synthetic example in Section 3.2 of DSA paper
    if velModel == "ak135_Section3.2":
        wantedTimeLengZ = 22
        
    #-- This commend is only for the synthetic example in Section3.3 of DSA paper
    if velModel == "ak135_Section3.3":
        wantedTimeLengZ = 30
        
    #%%-- These two commends are only for test diffetent azimuthal coverages
    '''
    if stRawZ[0].stats.sac.az < 180 or stRawZ[0].stats.sac.az >= 270:
           continue
    '''      
    
    #%%-- Extract wavefroms within wanted time window
    stWantedE = stRawE.trim(st_begin_timeUTC, st_begin_timeUTC+wantedTimeLengZ)
    stWantedN = stRawN.trim(st_begin_timeUTC, st_begin_timeUTC+wantedTimeLengZ)
    stWantedZ = stRawZ.trim(st_begin_timeUTC, st_begin_timeUTC+wantedTimeLengZ)

    
    #%%-- Remove mean value and trend
    stWantedE0 = stWantedE.copy()
    stWantedN0 = stWantedN.copy()
    stWantedZ0 = stWantedZ.copy()
    stWantedE0[0].detrend( type='demean')
    stWantedN0[0].detrend( type='demean')
    stWantedZ0[0].detrend( type='demean')
    stWantedE0[0].detrend( type='simple')
    stWantedN0[0].detrend( type='simple')
    stWantedZ0[0].detrend( type='simple')   

    
    #%%-- Remove response
    try:
        # inv = read_inventory( "{0}/{1}.{2}.xml".format( dataPath, stRawZ[0].stats.network, stRawZ[0].stats.station) )
        if stRawZ[0].stats.network != 'QM':
            inv = read_inventory( "{0}/{1}.xml".format( dataPath, stRawZ[0].stats.station) )
            pre_filt = (0.005, 0.006, 30.0, 35.0)

            stWantedE0[0].remove_response(inventory=inv, output="DISP", pre_filt=pre_filt)
            stWantedN0[0].remove_response(inventory=inv, output="DISP", pre_filt=pre_filt)
            stWantedZ0[0].remove_response(inventory=inv, output="DISP", pre_filt=pre_filt)
    
        #%% -- do ratation from Z12 to ZNE
        if stRawE[0].stats.channel == 'HH1' or stRawE[0].stats.channel == 'BH1':
            stZ12 = stWantedZ0 + stWantedE0 + stWantedN0
            print(stZ12)
            stZ12.rotate( method='->ZNE', inventory=inv )
            stWantedZ0 = stZ12.select(component="Z")
            stWantedN0 = stZ12.select(component="N")
            stWantedE0 = stZ12.select(component="E")
        
        #%%-- do ratation: NE to RT using back-azimuth angle
        stNE = stWantedN0 + stWantedE0
        stNE.rotate( method='NE->RT', back_azimuth=baz )
        stR0 = stNE.select(component="R")
        stT0 = stNE.select(component="T")        
        stZ0 = stWantedZ0.copy()
#        print(stNE)
#        print(stR0)
#        print(stT0)    
        
        #%%-- taper before filtering
        stZ0[0] = stZ0[0].taper(max_percentage=0.1, side='left')
        stR0[0] = stR0[0].taper(max_percentage=0.1, side='left')
        stT0[0] = stT0[0].taper(max_percentage=0.1, side='left')

        print( 'Real data, remove response done!\n')

    except:
        print( 'No response file! Maybe synthetic data?\n')
        stR0 = stWantedE0.copy()
        stT0 = stWantedN0.copy()        
        stZ0 = stWantedZ0.copy()
        
        
    #%%-- frequency filtering
    stZ0[0] = stZ0[0].filter('bandpass', freqmin=frequencyFrom, freqmax=frequencyTo,
                                         corners=4, zerophase=False)
    stR0[0] = stR0[0].filter('bandpass', freqmin=frequencyFrom, freqmax=frequencyTo,
                                         corners=4, zerophase=False)
    stT0[0] = stT0[0].filter('bandpass', freqmin=frequencyFrom, freqmax=frequencyTo,
                                         corners=4, zerophase=False)

    #%%-- check waveform
    #print('\n Check Z, R, and T waveforms: \n')
    #stZ0.plot()
    #stR0.plot()
    #stT0.plot()
    #################

    #%%-- Get P and S onset using kurtosis of scipy 
    #-- Z

    findOnsetIdxFlagZ = 0
    dfZ = pd.DataFrame()
    dfZ['stZ0[0]'] = stZ0[0].data
    dfZ['kurtosisZ'] = dfZ['stZ0[0]'].rolling(200).apply(kurt, raw=True)
    kurtosisZ = dfZ[ 'kurtosisZ' ]
    maxKurZ = np.max( np.fabs( kurtosisZ ))
    for i in range( len(kurtosisZ) ):
        if kurtosisZ[i] > (maxKurZ*0.99):
            onsetBegIdxZ = i
            findOnsetIdxFlagZ = 1
            break
    if findOnsetIdxFlagZ == 0:
        onsetBegIdxZ = 0
    onsetZ = onsetBegIdxZ*DT
    #stRawZ[0].stats.sac.t5 = -12345
    #stRawZ[0].stats.sac.t5 = stRawZ[0].stats.sac.t3
    if stRawZ[0].stats.sac.t4 != -12345:
        onsetZ = stRawZ[0].stats.sac.t4 - stRawZ[0].stats.sac.o
        findOnsetIdxFlagZ = 1
    onsetZNor = onsetZ
    onsetZUTC = UTCDateTime( stRawZ[0].stats.starttime + onsetZNor)
    print( "onsetZNor=", onsetZNor)
    print( "onsetZUTC=", onsetZUTC)
    
    #-- R

    findOnsetIdxFlagR = 0
    dfR = pd.DataFrame()
    dfR['stR0[0]'] = stR0[0].data
    dfR['kurtosisR'] = dfR['stR0[0]'].rolling(200).apply(kurt, raw=True)
    kurtosisR = dfR[ 'kurtosisR' ]
    maxKurR = np.max( np.fabs( kurtosisR ))
    for i in range( len(kurtosisR) ):
        if kurtosisR[i] > (maxKurR*0.99):
            onsetBegIdxR = i
            findOnsetIdxFlagR = 1
            break
    if findOnsetIdxFlagR == 0:
        onsetBegIdxR = 0
    onsetR = onsetBegIdxR*DT
    #stRawZ[0].stats.sac.t5 = -12345
    #stRawZ[0].stats.sac.t5 = stRawZ[0].stats.sac.t3
    if stRawZ[0].stats.sac.t4 != -12345:
        onsetR = stRawZ[0].stats.sac.t4 - stRawZ[0].stats.sac.o
        findOnsetIdxFlagR = 1

    onsetRNor = onsetR
    onsetRUTC = UTCDateTime( stRawZ[0].stats.starttime + onsetRNor)
    print( "onsetRNor=", onsetRNor)
    print( "onsetRUTC=", onsetRUTC)
    
    # The scanning time of S-wave starts at a time = onset of P-wave 

    scanBegTimeT = onsetR
    scanBegTimeIdxT = int(scanBegTimeT/DT)
    findOnsetIdxFlagT = 0
    dfT = pd.DataFrame()
    dfT['stT0[0]'] = stT0[0].data[scanBegTimeIdxT:]
    dfT['kurtosisT'] = dfT['stT0[0]'].rolling(500).apply(kurt, raw=True)
    kurtosisT = dfT[ 'kurtosisT' ]
    maxKurT = np.max( np.fabs( kurtosisT ))
    #-- scan from 3 s after the onset time of P-wave
    for i in range( len(kurtosisT) ):
        if kurtosisT[i] > (maxKurT*0.99):
            onsetBegIdxT = i + scanBegTimeIdxT
            findOnsetIdxFlagT = 1
            break
    if findOnsetIdxFlagT == 0:
        onsetBegIdxT = 0
    onsetT = onsetBegIdxT*DT
    #stRawZ[0].stats.sac.t4 = -12345
    if stRawZ[0].stats.sac.t3 != -12345:
        onsetT = stRawZ[0].stats.sac.t3 - stRawZ[0].stats.sac.o
        findOnsetIdxFlagT = 1

    onsetTNor = onsetT
    onsetTUTC = UTCDateTime( stRawZ[0].stats.starttime + onsetTNor)
    print( "onsetTNor=", onsetTNor)
    print( "onsetTUTC=", onsetTUTC)

    #%%-- here we will use two conditions to evaluate current station:
    # Condition 1: when onset time of Z/R/T cannot match its theoretical
    # arrival time, meaning this station is unreliable, then skip it
    recDepth = 0  # station's depth( default 0 km)
    refDepth = 10  #stRawZ[0].stats.sac.evdp # default depth for calculate referenced onset time of direct wave (km)
    crustInterfaceDepths = GetCrustInterfaceDepths( velModel )
    print( 'crustInterfaceDepths = ', crustInterfaceDepths )
    phaList = [ "p", "Pg" ]
    arrivals, rays = subArrivalTimeForward( velModel, refDepth, recDisInDeg, phaList, recDepth )
    arrivals, rays = subDeleteRefractedWave( crustInterfaceDepths, refDepth, arrivals, rays )
    calOnsetP = arrivals[0].time
    print( arrivals )
    print( "calOnsetP=", calOnsetP )            

    
    phaList = [ "s", "Sg" ]
    arrivals, rays = subArrivalTimeForward( velModel, refDepth, recDisInDeg, phaList, recDepth )
    arrivals, rays = subDeleteRefractedWave( crustInterfaceDepths, refDepth, arrivals, rays )
    calOnsetS = arrivals[0].time
    print( arrivals )
    print( "calOnsetS=", calOnsetS )

    if np.fabs(onsetZNor + stRawZ[0].stats.sac.b - calOnsetP) > 5 and \
       np.fabs(onsetRNor + stRawZ[0].stats.sac.b - calOnsetP) <= 5:
        onsetZNor = onsetRNor
        onsetZUTC = UTCDateTime(stRawZ[0].stats.starttime + onsetZNor)
    if np.fabs(onsetZNor + stRawZ[0].stats.sac.b - calOnsetP) <= 5 and \
            np.fabs(onsetRNor + stRawZ[0].stats.sac.b - calOnsetP) > 5:
        onsetRNor = onsetZNor
        onsetRUTC = UTCDateTime(stRawZ[0].stats.starttime + onsetRNor)

    Sta_timediffR.append(onsetRNor+stRawZ[0].stats.sac.b-calOnsetP)
    Sta_timediffT.append(onsetTNor+stRawZ[0].stats.sac.b-calOnsetS)
    Sta_timediffZ.append(onsetZNor+stRawZ[0].stats.sac.b-calOnsetP)
    Sta_distkm.append(recDisInKm)
    Sta_stanm.append(stRawZ[0].stats.sac.kstnm)
    Sta_stalat.append(stRawZ[0].stats.sac.stla)
    Sta_stalon.append(stRawZ[0].stats.sac.stlo)

    if np.fabs( onsetZNor+stRawZ[0].stats.sac.b-calOnsetP ) > 5 or\
       np.fabs( onsetRNor+stRawZ[0].stats.sac.b-calOnsetP ) > 5 or\
       np.fabs( onsetTNor+stRawZ[0].stats.sac.b-calOnsetS ) > 5:
        print("\n Onset time match failed, skip station=", stRawZ[0].stats.sac.kstnm, 
                  "EpiDis:", recDisInKm, "km \n")
        return ist, None
        
    
    #-- Condition 2: when all kurtosis values of Z/R/T is failed to meet the
    # threshold, meaning this station is with low-quality S/N, then skip it
    if findOnsetIdxFlagZ == 0 or\
       findOnsetIdxFlagR == 0 or\
       findOnsetIdxFlagT == 0:
        print("\n Cannot find onset time, skip station=",
                  stRawZ[0].stats.sac.kstnm, 
                 "EpiDis:",
                 recDisInKm, "km\n\n\n")
        return ist, None
    
        
    #%%-- plot kurtosis function
    if verboseFlag == 1:
        print('\n Check kurtosis picking: \n')
        kurNorZ = kurtosisZ / np.max(np.fabs(kurtosisZ))
        stNorZ0 = stZ0[0].data / np.max(np.fabs(stZ0[0].data))
        tKur = np.arange( 0, len(kurtosisZ), 1)*DT+stRawZ[0].stats.sac.b
        tT0  = np.arange( 0, len(stNorZ0), 1)*DT+stRawZ[0].stats.sac.b
        pltDebug.figure(figsize=(12,2))
        pltDebug.tick_params(axis='both', which='major', labelsize=10)
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Normalized Amp.', fontsize=12)
        pltDebug.title( 'Z: Large amplitudes (grey) and kurtosis (blue)', fontsize=12 )
        pltDebug.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
        pltDebug.plot( tKur, kurNorZ )
        pltDebug.plot( tT0, stNorZ0*1.0+1, color='lightgray' )
        pltDebug.scatter( onsetZ+stRawZ[0].stats.sac.b, 0.25, marker="o", s=100, label='Picked onset',
                          facecolor='none', edgecolor='black', lw=1, zorder=101 )
        pltDebug.scatter( calOnsetP, 0.5, marker="o", s=100, label='Theoretical onset', 
                          facecolor='none', edgecolor='red', lw=1, zorder=101 )
        pltDebug.margins(0)
        pltDebug.legend(prop={"size":10}, loc='upper right')

        kurNorR = kurtosisR / np.max(np.fabs(kurtosisR))
        stNorR0 = stR0[0].data / np.max(np.fabs(stR0[0].data))
        tKur = np.arange( 0, len(kurtosisR), 1)*DT+stRawZ[0].stats.sac.b
        tT0  = np.arange( 0, len(stNorR0), 1)*DT+stRawZ[0].stats.sac.b
        pltDebug.figure(figsize=(12,2))
        pltDebug.tick_params(axis='both', which='major', labelsize=10)
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Normalized Amp.', fontsize=12)
        pltDebug.title( 'R: Large amplitudes (grey) and kurtosis (blue)', fontsize=12 )
        pltDebug.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
        pltDebug.plot( tKur, kurNorR )
        pltDebug.plot( tT0, stNorR0*1.0+1, color='lightgray' )
        pltDebug.scatter( onsetR+stRawZ[0].stats.sac.b, 0.25, marker="o", s=100, label='Picked onset',
                          facecolor='none', edgecolor='black', lw=1, zorder=101 )
        pltDebug.scatter( calOnsetP, 0.5, marker="o", s=100, label='Theoretical onset', 
                          facecolor='none', edgecolor='red', lw=1, zorder=101 )
        pltDebug.margins(0)
        pltDebug.legend(prop={"size":10}, loc='upper right')

        kurNorT = kurtosisT / np.max(np.fabs(kurtosisT))
        stNorT0 = stT0[0].data / np.max(np.fabs(stT0[0].data))                
        tKur = np.arange( 0, len(kurtosisT), 1)*DT+scanBegTimeT+stRawZ[0].stats.sac.b
        tT0  = np.arange( 0, len(stNorT0), 1)*DT+stRawZ[0].stats.sac.b
        pltDebug.figure(figsize=(12,2))
        pltDebug.tick_params(axis='both', which='major', labelsize=10)
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Normalized Amp.', fontsize=12)
        pltDebug.title( 'T: Large amplitudes (grey) and kurtosis (blue)', fontsize=12 )
        pltDebug.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
        pltDebug.plot( tKur, kurNorT )
        pltDebug.plot( tT0, stNorT0*1.0+1, color='lightgray' )
        pltDebug.scatter( onsetT+stRawZ[0].stats.sac.b, 0.25, marker="o", s=100, label='Picked onset',
                          facecolor='none', edgecolor='black', lw=1, zorder=101 )
        pltDebug.scatter( calOnsetS, 0.5, marker="o", s=100, label='Theoretical onset', 
                          facecolor='none', edgecolor='red', lw=1, zorder=101 )
        pltDebug.margins(0)
        pltDebug.legend(prop={"size":10}, loc='upper right')
        
        plt.show()
                
    #%%-- first time to roughly chose direct wave [onset-0.5s, onset+0.5s]
    stZ1 = stZ0.copy()
    temZtBegNor = onsetZNor - 0.5
    temZtEndNor = onsetZNor + 0.5
    temZtBegUTC = UTCDateTime( stRawZ[0].stats.starttime + temZtBegNor)
    temZtEndUTC = UTCDateTime( stRawZ[0].stats.starttime + temZtEndNor )
    print("temZtBegUTC=", temZtBegUTC)
    print("temZtEndUTC=", temZtEndUTC)
    templateZ=stZ1.trim( temZtBegUTC, temZtEndUTC)
    maxAmpTemZ = np.max(templateZ[0].data)
    print("maxAmpTemZ=", maxAmpTemZ )
    
    stR1 = stR0.copy()
    temRtBegNor = onsetRNor - 0.5
    temRtEndNor = onsetRNor + 0.5
    temRtBegUTC = UTCDateTime( stRawZ[0].stats.starttime + temRtBegNor )
    temRtEndUTC = UTCDateTime( stRawZ[0].stats.starttime + temRtEndNor )
    print("temRtBegUTC", temRtBegUTC)
    templateR=stR1.trim( temRtBegUTC, temRtEndUTC)
    maxAmpTemR = np.max(templateR[0].data)
    print("maxAmpTemR=", maxAmpTemR )
    
    stT1 = stT0.copy()
    temTtBegNor = onsetTNor - 0.5
    temTtEndNor = onsetTNor + 0.5
    temTtBegUTC = UTCDateTime( stRawZ[0].stats.starttime + temTtBegNor )
    temTtEndUTC = UTCDateTime( stRawZ[0].stats.starttime + temTtEndNor )
    templateT=stT1.trim( temTtBegUTC, temTtEndUTC)
    maxAmpTemT = np.max(templateT[0].data)
    print("maxAmpTemT=", maxAmpTemT )

    
    # #-- find the minimum and maximum amplitudes and their arrival times
    # minAmpTemValZ = np.min(templateZ[0].data)
    # minAmpTemIdxZ = np.argmin(templateZ[0].data)
    # minAmpTemTimeZ = minAmpTemIdxZ * DT + temZtBegNor
    # maxAmpTemValZ = np.max(templateZ[0].data)
    # maxAmpTemIdxZ = np.argmax(templateZ[0].data)
    # maxAmpTemTimeZ = maxAmpTemIdxZ * DT + temZtBegNor
    #
    # minAmpTemValT = np.min(templateT[0].data)
    # minAmpTemIdxT = np.argmin(templateT[0].data)
    # minAmpTemTimeT = minAmpTemIdxT * DT + temTtBegNor
    # maxAmpTemValT = np.max(templateT[0].data)
    # maxAmpTemIdxT = np.argmax(templateT[0].data)
    # maxAmpTemTimeT =  maxAmpTemIdxT * DT + temTtBegNor
    #
    # halfCircleTimeZ = np.fabs( minAmpTemTimeZ - maxAmpTemTimeZ )
    # halfCircleTimeT = np.fabs( minAmpTemTimeT - maxAmpTemTimeT )

    # -- find the maximum amplitudes (abs) and local minimum amplitudes and their arrival times

    maxAmpTemValZ = np.max(np.fabs(templateZ[0].data))
    maxAmpTemIdxZ = np.argmax(np.fabs(templateZ[0].data))
    maxAmpTemTimeZ = maxAmpTemIdxZ * DT + temZtBegNor
    if templateZ[0].data[maxAmpTemIdxZ]>0:
        minAmpTemIdxZ_all = argrelextrema(templateZ[0].data, np.less)
        minAmpTemIdxZ_tmp = np.argmin(np.fabs(minAmpTemIdxZ_all[0]-maxAmpTemIdxZ))
        minAmpTemIdxZ_a = minAmpTemIdxZ_all[0][minAmpTemIdxZ_tmp]
        if minAmpTemIdxZ_a < maxAmpTemIdxZ:
            if minAmpTemIdxZ_tmp == len(minAmpTemIdxZ_all[0])-1:
                minAmpTemIdxZ_b = minAmpTemIdxZ_a
            else:
                minAmpTemIdxZ_b = minAmpTemIdxZ_all[0][minAmpTemIdxZ_tmp+1]
        else:
            if minAmpTemIdxZ_tmp == 0:
                minAmpTemIdxZ_b = minAmpTemIdxZ_a
            else:
                minAmpTemIdxZ_b = minAmpTemIdxZ_all[0][minAmpTemIdxZ_tmp - 1]

        if templateZ[0].data[minAmpTemIdxZ_a]>templateZ[0].data[minAmpTemIdxZ_b]:
            minAmpTemIdxZ_a = minAmpTemIdxZ_b
        minAmpTemIdxZ = minAmpTemIdxZ_a
        minAmpTemValZ = templateZ[0].data[minAmpTemIdxZ_a]
    else:
        minAmpTemIdxZ_all = argrelextrema(templateZ[0].data, np.greater)
        minAmpTemIdxZ_tmp = np.argmin(np.fabs(minAmpTemIdxZ_all[0] - maxAmpTemIdxZ))
        minAmpTemIdxZ_a = minAmpTemIdxZ_all[0][minAmpTemIdxZ_tmp]
        if minAmpTemIdxZ_a < maxAmpTemIdxZ:
            if minAmpTemIdxZ_tmp == len(minAmpTemIdxZ_all[0])-1:
                minAmpTemIdxZ_b = minAmpTemIdxZ_a
            else:
                minAmpTemIdxZ_b = minAmpTemIdxZ_all[0][minAmpTemIdxZ_tmp + 1]
        else:
            if minAmpTemIdxZ_tmp == 0:
                minAmpTemIdxZ_b = minAmpTemIdxZ_a
            else:
                minAmpTemIdxZ_b = minAmpTemIdxZ_all[0][minAmpTemIdxZ_tmp - 1]

        if templateZ[0].data[minAmpTemIdxZ_a] < templateZ[0].data[minAmpTemIdxZ_b]:
            minAmpTemIdxZ_a = minAmpTemIdxZ_b

        minAmpTemIdxZ = minAmpTemIdxZ_a
        minAmpTemValZ = templateZ[0].data[minAmpTemIdxZ_a]

    minAmpTemTimeZ = minAmpTemIdxZ * DT + temZtBegNor

    maxAmpTemValT = np.max(np.fabs(templateT[0].data))
    maxAmpTemIdxT = np.argmax(np.fabs(templateT[0].data))
    maxAmpTemTimeT = maxAmpTemIdxT * DT + temTtBegNor
    if templateT[0].data[maxAmpTemIdxT] > 0:
        minAmpTemIdxT_all = argrelextrema(templateT[0].data, np.less)
        minAmpTemIdxT_tmp = np.argmin(np.fabs(minAmpTemIdxT_all[0] - maxAmpTemIdxT))
        minAmpTemIdxT_a = minAmpTemIdxT_all[0][minAmpTemIdxT_tmp]
        if minAmpTemIdxT_a < maxAmpTemIdxT:
            if minAmpTemIdxT_tmp == len(minAmpTemIdxT_all[0])-1:
                minAmpTemIdxT_b = minAmpTemIdxT_a
            else:
                minAmpTemIdxT_b = minAmpTemIdxT_all[0][minAmpTemIdxT_tmp + 1]
        else:
            if minAmpTemIdxT_tmp == 0:
                minAmpTemIdxT_b = minAmpTemIdxT_a
            else:
                minAmpTemIdxT_b = minAmpTemIdxT_all[0][minAmpTemIdxT_tmp - 1]

        if templateT[0].data[minAmpTemIdxT_a] > templateT[0].data[minAmpTemIdxT_b]:
            minAmpTemIdxT_a = minAmpTemIdxT_b

        minAmpTemIdxT = minAmpTemIdxT_a
        minAmpTemValT = templateT[0].data[minAmpTemIdxT_a]
    else:
        minAmpTemIdxT_all = argrelextrema(templateT[0].data, np.greater)
        minAmpTemIdxT_tmp = np.argmin(np.fabs(minAmpTemIdxT_all[0] - maxAmpTemIdxT))
        minAmpTemIdxT_a = minAmpTemIdxT_all[0][minAmpTemIdxT_tmp]
        if minAmpTemIdxT_a < maxAmpTemIdxT:
            if minAmpTemIdxT_tmp == len(minAmpTemIdxT_all[0])-1:
                minAmpTemIdxT_b = minAmpTemIdxT_a
            else:
                minAmpTemIdxT_b = minAmpTemIdxT_all[0][minAmpTemIdxT_tmp + 1]
        else:
            if minAmpTemIdxT_tmp == 0:
                minAmpTemIdxT_b = minAmpTemIdxT_a
            else:
                minAmpTemIdxT_b = minAmpTemIdxT_all[0][minAmpTemIdxT_tmp - 1]

        if templateT[0].data[minAmpTemIdxT_a] < templateT[0].data[minAmpTemIdxT_b]:
            minAmpTemIdxT_a = minAmpTemIdxT_b

        minAmpTemIdxT = minAmpTemIdxT_a
        minAmpTemValT = templateT[0].data[minAmpTemIdxT_a]

    minAmpTemTimeT = minAmpTemIdxT * DT + temTtBegNor


    halfCircleTimeZ = np.fabs(minAmpTemTimeZ - maxAmpTemTimeZ)
    halfCircleTimeT = np.fabs(minAmpTemTimeT - maxAmpTemTimeT)

    print("halfCircleTimeZ = ", halfCircleTimeZ, "sec")
    print("halfCircleTimeT = ", halfCircleTimeT, "sec")
          
    if np.fabs(minAmpTemValZ) > np.fabs(maxAmpTemValZ):
        t0P = minAmpTemTimeZ
        t1P = minAmpTemTimeZ
    else:
        t0P = maxAmpTemTimeZ
        t1P = maxAmpTemTimeZ
    
    if np.fabs(minAmpTemValT) > np.fabs(maxAmpTemValT):
        t0S = minAmpTemTimeT
        t1S = minAmpTemTimeT
    else:
        t0S = maxAmpTemTimeT
        t1S = maxAmpTemTimeT

    #%%-- using 2.5 times periods as the time length of direct wave     
    t0P = t0P - 2.5 * halfCircleTimeZ    
    t1P = t1P + 2.5 * halfCircleTimeZ    
    t0S = t0S - 2.5 * halfCircleTimeT 
    t1S = t1S + 2.5 * halfCircleTimeT
    print("t0P, t1P=", t0P, t1P)
    print("t0S, t1S=", t0S, t1S)
    onsetPEachStation[ist] = t0P
    onsetSEachStation[ist] = t0S

    stZ1 = stZ0.copy()
    temZtBegNor = t0P
    temZtEndNor = t1P
    temZtBegUTC = UTCDateTime( stRawZ[0].stats.starttime + t0P )
    temZtEndUTC = UTCDateTime( stRawZ[0].stats.starttime + t1P )
    templateZ=stZ1.trim( temZtBegUTC, temZtEndUTC)
    maxAmpTemZ = np.max(templateZ[0].data)
    print("maxAmpTemZ=", maxAmpTemZ )  
    
    stR1 = stR0.copy()
    temRtBegNor = t0P
    temRtEndNor = t1P
    temRtBegUTC = UTCDateTime( stRawZ[0].stats.starttime + t0P )
    temRtEndUTC = UTCDateTime( stRawZ[0].stats.starttime + t1P )
    print("temRtBegUTC", temRtBegUTC)
    templateR=stR1.trim( temRtBegUTC, temRtEndUTC)
    maxAmpTemR = np.max(templateR[0].data)
    print("maxAmpTemR=", maxAmpTemR )
    
    stT1 = stT0.copy()
    temTtBegNor = t0S
    temTtEndNor = t1S
    temTtBegUTC = UTCDateTime( stRawZ[0].stats.starttime + t0S )
    temTtEndUTC = UTCDateTime( stRawZ[0].stats.starttime + t1S )
    templateT=stT1.trim( temTtBegUTC, temTtEndUTC)
    maxAmpTemT = np.max(templateT[0].data)
    print("maxAmpTemT=", maxAmpTemT )
        
    #%%-- plot direct-wave templates
    if verboseFlag == 1:       
        print('\n Check the selected direct phases: \n')
        pltDebug.figure(figsize=(5,2))
        t = np.arange( 0, len( templateZ[0] ), 1 )*DT+temZtBegNor+stRawZ[0].stats.sac.b
        pltDebug.axhline( 0, linewidth=0.5, color='gray' )
        pltDebug.plot( t, templateZ[0])
        pltDebug.title( 'P template (Z)' )
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Amplitude', fontsize=12)
        pltDebug.margins(0)
        
        pltDebug.figure( figsize=(5,2))
        t = np.arange( 0, len( templateR[0] ), 1 )*DT+temRtBegNor+stRawZ[0].stats.sac.b
        pltDebug.axhline( 0, linewidth=0.5, color='gray' )
        pltDebug.plot( t, templateR[0] )
        pltDebug.title( 'P template (R)' )
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Amplitude', fontsize=12)
        pltDebug.margins(0)
        
        pltDebug.figure( figsize=(5,2) )
        t = np.arange( 0, len( templateT[0] ), 1 )*DT+temTtBegNor+stRawZ[0].stats.sac.b
        pltDebug.axhline( 0, linewidth=0.5, color='gray' )
        pltDebug.plot( t, templateT[0] )
        pltDebug.title( 'S template (T)' )
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Amplitude', fontsize=12)
        pltDebug.margins(0)
        pltDebug.show()


    #%%##############################################################
    # Step 2: Match-filtering of all possible depth phases by using #
    #         1) phase shifting and 2) match-flitering              #
    #################################################################
    
    phaseShiftStart = -180
    phaseShiftEnd   = 180
    PhaseShiftInc   = 10
    numPhase = int( (phaseShiftEnd-phaseShiftStart)/PhaseShiftInc )
    print("Number of phase shift=", numPhase)
    scanTimeBegP = (int) ( temZtBegNor / DT)
    scanTimeEndP = len( stZ0[0] )
    scanTimeBegS = (int) ( temTtBegNor / DT)
    scanTimeEndS = len( stZ0[0] )
    print("P scanning from",scanTimeBegP*DT, "to", scanTimeEndP*DT, "sec")
    print("S scanning from",scanTimeBegS*DT, "to", scanTimeEndS*DT, "sec")

    #-- calculate cross-correlation coefficient (CC) on Z component
    count = 0
    temLengZ = len(templateZ[0])
    traLengZ = len(stZ0[0])
    corrLengZ= traLengZ - temLengZ + 1
    corrValZ = np.zeros((numPhase, corrLengZ))
    
    for phaseShift in range(phaseShiftStart, phaseShiftEnd, PhaseShiftInc):
        #-- Phase shift using Hilbert transform
        st2 = hilbert(templateZ[0])
        st2 = np.real(np.abs(st2) * np.exp((np.angle(st2) +\
                        (phaseShift)/180.0 * np.pi) * 1j))      
        #-- cross-corelation
        corrValZ[count] = xcorrssl( scanTimeBegP, scanTimeEndP, st2, stZ0[0])
        count += 1
        
        #-- plotting for paper
        if phaseShift == 60:
            templateZ60 = st2
        if phaseShift == 120:
            templateZ120 = st2
        if phaseShift == 170:
            templateZ170 = st2

    #-- calculate cross-correlation coefficient (CC) on R component
    count = 0
    temLengR = len(templateR[0])
    traLengR = len(stR0[0])
    corrLengR= traLengR - temLengR + 1
    corrValR = np.zeros((numPhase, corrLengR))    
 
    for phaseShift in range(phaseShiftStart, phaseShiftEnd, PhaseShiftInc):
        #-- Phase shift using Hilbert transform
        st2 = hilbert(templateR[0])
        st2 = np.real(np.abs(st2) * np.exp((np.angle(st2) +\
                        (phaseShift)/180.0 * np.pi) * 1j))          
        #-- cross-corelation
        corrValR[count] = xcorrssl( scanTimeBegP, scanTimeEndP, st2, stR0[0])
        count += 1
        
        #-- plotting for paper
        if phaseShift == 60:
            templateR60 = st2
        if phaseShift == 120:
            templateR120 = st2
        if phaseShift == 170:
            templateR170 = st2
    
    #-- calculate cross-correlation coefficient (CC) on T component        
    count = 0
    temLengT = len(templateT[0])
    traLengT = len(stT0[0])
    corrLengT= traLengT - temLengT + 1
    corrValT = np.zeros((numPhase, corrLengT))
   
    for phaseShift in range(phaseShiftStart, phaseShiftEnd, PhaseShiftInc):
        #-- Phase shift using Hilbert transform
        st2 = hilbert(templateT[0])
        st2 = np.real(np.abs(st2) * np.exp((np.angle(st2) +\
                        (phaseShift)/180.0 * np.pi) * 1j))          
        #-- cross-corelation
        corrValT[count] = xcorrssl( scanTimeBegS, scanTimeEndS, st2, stT0[0])
        count += 1
        
        #-- plotting for paper
        if phaseShift == 60:
            templateT60 = st2
        if phaseShift == 120:
            templateT120 = st2
        if phaseShift == 170:
            templateT170 = st2
    

        
    #%%-- Get the maximum cross-correlation value of each template
    PickCorrR = np.amax( corrValR, axis=0 )
    PickCorrT = np.amax( corrValT, axis=0 )
    PickCorrZ = np.amax( corrValZ, axis=0 )
    
    #%%-- Get the time lag showing the peak value of cross-correlation  
    peaksR0, _ = find_peaks(PickCorrR, height=ccThreshold, distance=50)
    peaksT0, _ = find_peaks(PickCorrT, height=ccThreshold, distance=50)
    peaksZ0, _ = find_peaks(PickCorrZ, height=ccThreshold, distance=50)
    pickedPhaseTimeR0 = peaksR0 * DT
    pickedPhaseTimeT0 = peaksT0 * DT
    pickedPhaseTimeZ0 = peaksZ0 * DT
    

    peaksCurveR = np.zeros( len(PickCorrR) )
    for i in range( len(peaksR0) ):
        peaksCurveR[ peaksR0[i] ] = PickCorrR[ peaksR0[i] ]
    peaksCurveT = np.zeros( len(PickCorrT) )        
    for i in range( len(peaksT0) ):
        peaksCurveT[ peaksT0[i] ] = PickCorrT[ peaksT0[i] ]
    peaksCurveZ = np.zeros( len(PickCorrZ) )
    for i in range( len(peaksZ0) ):
        peaksCurveZ[ peaksZ0[i] ] = PickCorrZ[ peaksZ0[i] ]        
                         
    #%%-- Select phase with large amplitude using the distribution
    # of amplitude peak and drop
    stNorZ = stZ0[0].data / max( np.fabs(stZ0[0].data))
    stNorR = stR0[0].data / max( np.fabs(stR0[0].data))
    stNorT = stT0[0].data / max( np.fabs(stT0[0].data))
    maxAmpTemZ = np.max( np.fabs(templateZ[0].data) ) / max( np.fabs(stZ0[0].data))
    maxAmpTemR = np.max( np.fabs(templateR[0].data) ) / max( np.fabs(stR0[0].data))
    maxAmpTemT = np.max( np.fabs(templateT[0].data) ) / max( np.fabs(stT0[0].data))
    tGlobal = np.arange( 0, wantedTimeLengZ, DT)

    extremaMinIdxZ = signal.argrelextrema( np.array( stNorZ ), np.less)
    extremaMinIdxR = signal.argrelextrema( np.array( stNorR ), np.less)
    extremaMinIdxT = signal.argrelextrema( np.array( stNorT ), np.less)
    extremaMaxIdxZ = signal.argrelextrema( np.array( stNorZ ), np.greater)
    extremaMaxIdxR = signal.argrelextrema( np.array( stNorR ), np.greater)
    extremaMaxIdxT = signal.argrelextrema( np.array( stNorT ), np.greater)      
    extremaIdxZ = np.concatenate( (extremaMinIdxZ, extremaMaxIdxZ), axis=1 )
    extremaIdxR = np.concatenate( (extremaMinIdxR, extremaMaxIdxR), axis=1 )
    extremaIdxT = np.concatenate( (extremaMinIdxT, extremaMaxIdxT), axis=1 )
    
    histZ = sorted(  stNorZ[extremaIdxZ][0] )
    histR = sorted(  stNorR[extremaIdxR][0] )
    histT = sorted(  stNorT[extremaIdxT][0] )
    
    meanZ = np.mean(histZ)
    meanR = np.mean(histR)
    meanT = np.mean(histT)
    stdZ  = np.std(histZ)
    stdR  = np.std(histR)
    stdT  = np.std(histT)
    
    ratioStd1 = 1.0
    leftBoundry1Z  = meanZ - stdZ * ratioStd1
    leftBoundry1R  = meanR - stdR * ratioStd1
    leftBoundry1T  = meanT - stdT * ratioStd1
    rightBoundry1Z = meanZ + stdZ * ratioStd1
    rightBoundry1R = meanR + stdR * ratioStd1
    rightBoundry1T = meanT + stdT * ratioStd1
    

    bigAmpZ0 = np.zeros( len(stNorZ) )
    bigAmpR0 = np.zeros( len(stNorR) )
    bigAmpT0 = np.zeros( len(stNorT) )
    
    #%%-- To show the depth-phase waveform corresponding to the peak/troughthat
    # meets the CC threshold, we keep the waveform within a time-window 
    # centering on the peak/trough amplitude. 
    for i in range (len(stNorZ) ):
        if stNorZ[i] <= leftBoundry1Z or stNorZ[i] >= rightBoundry1Z:
            for j in range( int(temLengZ/2) ):
                if (i-j) >=0 and (i+j) < len(stNorZ):
                    bigAmpZ0[ i-j ] = 1.
                   
    for i in range (len(stNorR) ):
        if stNorR[i] <= leftBoundry1R or stNorR[i] >= rightBoundry1R:
            for j in range( int(temLengR/2) ):
                if (i-j) >=0 and (i+j) < len(stNorR):
                    bigAmpR0[ i-j ] = 1.
                    
    for i in range (len(stNorT) ):
        if stNorT[i] <= leftBoundry1T or stNorT[i] >= rightBoundry1T:
            for j in range( int(temLengT/2) ):
                if (i-j) >=0 and (i+j) < len(stNorT):
                    bigAmpT0[ i-j ] = 1.
                             
        
    #%%-- Select CC for the phase with large amplitude
    PickCorrZ_span = np.zeros( len(PickCorrZ) )      
    for i in range( len(PickCorrZ) ):
        PickCorrZ_span[i] = bigAmpZ0[i] * peaksCurveZ[i]
    peaksZ, _ = find_peaks(PickCorrZ_span, height=ccThreshold, distance=1)
    pickedPhaseTimeZ = peaksZ * DT
    
    PickCorrR_span = np.zeros( len(PickCorrR) )
    for i in range( len(PickCorrR) ):
        PickCorrR_span[i] = bigAmpR0[i] * peaksCurveR[i]
    peaksR, _ = find_peaks(PickCorrR_span, height=ccThreshold, distance=1)
    pickedPhaseTimeR = peaksR * DT    
 
    PickCorrT_span = np.zeros( len(PickCorrT) )
    for i in range( len(PickCorrT) ):
        PickCorrT_span[i] = bigAmpT0[i] * peaksCurveT[i] 
    peaksT, _ = find_peaks(PickCorrT_span, height=ccThreshold, distance=1)
    pickedPhaseTimeT = peaksT * DT
           
    
    #%%-- Evaluate the quality of data by using a condition:
    # If only the direct phase has CC, then skip the current station.
    if (len(pickedPhaseTimeZ) < 2) or\
       (len(pickedPhaseTimeR) < 2) or\
       (len(pickedPhaseTimeT) < 2):
        print( "\n\n\n Only the direct phase has CC, skip station =",
               stRawZ[0].stats.sac.kstnm, 
               "EpiDis:", recDisInKm, "km\n\n\n" )
        return ist, None

    
    #%%-- calculate arrival time differences between the selected phases and
    # the direct waves
    pickedPhaseTimeDiffR = pickedPhaseTimeR - pickedPhaseTimeR0[0]
    print('pickedPhaseTimeDiffR =', pickedPhaseTimeDiffR)
    pickedPhaseTimeDiffT = pickedPhaseTimeT - pickedPhaseTimeT0[0]
    print('pickedPhaseTimeDiffT =', pickedPhaseTimeDiffT)
    pickedPhaseTimeDiffZ = pickedPhaseTimeZ - pickedPhaseTimeZ0[0]
    print('pickedPhaseTimeDiffZ =', pickedPhaseTimeDiffZ)  
    
    #%%-- save the phase-shifting angle of the picked CC
    phaShiftAngR = []
    phaShiftAngIdxR = []
    phaShiftAngTimeR = []
    for i in range( len(peaksR) ):
        for j in range(numPhase):
            if PickCorrR[ peaksR[i] ] == corrValR[j][ peaksR[i] ]:
                phaShiftAngR.append( j*PhaseShiftInc+phaseShiftStart )
                phaShiftAngIdxR.append( peaksR[i] )
                phaShiftAngTimeR.append( peaksR[i]*DT )

    phaShiftAngT = []
    phaShiftAngIdxT = []
    phaShiftAngTimeT = []
    for i in range( len(peaksT) ):
        for j in range(numPhase):
            if PickCorrT[ peaksT[i] ] == corrValT[j][ peaksT[i] ]:
                phaShiftAngT.append( j*PhaseShiftInc+phaseShiftStart )
                phaShiftAngIdxT.append( peaksT[i] )
                phaShiftAngTimeT.append( peaksT[i]*DT )

    phaShiftAngZ = []
    phaShiftAngIdxZ = []
    phaShiftAngTimeZ = []
    for i in range( len(peaksZ) ):
        for j in range(numPhase):
            if PickCorrZ[ peaksZ[i] ] == corrValZ[j][ peaksZ[i] ]:
                phaShiftAngZ.append( j*PhaseShiftInc+phaseShiftStart )
                phaShiftAngIdxZ.append( peaksZ[i] )
                phaShiftAngTimeZ.append( peaksZ[i]*DT )



    #%%#####################################################
    # Step 3: Preliminary determination of the focal depth #
    ########################################################
    #-- search for focal depth using TauP
    scanDepthMembers = []




    if 'stZ0' in locals() and len(stZ0) > 0:
        # 只保存一次
        if not waveformGlobalZ[ist]:
            waveformGlobalZ[ist].append( stZ0[0].data )
            waveformGlobalR[ist].append( stR0[0].data )
            waveformGlobalT[ist].append( stT0[0].data )
            normWaveformGlobalZ[ist].append( stNorZ )
            normWaveformGlobalR[ist].append( stNorR )
            normWaveformGlobalT[ist].append( stNorT )
            template0GlobalZ[ist].append( templateZ[0].data )
            template0GlobalR[ist].append( templateR[0].data )
            template0GlobalT[ist].append( templateT[0].data )
            template60GlobalZ[ist].append( templateZ60 )
            template60GlobalR[ist].append( templateR60 )
            template60GlobalT[ist].append( templateT60 )
            template120GlobalZ[ist].append( templateZ120 )
            template120GlobalR[ist].append( templateR120 )
            template120GlobalT[ist].append( templateT120 )
            template170GlobalZ[ist].append( templateZ170 )
            template170GlobalR[ist].append( templateR170 )
            template170GlobalT[ist].append( templateT170 )
            histogramGlobalZ[ist].append( histZ )
            histogramGlobalR[ist].append( histR )
            histogramGlobalT[ist].append( histT )
            bigAmpGlobalZ[ist].append( bigAmpZ0 )
            bigAmpGlobalR[ist].append( bigAmpR0 )
            bigAmpGlobalT[ist].append( bigAmpT0 )
            bigAmpTimeGlobalZ[ist].append( pickedPhaseTimeZ )
            bigAmpTimeGlobalR[ist].append( pickedPhaseTimeR )
            bigAmpTimeGlobalT[ist].append( pickedPhaseTimeT )
            peaksCurveGlobalZ[ist].append( peaksCurveZ )
            peaksCurveGlobalR[ist].append( peaksCurveR )
            peaksCurveGlobalT[ist].append( peaksCurveT )
            finalCcGlobalZ[ist].append( PickCorrZ_span )
            finalCcGlobalR[ist].append( PickCorrR_span )
            finalCcGlobalT[ist].append( PickCorrT_span )
            finalpeaksPtsGlobalZ[ist].append( peaksZ )
            finalpeaksPtsGlobalR[ist].append( peaksR )
            finalpeaksPtsGlobalT[ist].append( peaksT )
            phaShiftAngGlobalZ[ist].append( phaShiftAngZ )
            phaShiftAngGlobalR[ist].append( phaShiftAngR )
            phaShiftAngGlobalT[ist].append( phaShiftAngT )
            phaShiftAngTimeGlobalZ[ist].append( phaShiftAngTimeZ )
            phaShiftAngTimeGlobalR[ist].append( phaShiftAngTimeR )
            phaShiftAngTimeGlobalT[ist].append( phaShiftAngTimeT )
            corrLengGlobalZ[ist].append( corrLengZ )
            corrLengGlobalR[ist].append( corrLengR )
            corrLengGlobalT[ist].append( corrLengT )
            DTGlobalZ[ist].append( DT )
            DTGlobalR[ist].append( DT )
            DTGlobalT[ist].append( DT )
            temBegNorGlobalZ[ist].append( temZtBegNor )
            temBegNorGlobalR[ist].append( temRtBegNor ) 
            temBegNorGlobalT[ist].append( temTtBegNor ) 
            leftBoundry1GlobalZ[ist].append( leftBoundry1Z )
            leftBoundry1GlobalR[ist].append( leftBoundry1R )
            leftBoundry1GlobalT[ist].append( leftBoundry1T )
            rightBoundry1GlobalZ[ist].append( rightBoundry1Z )
            rightBoundry1GlobalR[ist].append( rightBoundry1R )
            rightBoundry1GlobalT[ist].append( rightBoundry1T )
            st_begin_timeNorGlobalZ[ist].append( st_begin_timeNor )
            st_begin_timeNorGlobalR[ist].append( st_begin_timeNor )
            st_begin_timeNorGlobalT[ist].append( st_begin_timeNor )
            wantedTimeLengGlobalZ[ist].append( wantedTimeLengZ )
            wantedTimeLengGlobalR[ist].append( wantedTimeLengZ )
            wantedTimeLengGlobalT[ist].append( wantedTimeLengZ )

    # 准备深度列表
    srcDepthScanBeg = scanDepthFrom
    srcDepthScanEnd = scanDepthTo
    srcDepthScanInc = 1
    depths = list(np.arange(srcDepthScanBeg, srcDepthScanEnd, srcDepthScanInc))

    # 构建参数列表
    args_list = []
    for tmp_depth in depths:
        args = (tmp_depth, velModel, recDisInDeg, recDepth, crustInterfaceDepths,
                arrTimeDiffTole,
                list(pickedPhaseTimeDiffZ) if 'pickedPhaseTimeDiffZ' in locals() else [],
                list(pickedPhaseTimeDiffR) if 'pickedPhaseTimeDiffR' in locals() else [],
                list(pickedPhaseTimeDiffT) if 'pickedPhaseTimeDiffT' in locals() else [],
                list(pickedPhaseTimeZ) if 'pickedPhaseTimeZ' in locals() else [],
                list(pickedPhaseTimeR) if 'pickedPhaseTimeR' in locals() else [],
                list(pickedPhaseTimeT) if 'pickedPhaseTimeT' in locals() else [])
        args_list.append(args)

    # 并行计算深度结果
    depth_results = []
    with ProcessPoolExecutor(max_workers=7) as executor:
        futures = [executor.submit(_calc_one_depth, args) for args in args_list]
        for future in futures:
            try:
                depth_results.append(future.result())
            except Exception as e:
                print(f"Error in parallel computation: {e}")
                # 默认结果（确保键齐全）
                depth_results.append({
                    'depth': args[0],
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
                })

    # 按深度排序
    depth_results.sort(key=lambda x: x['depth'])

    # 初始化局部数组（每个深度一个空列表，用于存储该深度的匹配信息）
    numScanDepth = len(depths)
    local_depthCandidateArrGlobalZ = [[] for _ in range(numScanDepth)]
    local_depthCandidateArrGlobalR = [[] for _ in range(numScanDepth)]
    local_depthCandidateArrGlobalT = [[] for _ in range(numScanDepth)]
    local_depthCandidateArrDiffGlobalZ = [[] for _ in range(numScanDepth)]
    local_depthCandidateArrDiffGlobalR = [[] for _ in range(numScanDepth)]
    local_depthCandidateArrDiffGlobalT = [[] for _ in range(numScanDepth)]
    local_depthCandidateArrTaupDiffGlobalZ = [[] for _ in range(numScanDepth)]
    local_depthCandidateArrTaupDiffGlobalR = [[] for _ in range(numScanDepth)]
    local_depthCandidateArrTaupDiffGlobalT = [[] for _ in range(numScanDepth)]
    local_depthCandidateArrdtGlobalZ = [[] for _ in range(numScanDepth)]
    local_depthCandidateArrdtGlobalR = [[] for _ in range(numScanDepth)]
    local_depthCandidateArrdtGlobalT = [[] for _ in range(numScanDepth)]
    local_depthCandidatePhaDigNameGlobalZ = [[] for _ in range(numScanDepth)]
    local_depthCandidatePhaDigNameGlobalR = [[] for _ in range(numScanDepth)]
    local_depthCandidatePhaDigNameGlobalT = [[] for _ in range(numScanDepth)]
    local_depthCandidatePhaOrgNameGlobalZ = [[] for _ in range(numScanDepth)]
    local_depthCandidatePhaOrgNameGlobalR = [[] for _ in range(numScanDepth)]
    local_depthCandidatePhaOrgNameGlobalT = [[] for _ in range(numScanDepth)]

    # 统计数组（每个深度一个数值）
    local_totNumMatPha = np.zeros(numScanDepth)
    local_totNumCalPha = np.zeros(numScanDepth)
    local_percNumMatPha = np.zeros(numScanDepth)
    local_avgArrTimeDiffResZ = np.full(numScanDepth, 9999.0)
    local_avgArrTimeDiffResR = np.full(numScanDepth, 9999.0)
    local_avgArrTimeDiffResT = np.full(numScanDepth, 9999.0)
    local_avgArrTimeDiffResSum = np.full(numScanDepth, 9999.0)

    # 遍历深度结果，填充局部数组和统计数组
    for idx, depth_res in enumerate(depth_results):
        # 填充全局数组（这些是函数外定义的，用于后续计算）
        if depth_res['has_valid_matches']:
            totNumCalPhaGlobal[ist][idx] = depth_res['totNumCalPha']
            totNumMatPhaGlobal[ist][idx] = depth_res['totNumMatPha']
            percNumMatPhaGlobal[ist][idx] = depth_res['percNumMatPha']

        if len(depth_res['matchedTaupTimeDiffZ']) >= 1:
            avgArrTimeDiffResEachStationZ[ist][idx] = depth_res['avgArrTimeDiffResZ']
        if len(depth_res['matchedTaupTimeDiffR']) >= 1:
            avgArrTimeDiffResEachStationR[ist][idx] = depth_res['avgArrTimeDiffResR']
        if len(depth_res['matchedTaupTimeDiffT']) >= 1:
            avgArrTimeDiffResEachStationT[ist][idx] = depth_res['avgArrTimeDiffResT']
        if depth_res['has_valid_matches']:
            avgArrTimeDiffResEachStationSum[ist][idx] = depth_res['avgArrTimeDiffResSum']

        # 填充局部统计数组
        if depth_res['has_valid_matches']:
            local_totNumMatPha[idx] = depth_res['totNumMatPha']
            local_totNumCalPha[idx] = depth_res['totNumCalPha']
            local_percNumMatPha[idx] = depth_res['percNumMatPha']
        if len(depth_res['matchedTaupTimeDiffZ']) >= 1:
            local_avgArrTimeDiffResZ[idx] = depth_res['avgArrTimeDiffResZ']
        if len(depth_res['matchedTaupTimeDiffR']) >= 1:
            local_avgArrTimeDiffResR[idx] = depth_res['avgArrTimeDiffResR']
        if len(depth_res['matchedTaupTimeDiffT']) >= 1:
            local_avgArrTimeDiffResT[idx] = depth_res['avgArrTimeDiffResT']
        if depth_res['has_valid_matches']:
            local_avgArrTimeDiffResSum[idx] = depth_res['avgArrTimeDiffResSum']

        # 填充匹配的相位时间列表（直接赋值，每个深度一个列表）
        local_depthCandidateArrGlobalZ[idx] = depth_res['matchedPickedTimeZ']
        local_depthCandidateArrGlobalR[idx] = depth_res['matchedPickedTimeR']
        local_depthCandidateArrGlobalT[idx] = depth_res['matchedPickedTimeT']

        local_depthCandidateArrDiffGlobalZ[idx] = depth_res['matchedPickedTimeDiffZ']
        local_depthCandidateArrDiffGlobalR[idx] = depth_res['matchedPickedTimeDiffR']
        local_depthCandidateArrDiffGlobalT[idx] = depth_res['matchedPickedTimeDiffT']

        local_depthCandidateArrTaupDiffGlobalZ[idx] = depth_res['matchedTaupTimeDiffZ']
        local_depthCandidateArrTaupDiffGlobalR[idx] = depth_res['matchedTaupTimeDiffR']
        local_depthCandidateArrTaupDiffGlobalT[idx] = depth_res['matchedTaupTimeDiffT']

        # 计算 dt 差值
        if depth_res['matchedPickedTimeDiffZ'] and depth_res['matchedTaupTimeDiffZ']:
            p_list = depth_res['matchedPickedTimeDiffZ']
            t_list = depth_res['matchedTaupTimeDiffZ']
            if not isinstance(p_list, (list, np.ndarray)):
                p_list = [p_list]
            if not isinstance(t_list, (list, np.ndarray)):
                t_list = [t_list]
            diff_arr = np.array(p_list) - np.array(t_list)
            diff_arr = np.atleast_1d(diff_arr)
            local_depthCandidateArrdtGlobalZ[idx] = diff_arr.tolist()
        else:
            local_depthCandidateArrdtGlobalZ[idx] = []

        if depth_res['matchedPickedTimeDiffR'] and depth_res['matchedTaupTimeDiffR']:
            p_list = depth_res['matchedPickedTimeDiffR']
            t_list = depth_res['matchedTaupTimeDiffR']
            if not isinstance(p_list, (list, np.ndarray)):
                p_list = [p_list]
            if not isinstance(t_list, (list, np.ndarray)):
                t_list = [t_list]
            diff_arr = np.array(p_list) - np.array(t_list)
            diff_arr = np.atleast_1d(diff_arr)
            local_depthCandidateArrdtGlobalR[idx] = diff_arr.tolist()
        else:
            local_depthCandidateArrdtGlobalR[idx] = []

        if depth_res['matchedPickedTimeDiffT'] and depth_res['matchedTaupTimeDiffT']:
            p_list = depth_res['matchedPickedTimeDiffT']
            t_list = depth_res['matchedTaupTimeDiffT']
            if not isinstance(p_list, (list, np.ndarray)):
                p_list = [p_list]
            if not isinstance(t_list, (list, np.ndarray)):
                t_list = [t_list]
            diff_arr = np.array(p_list) - np.array(t_list)
            diff_arr = np.atleast_1d(diff_arr)
            local_depthCandidateArrdtGlobalT[idx] = diff_arr.tolist()
        else:
            local_depthCandidateArrdtGlobalT[idx] = []

        local_depthCandidatePhaDigNameGlobalZ[idx] = depth_res['matchedTaupPhaseDigNameZ']
        local_depthCandidatePhaDigNameGlobalR[idx] = depth_res['matchedTaupPhaseDigNameR']
        local_depthCandidatePhaDigNameGlobalT[idx] = depth_res['matchedTaupPhaseDigNameT']

        local_depthCandidatePhaOrgNameGlobalZ[idx] = depth_res['matchedTaupPhaseOrgNameZ']
        local_depthCandidatePhaOrgNameGlobalR[idx] = depth_res['matchedTaupPhaseOrgNameR']
        local_depthCandidatePhaOrgNameGlobalT[idx] = depth_res['matchedTaupPhaseOrgNameT']

    # 找到当前台站匹配数最多的深度索引（用于后续步骤）
    idxMaxNumPha = np.argmax(totNumMatPhaGlobal[ist])
    depthMaxNumPhaEachStation = idxMaxNumPha * srcDepthScanInc + srcDepthScanBeg
    idxMaxNumPhaEachStation[ist] = idxMaxNumPha

    result = {
        'valid': True,
        'name': nameEachStation[ist],
        'azimuth': azimuthEachStation[ist],
        'epiDis': epiDisEachStation[ist],
        'onsetP': onsetPEachStation[ist],
        'onsetS': onsetSEachStation[ist],

        # 统计数组
        'totNumMatPha': local_totNumMatPha,
        'totNumCalPha': local_totNumCalPha,
        'percNumMatPha': local_percNumMatPha,
        'avgArrTimeDiffResZ': local_avgArrTimeDiffResZ,
        'avgArrTimeDiffResR': local_avgArrTimeDiffResR,
        'avgArrTimeDiffResT': local_avgArrTimeDiffResT,
        'avgArrTimeDiffResSum': local_avgArrTimeDiffResSum,

        # 深度相关列表（每个深度一个列表，没有多余维度）
        'depthCandidateArrGlobalZ': local_depthCandidateArrGlobalZ,
        'depthCandidateArrGlobalR': local_depthCandidateArrGlobalR,
        'depthCandidateArrGlobalT': local_depthCandidateArrGlobalT,
        'depthCandidateArrDiffGlobalZ': local_depthCandidateArrDiffGlobalZ,
        'depthCandidateArrDiffGlobalR': local_depthCandidateArrDiffGlobalR,
        'depthCandidateArrDiffGlobalT': local_depthCandidateArrDiffGlobalT,
        'depthCandidateArrTaupDiffGlobalZ': local_depthCandidateArrTaupDiffGlobalZ,
        'depthCandidateArrTaupDiffGlobalR': local_depthCandidateArrTaupDiffGlobalR,
        'depthCandidateArrTaupDiffGlobalT': local_depthCandidateArrTaupDiffGlobalT,
        'depthCandidateArrdtGlobalZ': local_depthCandidateArrdtGlobalZ,
        'depthCandidateArrdtGlobalR': local_depthCandidateArrdtGlobalR,
        'depthCandidateArrdtGlobalT': local_depthCandidateArrdtGlobalT,
        'depthCandidatePhaDigNameGlobalZ': local_depthCandidatePhaDigNameGlobalZ,
        'depthCandidatePhaDigNameGlobalR': local_depthCandidatePhaDigNameGlobalR,
        'depthCandidatePhaDigNameGlobalT': local_depthCandidatePhaDigNameGlobalT,
        'depthCandidatePhaOrgNameGlobalZ': local_depthCandidatePhaOrgNameGlobalZ,
        'depthCandidatePhaOrgNameGlobalR': local_depthCandidatePhaOrgNameGlobalR,
        'depthCandidatePhaOrgNameGlobalT': local_depthCandidatePhaOrgNameGlobalT,

        # 波形等数据（保持不变）
        'waveformZ': [stZ0[0].data],
        'waveformR': [stR0[0].data],
        'waveformT': [stT0[0].data],
        'normWaveformZ': [stNorZ],
        'normWaveformR': [stNorR],
        'normWaveformT': [stNorT],
        'template0Z': [templateZ[0].data],
        'template0R': [templateR[0].data],
        'template0T': [templateT[0].data],
        'template60Z': [templateZ60],
        'template60R': [templateR60],
        'template60T': [templateT60],
        'template120Z': [templateZ120],
        'template120R': [templateR120],
        'template120T': [templateT120],
        'template170Z': [templateZ170],
        'template170R': [templateR170],
        'template170T': [templateT170],
        'histogramZ': [histZ],
        'histogramR': [histR],
        'histogramT': [histT],
        'bigAmpZ': [bigAmpZ0],
        'bigAmpR': [bigAmpR0],
        'bigAmpT': [bigAmpT0],
        'bigAmpTimeZ': [pickedPhaseTimeZ],
        'bigAmpTimeR': [pickedPhaseTimeR],
        'bigAmpTimeT': [pickedPhaseTimeT],
        'peaksCurveZ': [peaksCurveZ],
        'peaksCurveR': [peaksCurveR],
        'peaksCurveT': [peaksCurveT],
        'finalpeaksPtsZ': [peaksZ],
        'finalpeaksPtsR': [peaksR],
        'finalpeaksPtsT': [peaksT],
        'finalCcZ': [PickCorrZ_span],
        'finalCcR': [PickCorrR_span],
        'finalCcT': [PickCorrT_span],
        'phaShiftAngZ': [phaShiftAngZ],
        'phaShiftAngR': [phaShiftAngR],
        'phaShiftAngT': [phaShiftAngT],
        'phaShiftAngTimeZ': [phaShiftAngTimeZ],
        'phaShiftAngTimeR': [phaShiftAngTimeR],
        'phaShiftAngTimeT': [phaShiftAngTimeT],
        'corrLengZ': [corrLengZ],
        'corrLengR': [corrLengR],
        'corrLengT': [corrLengT],
        'DTZ': [DT],
        'DTR': [DT],
        'DTT': [DT],
        'temBegNorZ': [temZtBegNor],
        'temBegNorR': [temRtBegNor],
        'temBegNorT': [temTtBegNor],
        'leftBoundry1Z': [leftBoundry1Z],
        'leftBoundry1R': [leftBoundry1R],
        'leftBoundry1T': [leftBoundry1T],
        'rightBoundry1Z': [rightBoundry1Z],
        'rightBoundry1R': [rightBoundry1R],
        'rightBoundry1T': [rightBoundry1T],
        'st_begin_timeNorZ': [st_begin_timeNor],
        'st_begin_timeNorR': [st_begin_timeNor],
        'st_begin_timeNorT': [st_begin_timeNor],
        'wantedTimeLengZ': [wantedTimeLengZ],
        'wantedTimeLengR': [wantedTimeLengZ],
        'wantedTimeLengT': [wantedTimeLengZ],

        # 台站信息
        'timediffR': Sta_timediffR[-1] if Sta_timediffR else None,
        'timediffT': Sta_timediffT[-1] if Sta_timediffT else None,
        'timediffZ': Sta_timediffZ[-1] if Sta_timediffZ else None,
        'distkm': recDisInKm,
        'stanm': stRawZ[0].stats.sac.kstnm.strip(),
        'stalat': stLa,
        'stalon': stLo,
    }
    print("DEBUG:", ist, type(result), result)

    print(f"DEBUG: station {ist} returning valid={result['valid']}")
    return ist, result



crustInterfaceDepths = GetCrustInterfaceDepths(velModel)
recDepth = 0                     # 台站深度（km）
srcDepthScanInc = 1               # 深度扫描步长
scanDepthMembers = list(np.arange(scanDepthFrom, scanDepthTo, srcDepthScanInc))




args_list = []
for ist in range(numSt):
    args = (ist, dataPath, wfFiles, velModel, arrTimeDiffTole,
            ccThreshold, frequencyFrom, frequencyTo, scanDepthFrom,
            scanDepthTo, verboseFlag, crustInterfaceDepths, 
            recDepth, srcDepthScanInc,numScanDepth)
    args_list.append(args)

# 并行执行所有台站
results = []

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=5) as executor:
        future_to_station = {
            executor.submit(process_one_station, *args): args[0]
            for args in args_list
        }

        for future in future_to_station:
            ist = future_to_station[future]
            try:
                _, res = future.result()
                results.append((ist, res))
            except Exception as e:
                print(f"台站 {ist} 处理出错: {e}")
                results.append((ist, None))

# 按台站索引排序（确保顺序）
results.sort(key=lambda x: x[0])
successful_ist = []

# 填充全局数组
for ist, res in results:
    try:

    
        if res is None or not res.get('valid', False):
            continue
        

        # 填充标量值
        successful_ist.append(ist)
        nameEachStation[ist] = res['name']
        azimuthEachStation[ist] = res['azimuth']
        epiDisEachStation[ist] = res['epiDis']
        onsetPEachStation[ist] = res['onsetP']
        onsetSEachStation[ist] = res['onsetS']

        depthCandidateArrGlobalZ[ist] = res['depthCandidateArrGlobalZ']
        depthCandidateArrGlobalR[ist] = res['depthCandidateArrGlobalR']
        depthCandidateArrGlobalT[ist] = res['depthCandidateArrGlobalT']

        depthCandidateArrDiffGlobalZ[ist] = res['depthCandidateArrDiffGlobalZ']
        depthCandidateArrDiffGlobalR[ist] = res['depthCandidateArrDiffGlobalR']
        depthCandidateArrDiffGlobalT[ist] = res['depthCandidateArrDiffGlobalT']

        # 填充一维数组
        totNumMatPhaGlobal[ist] = res['totNumMatPha']
        totNumCalPhaGlobal[ist] = res['totNumCalPha']
        percNumMatPhaGlobal[ist] = res['percNumMatPha']
        avgArrTimeDiffResEachStationZ[ist] = res['avgArrTimeDiffResZ']
        avgArrTimeDiffResEachStationR[ist] = res['avgArrTimeDiffResR']
        avgArrTimeDiffResEachStationT[ist] = res['avgArrTimeDiffResT']
        avgArrTimeDiffResEachStationSum[ist] = res['avgArrTimeDiffResSum']

        # 填充波形列表（每个元素是数组）
        waveformGlobalZ[ist] = res['waveformZ']
        waveformGlobalR[ist] = res['waveformR']
        waveformGlobalT[ist] = res['waveformT']
        normWaveformGlobalZ[ist] = res['normWaveformZ']
        normWaveformGlobalR[ist] = res['normWaveformR']
        normWaveformGlobalT[ist] = res['normWaveformT']
        template0GlobalZ[ist] = res['template0Z']
        template0GlobalR[ist] = res['template0R']
        template0GlobalT[ist] = res['template0T']
        template60GlobalZ[ist] = res['template60Z']
        template60GlobalR[ist] = res['template60R']
        template60GlobalT[ist] = res['template60T']
        template120GlobalZ[ist] = res['template120Z']
        template120GlobalR[ist] = res['template120R']
        template120GlobalT[ist] = res['template120T']
        template170GlobalZ[ist] = res['template170Z']
        template170GlobalR[ist] = res['template170R']
        template170GlobalT[ist] = res['template170T']
        histogramGlobalZ[ist] = res['histogramZ']
        histogramGlobalR[ist] = res['histogramR']
        histogramGlobalT[ist] = res['histogramT']
        bigAmpGlobalZ[ist] = res['bigAmpZ']
        bigAmpGlobalR[ist] = res['bigAmpR']
        bigAmpGlobalT[ist] = res['bigAmpT']
        bigAmpTimeGlobalZ[ist] = res['bigAmpTimeZ']
        bigAmpTimeGlobalR[ist] = res['bigAmpTimeR']
        bigAmpTimeGlobalT[ist] = res['bigAmpTimeT']
        peaksCurveGlobalZ[ist] = res['peaksCurveZ']
        peaksCurveGlobalR[ist] = res['peaksCurveR']
        peaksCurveGlobalT[ist] = res['peaksCurveT']
        finalpeaksPtsGlobalZ[ist] = res['finalpeaksPtsZ']
        finalpeaksPtsGlobalR[ist] = res['finalpeaksPtsR']
        finalpeaksPtsGlobalT[ist] = res['finalpeaksPtsT']
        finalCcGlobalZ[ist] = res['finalCcZ']
        finalCcGlobalR[ist] = res['finalCcR']
        finalCcGlobalT[ist] = res['finalCcT']
        phaShiftAngGlobalZ[ist] = res['phaShiftAngZ']
        phaShiftAngGlobalR[ist] = res['phaShiftAngR']
        phaShiftAngGlobalT[ist] = res['phaShiftAngT']
        phaShiftAngTimeGlobalZ[ist] = res['phaShiftAngTimeZ']
        phaShiftAngTimeGlobalR[ist] = res['phaShiftAngTimeR']
        phaShiftAngTimeGlobalT[ist] = res['phaShiftAngTimeT']
        corrLengGlobalZ[ist] = res['corrLengZ']
        corrLengGlobalR[ist] = res['corrLengR']
        corrLengGlobalT[ist] = res['corrLengT']
        DTGlobalZ[ist] = res['DTZ']
        DTGlobalR[ist] = res['DTR']
        DTGlobalT[ist] = res['DTT']
        temBegNorGlobalZ[ist] = res['temBegNorZ']
        temBegNorGlobalR[ist] = res['temBegNorR']
        temBegNorGlobalT[ist] = res['temBegNorT']
        leftBoundry1GlobalZ[ist] = res['leftBoundry1Z']
        leftBoundry1GlobalR[ist] = res['leftBoundry1R']
        leftBoundry1GlobalT[ist] = res['leftBoundry1T']
        rightBoundry1GlobalZ[ist] = res['rightBoundry1Z']
        rightBoundry1GlobalR[ist] = res['rightBoundry1R']
        rightBoundry1GlobalT[ist] = res['rightBoundry1T']
        st_begin_timeNorGlobalZ[ist] = res['st_begin_timeNorZ']
        st_begin_timeNorGlobalR[ist] = res['st_begin_timeNorR']
        st_begin_timeNorGlobalT[ist] = res['st_begin_timeNorT']
        wantedTimeLengGlobalZ[ist] = res['wantedTimeLengZ']
        wantedTimeLengGlobalR[ist] = res['wantedTimeLengR']
        wantedTimeLengGlobalT[ist] = res['wantedTimeLengT']
        depthCandidateArrGlobalZ[ist] = res['depthCandidateArrGlobalZ']
        depthCandidateArrGlobalR[ist] = res['depthCandidateArrGlobalR']
        depthCandidateArrGlobalT[ist] = res['depthCandidateArrGlobalT']
        depthCandidateArrDiffGlobalZ[ist] = res['depthCandidateArrDiffGlobalZ']
        depthCandidateArrDiffGlobalR[ist] = res['depthCandidateArrDiffGlobalR']
        depthCandidateArrDiffGlobalT[ist] = res['depthCandidateArrDiffGlobalT']
        depthCandidateArrTaupDiffGlobalZ[ist] = res['depthCandidateArrTaupDiffGlobalZ']
        depthCandidateArrTaupDiffGlobalR[ist] = res['depthCandidateArrTaupDiffGlobalR']
        depthCandidateArrTaupDiffGlobalT[ist] = res['depthCandidateArrTaupDiffGlobalT']
        depthCandidatePhaDigNameGlobalZ[ist] = res['depthCandidatePhaDigNameGlobalZ']
        depthCandidatePhaDigNameGlobalR[ist] = res['depthCandidatePhaDigNameGlobalR']
        depthCandidatePhaDigNameGlobalT[ist] = res['depthCandidatePhaDigNameGlobalT']
        depthCandidatePhaOrgNameGlobalZ[ist] = res['depthCandidatePhaOrgNameGlobalZ']
        depthCandidatePhaOrgNameGlobalR[ist] = res['depthCandidatePhaOrgNameGlobalR']
        depthCandidatePhaOrgNameGlobalT[ist] = res['depthCandidatePhaOrgNameGlobalT']
        depthCandidateArrdtGlobalZ[ist] = res['depthCandidateArrdtGlobalZ']
        depthCandidateArrdtGlobalR[ist] = res['depthCandidateArrdtGlobalR']
        depthCandidateArrdtGlobalT[ist] = res['depthCandidateArrdtGlobalT']

        # 填充 Sta_timediff 等列表（注意这些是全局列表，不是数组）
        if res['timediffR'] is not None:
            Sta_timediffR.append(res['timediffR'])
            Sta_timediffT.append(res['timediffT'])
            Sta_timediffZ.append(res['timediffZ'])
            Sta_distkm.append(res['distkm'])
            Sta_stanm.append(res['stanm'])
            Sta_stalat.append(res['stalat'])
            Sta_stalon.append(res['stalon'])

    except Exception as e:
        print(f"Error filling station {ist}: {e}")
        import traceback
        traceback.print_exc()


    
#%%######################################################
# Step 3: Preliminary solution                          #
#########################################################
#-- Calculate the total number of matched phases
sumGlobal = totNumMatPhaGlobal.sum(axis=0)
srcDepthScanBeg = scanDepthFrom
# sumGlobal = percNumMatPhaGlobal.sum(axis=0)
#-- sum the valid arrival time difference
for idepth in range( numScanDepth ):
    tmpCount = 0
    tmpSum   = 0.
    for ist in range( numSt ):
        if( avgArrTimeDiffResEachStationSum[ist][idepth] < 9999 ):
            tmpSum   += avgArrTimeDiffResEachStationSum[ist][idepth]
            tmpCount += 1
    if tmpCount > 0:
        sumAvgArrTimeDiffResGlobal[idepth] = tmpSum / tmpCount
   
   
#-- the depth exceeds the set threshold
thresholdMaxNumb = np.max(sumGlobal)*0.9

#- origin version of preliminary depth.
#- 1. rms of each station at each depth 2. averaged rms of all the stations at each depth
prelimCandidatesGlobal = []
for i in range( len(scanDepthMembers) ):
    if sumGlobal[i] >= thresholdMaxNumb:
        candidates = sumAvgArrTimeDiffResGlobal[i], scanDepthMembers[i]
        prelimCandidatesGlobal.append( candidates )

tmpDepthRange1 = []
tmpAvgArrTimeDiffResEachStationSum1 = []
for i in range( len(scanDepthMembers) ):
    if sumGlobal[i] >= thresholdMaxNumb:
        tmpAvgArrTimeDiffResEachStationSum1.append( sumAvgArrTimeDiffResGlobal[i] )
        tmpDepthRange1.append(i)
    else:
        tmpAvgArrTimeDiffResEachStationSum1.append( 9999 )
prelimSolution_0 = np.argmin( tmpAvgArrTimeDiffResEachStationSum1 ) * srcDepthScanInc + srcDepthScanBeg
print('prelimSolution_step3_1=', prelimSolution_0, 'km' )

## the maximum percentatge as the prelinimary resolved depth
# per_dep = []
# for i in range( len(scanDepthMembers) ):
#     tmp_matched_dt = 0
#     for j in range(numSt):
#         tmp_matched_dt += (percNumMatPhaGlobal[j][i])
#     per_dep.append(tmp_matched_dt/numSt)
# prelimSolution_0 = np.argmin( per_dep) * srcDepthScanInc + srcDepthScanBeg
# print('prelimSolution_step3_1=', prelimSolution_0, 'km' )

## I change the Step 3.1 by calculate the RMS using mean value of matches, instead of stations.
# tmpDepthRange = []
# tmpAvgArrTimeDiffResEachStationSum1= []
# for i in range( len(scanDepthMembers) ):
#     if sumGlobal[i] >= thresholdMaxNumb:
#
#         tmp_matched_dt = []
#         tmp_matched_dt.extend(list(chain.from_iterable((chain.from_iterable(depthCandidateArrdtGlobalR[:][i])))))
#         tmp_matched_dt.extend(list(chain.from_iterable((chain.from_iterable(depthCandidateArrdtGlobalT[:][i])))))
#         tmp_matched_dt.extend(list(chain.from_iterable((chain.from_iterable(depthCandidateArrdtGlobalZ[:][i])))))
#
#         sumAvgArrTimeDiffResGlobal1[i] = np.mean(np.fabs(tmp_matched_dt))
#
#         tmpAvgArrTimeDiffResEachStationSum1.append( sumAvgArrTimeDiffResGlobal1[i] )
#         tmpDepthRange.append(i)
#     else:
#         tmpAvgArrTimeDiffResEachStationSum1.append( 9999 )
# prelimSolution_0 = np.argmin( tmpAvgArrTimeDiffResEachStationSum1 ) * srcDepthScanInc + srcDepthScanBeg
# print('prelimSolution_step3_1=', prelimSolution_0, 'km' )

prelimCandidatesGlobal_2 = []
for i in range( len(scanDepthMembers) ):
    if sumGlobal[i] >= thresholdMaxNumb:
        candidates = sumAvgArrTimeDiffResGlobal_2[i], scanDepthMembers[i]
        prelimCandidatesGlobal_2.append( candidates )





# #-- add Step 3.2  for simple temperary use
# # for the depths have more matches than the criteria,
# # then choose the preferable depth as minimum (rms/matches)
# Idx_pre_min = np.argmin( tmpAvgArrTimeDiffResEachStationSum1 )
# thresholdMaxNumb_2 = sumGlobal[Idx_pre_min]
# tmpDepthRange_2 = []
# tmpAvgArrTimeDiffResEachStationSum_2 = []
# for i in range( len(scanDepthMembers) ):
#     if sumGlobal[i] >= thresholdMaxNumb_2:
#         tmpAvgArrTimeDiffResEachStationSum_2.append( sumAvgArrTimeDiffResGlobal[i]/sumGlobal[i] )
#         tmpDepthRange_2.append(i)
#     else:
#         tmpAvgArrTimeDiffResEachStationSum_2.append( 9999 )
# prelimSolution = np.argmin( tmpAvgArrTimeDiffResEachStationSum_2 ) * srcDepthScanInc + srcDepthScanBeg
# print('prelimSolution_step3_2=', prelimSolution, 'km' )



#-- add Step 3.2
# set the number of matches of  the preliminary depth as the criteria for min rms,
# for the depths have more matches than the criteria, using the same number of smallest rms,
# then choose the minimum rms choice.
Idx_pre_min = np.argmin( tmpAvgArrTimeDiffResEachStationSum1 )
thresholdMaxNumb_2 = sumGlobal[Idx_pre_min]
tmpDepthRange_2 = []
tmpAvgArrTimeDiffResEachStationSum_2= []
for i in range( len(scanDepthMembers) ):
    if sumGlobal[i] >= thresholdMaxNumb_2:

        tmp_matched_dt = []
        for j in range(numSt):
            print("Z:", depthCandidateArrdtGlobalZ[j][i])
            tmp_matched_dt.extend(depthCandidateArrdtGlobalZ[j][i])
            tmp_matched_dt.extend(depthCandidateArrdtGlobalR[j][i])
            tmp_matched_dt.extend(depthCandidateArrdtGlobalT[j][i])

        tmp_matched_dt_sort = sorted(np.fabs(tmp_matched_dt))

        sumAvgArrTimeDiffResGlobal_2[i] = sum(tmp_matched_dt_sort[:thresholdMaxNumb_2.astype(int)]) / thresholdMaxNumb_2

        tmpAvgArrTimeDiffResEachStationSum_2.append( sumAvgArrTimeDiffResGlobal_2[i] )
        tmpDepthRange_2.append(i)
    else:
        tmpAvgArrTimeDiffResEachStationSum_2.append( 9999 )
prelimSolution = np.argmin( tmpAvgArrTimeDiffResEachStationSum_2 ) * srcDepthScanInc + srcDepthScanBeg
print('prelimSolution_step3_2=', prelimSolution, 'km' )
# prelimSolution = prelimSolution_0

prelimCandidatesGlobal_2 = []
for i in range( len(scanDepthMembers) ):
    if sumGlobal[i] >= thresholdMaxNumb:
        candidates = sumAvgArrTimeDiffResGlobal_2[i], scanDepthMembers[i]
        prelimCandidatesGlobal_2.append( candidates )

#%% -- plotting for debugging
# set figure layout
fig = plt.figure( constrained_layout=True, figsize=(5,2.5))
fig.subplots_adjust(hspace=0.4)
fig.subplots_adjust(wspace=0.18)
gs0 = fig.add_gridspec(1, 1 )
gs00 = gs0[0].subgridspec(1,1)
ax1 = fig.add_subplot(gs00[0, 0])
#-- plot data
t = scanDepthMembers
ax1.plot(t, sumGlobal, color="black", linewidth=2., alpha=1)
ax11 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax11.scatter(t, sumAvgArrTimeDiffResGlobal, s=25, marker='s',
             facecolors='none', edgecolor='gray', zorder=100)
# ax11.scatter(prelimSolution_0, sumAvgArrTimeDiffResGlobal[int(prelimSolution)-1],
#              s=30, marker='s', facecolors='gray', edgecolor='gray', zorder=100)
ax11.scatter(prelimSolution_0, sumAvgArrTimeDiffResGlobal[int(prelimSolution)-int(scanDepthFrom)],
             s=30, marker='s', facecolors='gray', edgecolor='gray', zorder=100)

#-- plot the new step3 data


ax11.scatter(t, sumAvgArrTimeDiffResGlobal_2, s=25, marker='s',
             facecolors='none', edgecolor='blue', zorder=100)
ax11.scatter(prelimSolution, sumAvgArrTimeDiffResGlobal_2[int(prelimSolution)-int(scanDepthFrom)],
             s=30, marker='s', facecolors='blue', edgecolor='blue', zorder=100)

ax11.set_ylim( 0, 0.4 )
#-- plot the number of matched of each station
for ist in range(numSt):
    ax1.plot(t, totNumMatPhaGlobal[ist],  linewidth=0.45, color="grey")

ax1.set_title('Step 3', fontsize=12)
# set labels
ax1.set_ylabel('Number of matches', fontsize=12)
ax1.set_xlabel('Depth (km)', fontsize=12)
ax11.set_ylabel('Sum of differential\narrival time residuals (s)', color='blue', fontsize=12)
#set grid and threshold lines
ax1.grid(True, linestyle='--', linewidth=0.25)
ax1.axvline( prelimSolution, linewidth=1, color='blue', linestyle='--')
ax1.axhline( thresholdMaxNumb,linewidth=1, color='black', linestyle='--')   
#-- set axis
ax1.margins(x=0)
ax1.set_xticks( np.arange(scanDepthFrom,scanDepthTo, step=2) )
ax11.tick_params(axis='y', colors='blue')
   
#save
fig.tight_layout()
# fig.savefig( "{0}/step3_locSrc{1}km.png".format(
#                     outfilePath,
#                     format( prelimSolution, ".1f"),
#                     dpi=360 ) )
fig.savefig( "{0}/step3_locSrc{1}km.pdf".format(outfilePath,prelimSolution, ".1f"))
plt.show()

#-- the end of debugging

    
    
#%%######################################################
# Step 4: Final solution based on travel time residuals #
#########################################################
wellBehavedStationId = []
wellBehavedStationIdx = []
wellBehavedStationDepth = []


def _step4_one_station(isw):

    ist = wellBehavedStationId[isw]
    idx = wellBehavedStationIdx[isw]
    depthCandidate = wellBehavedStationDepth[isw]

    # ---- scanning range ----
    srcDepth_b = max(scanDepthFrom, depthCandidate - 1)
    srcDepth_e = min(scanDepthTo,   depthCandidate + 1)
    depth_candidates = np.arange(srcDepth_b, srcDepth_e, 0.1)

    # ---- depth parallel (original logic) ----
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = []                     
        for d in depth_candidates:
            futures.append(
                ex.submit(
                    compute_one_depth,
                    d,
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
                    arrTimeDiffTole
                )
            )

        for fu in futures:
            results.append(fu.result())

    # ---- collect ----
    rmsRTZ = [r["sumRmsRTZ"] for r in results]
    exactDepthRTZ = [r["srcDepth"] for r in results]
    match_count = [r["match_count"] for r in results]

    idx_min = int(np.argmin(rmsRTZ))

    return {
        "ist": ist,
        "idx": idx,
        "best_depth": exactDepthRTZ[idx_min],
        "best_rms": rmsRTZ[idx_min],
        "rmsRTZ": rmsRTZ,
        "exactDepthRTZ": exactDepthRTZ,
        "match_count": match_count,
        "idx_min": idx_min
    }


for ist in range( numSt ):
    # avoid out of scanning depth range
    # if (prelimSolution <=  scanDepthFrom + 1 or
    #     prelimSolution >=  scanDepthTo - 1 ):
    #     break

  
    tmpData = avgArrTimeDiffResEachStationSum[ist][ np.min(tmpDepthRange1):np.max(tmpDepthRange1)+1 ]
    tmpMinValIdx = np.argmin( tmpData ) + np.min(tmpDepthRange1)
    prelimSolutionSingleStation = tmpMinValIdx * srcDepthScanInc + srcDepthScanBeg
    
    if ( np.abs( prelimSolution - prelimSolutionSingleStation ) <= 1 ):
        if ( len(depthCandidatePhaDigNameGlobalZ[ist][tmpMinValIdx]) > 0 and
             len(depthCandidatePhaDigNameGlobalR[ist][tmpMinValIdx]) > 0 and
             len(depthCandidatePhaDigNameGlobalT[ist][tmpMinValIdx]) > 0 ):
            wellBehavedStationId.append( ist )
            wellBehavedStationIdx.append( tmpMinValIdx )
            wellBehavedStationDepth.append( prelimSolutionSingleStation )
            
 
print("wellBehavedStationId    =", wellBehavedStationId)
print("wellBehavedStationIdx   =", wellBehavedStationIdx)
print("wellBehavedStationDepth =", wellBehavedStationDepth)


#%%
if len( wellBehavedStationId ) < 1:
    print("\n\n\n")
    print("No good station for Step 4, the solution of Step 3 is the final focal depth!")
    print("\n\n\n")
    #-- write info of good stations
    with open( '{0}'.format( outPath ),  mode='a', newline='' ) as resultsFile:
        writer = csv.writer( resultsFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['{0}'.format( 9999 ),
                         '{0}'.format( 9999 ),
                         '{0}'.format( 9999 ),
                         '{0}'.format( 9999 ),
                         '{0}'.format( format( prelimSolution, ".2f" ) ),
                         '{0}'.format( 9999 ),
                         '{0}'.format( 9999 ),
                         '{0}'.format( 9999 ) ])
    
else:   
    numWellBehavedStation = len(wellBehavedStationId)
    rmsWellBehavedStation = []
    depthWellBehavedStation = []
    depthRangeWellBehavedStation= []

    if __name__ == "__main__":

        DEPTH_WORKERS   = 9
        STATION_WORKERS = min(numWellBehavedStation, os.cpu_count())

        rmsWellBehavedStation = []
        depthWellBehavedStation = []
        depthRangeWellBehavedStation = []

        with ProcessPoolExecutor(max_workers=3) as exe:

            futures = {
                exe.submit(_step4_one_station, i): i
                for i in range(numWellBehavedStation)
            }

            results_station = []

            for future in as_completed(futures):

                ist = futures[future]

                try:
                    res = future.result()
                    results_station.append(res)

                except Exception as e:
                    print("ERROR STATION:", ist)
                    raise
            print("TYPE:", type(results_station))
            print("VALUE:", results_station)
        # ---- serial write-back ----
        for res in results_station:

            ist = res["ist"]
            idx_min = res["idx_min"]
            idx = res["idx"]
            histZ = histogramGlobalZ[ist][0]
            histR = histogramGlobalR[ist][0]
            histT = histogramGlobalT[ist][0]

            Sta_dep_final[ist] = res["best_depth"]
            Sta_dt_final[ist]  = res["best_rms"]
            Sta_match[ist]     = res["match_count"][idx_min]

            rmsWellBehavedStation.append(res["rmsRTZ"])
            depthWellBehavedStation.append(res["best_depth"])
            depthRangeWellBehavedStation.append(res["exactDepthRTZ"])

            # ---- CSV output (serial, safe) ----
            with open(outPath, mode='a', newline='') as f:
                writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([
                    nameEachStation[ist][0],
                    azimuthEachStation[ist],
                    format(epiDisEachStation[ist], '.2f'),
                    list(totNumMatPhaGlobal[ist]),
                    res["match_count"],
                    format(res["best_depth"], '.2f'),
                    res["exactDepthRTZ"],
                    res["rmsRTZ"],
                    format(res["best_rms"], '.2f')
                ])



            #%% -- plot for debugging
            if plotSteps1n2Flag == 1:
                #%%####################################################################
                # plot wavefrom, templates, cc, depth-phase matches, phase-shifted angles 
                #######################################################################
                ##### Vertical component
                #-- prepare data for plotting
                DT       = DTGlobalZ[ist][0]
                tempZ0   = template0GlobalZ[ist][0] / max( np.fabs( template0GlobalZ[ist][0] )) # phase-shift = 0 deg
                tempZ60  = template60GlobalZ[ist][0] / max( np.fabs(template60GlobalZ[ist][0]))# phase-shift = 60 deg
                tempZ120 = template120GlobalZ[ist][0] / max( np.fabs(template120GlobalZ[ist][0]))# phase-shift = 120 deg
                tempZ170 = template170GlobalZ[ist][0] / max( np.fabs(template170GlobalZ[ist][0]))# phase-shift = 170 deg
                dataZ    = waveformGlobalZ[ist][0] / max( np.fabs( waveformGlobalZ[ist][0] ))
                tempZ00  = template0GlobalZ[ist][0] / max( np.fabs( waveformGlobalZ[ist][0] ))
                bigAmpZ  = bigAmpGlobalZ[ist][0] * normWaveformGlobalZ[ist][0]
                stNorZ   = normWaveformGlobalZ[ist][0]
                orgCcZ   = peaksCurveGlobalZ[ist][0]
                finalCcZ = finalCcGlobalZ[ist][0]
                peaksPts = finalpeaksPtsGlobalZ[ist][0]
                corrLengZ= corrLengGlobalZ[ist][0]
                bigAmpTime       = bigAmpTimeGlobalZ[ist][0]
                phaShiftAngZ     = phaShiftAngGlobalZ[ist][0]
                phaShiftAngTimeZ = phaShiftAngTimeGlobalZ[ist][0] 
                temZtBegNor      = temBegNorGlobalZ[ist][0]
                leftBoundry1Z    = leftBoundry1GlobalZ[ist][0]
                rightBoundry1Z   = rightBoundry1GlobalZ[ist][0]
                wantedTimeLengZ  = wantedTimeLengGlobalZ[ist][0]
                finalArrZ        = depthCandidateArrGlobalZ[ist][idx]
                finalPhaOrgNameZ = depthCandidatePhaOrgNameGlobalZ[ist][idx]
                
                # set figure layout
                fig = plt.figure( constrained_layout=True, figsize=(8,4))
                fig.subplots_adjust(hspace=0.5)
                fig.subplots_adjust(wspace=0.05)
                gs0 = fig.add_gridspec(1, 2, width_ratios=[8,1] )
                gs00 = gs0[1].subgridspec(6,1)
                gs01 = gs0[0].subgridspec(10,1)
                
                
                ax0 = fig.add_subplot(gs00[0:2, 0])
                ax1 = fig.add_subplot(gs00[2, 0])
                ax2 = fig.add_subplot(gs00[3, 0])
                ax3 = fig.add_subplot(gs00[4, 0])
                ax4 = fig.add_subplot(gs00[5, 0])
                ax5 = fig.add_subplot(gs01[0:3, 0:])
                ax6 = fig.add_subplot(gs01[3:6, 0:])
                ax7 = fig.add_subplot(gs01[6, 0:])
                ax8 = fig.add_subplot(gs01[7:10, 0:])
                
                # plot data
                t1Z = np.arange( 0, len(tempZ0), 1)*DT
                t2Z = np.arange( 0, len(dataZ),  1)*DT
                tZcc= np.arange( 0, corrLengZ,   1)*DT
                
                #%%
                ax0.hist( histZ, density=True, bins=11, orientation='horizontal')
                ax1.plot( t1Z, tempZ0, color='orange')
                ax2.plot( t1Z, tempZ60)
                ax3.plot( t1Z, tempZ120 )
                ax4.plot( t1Z, tempZ170 )
                ax5.plot( t2Z, dataZ ,linewidth=0.7)
                ax6.plot( t2Z, stNorZ, color='lightgray' ,linewidth=0.7)
                ax6.plot( t2Z, bigAmpZ,linewidth=0.7 )
                ax7.plot( tZcc, orgCcZ, color='lightgray',linewidth=0.7)
                ax7.plot( tZcc, finalCcZ,linewidth=0.7 )
                
                # add templates
                tTempZ = np.arange( 0, len(tempZ0), 1)* DT+temZtBegNor
                ax5.plot( tTempZ, tempZ00, color='orange',linewidth=1 )
                # add texts
                ax0.text( 2,   -0.4, r'$\mu-\sigma$', fontsize=10, rotation=0 )
                ax0.text( 2,    0.3, r'$\mu+\sigma$', fontsize=10, rotation=0 )
                ax1.text( 0.01, 0.45, '0 °', fontsize=10, color='black')
                ax2.text( 0.01, 0.45, '60 °', fontsize=10, color='black')
                ax3.text( 0.01, 0.45, '120 °', fontsize=10, color='black')
                ax4.text( 0.01, 0.45, '170 °', fontsize=10, color='black')
                
                # add matched phase and arrival time
                # find the phases sharing same arrival
                uniFinalArrZ = list(set(finalArrZ))
                for i in range( len( uniFinalArrZ ) ):
                    phaNameForPlot = []
                    count = 1
                    for j in range( len( finalPhaOrgNameZ ) ):
                        if finalArrZ[j] == uniFinalArrZ[i]:
                            if (len(phaNameForPlot)) > 0:
                                phaNameForPlot.append( "{0}{1}".format( finalPhaOrgNameZ[j], count ) )
                                if finalPhaOrgNameZ[j] == "s":
                                    phaNameForPlot[-1] = "s"
                                count += 1
                            else:
                                phaNameForPlot.append( "{0}".format( finalPhaOrgNameZ[j] ) )
                        
                    nameAmpOffset = -0.7
                    ax6.axvline( uniFinalArrZ[i], ymin=0, ymax=0.5, linewidth=1, color='black', linestyle='--')
                    for k in range( len( phaNameForPlot ) ):
                        if k>=0 and k < len( phaNameForPlot ) -1:
                            ax6.text( uniFinalArrZ[i]-0.3, nameAmpOffset+0.5*k, "{0} + ".format( phaNameForPlot[k] ),
                                        fontsize=11, color='black', rotation=90)
                        else:
                            ax6.text( uniFinalArrZ[i]-0.3, nameAmpOffset+0.5*k, "{0}".format( phaNameForPlot[k] ),
                                        fontsize=11, color='black', rotation=90)
                        
                ax7.plot( bigAmpTime, finalCcZ[ peaksPts ], "o",
                        color='black', markersize=4, zorder=101)
                
                # plot phase-shifting angles
                ax8.plot( tZcc, finalCcZ, alpha=0 ) # just use its time axis
                ax8.scatter( phaShiftAngTimeZ, phaShiftAngZ,
                            s=20, color='black', zorder=101 )
                
                #plot CC and phase-shifting angle of matched phases 
                for i in range( len( uniFinalArrZ ) ):
                    for j in range( len( phaShiftAngZ ) ):
                        if uniFinalArrZ[i] == phaShiftAngTimeZ[j]:
                                ax8.scatter( uniFinalArrZ[i], phaShiftAngZ[j],
                                            s=20, color='red', zorder=101 )
                    for j in range( len( peaksPts ) ):
                        if uniFinalArrZ[i] == bigAmpTime[j]:
                                ax7.plot( uniFinalArrZ[i], finalCcZ[ peaksPts[j] ], "o",
                                        color='red',markersize=4, zorder=101)
                #set grid
                ax8.grid(True, linestyle='--', linewidth=0.5)
                
                # set title
                ax1.tick_params(axis='both', which='major', labelsize=10)
                #
                ax0.set_xscale('log')
                
                #set lim
                ax0.set_xlim(1e0, 1e3)
                ax0.set_ylim(-1.1, 1.1)
                ax1.set_ylim(-1.2, 1.2)
                ax2.set_ylim(-1.2, 1.2)
                ax3.set_ylim(-1.2, 1.2)
                ax4.set_ylim(-1.2, 1.2)
                ax5.set_ylim(-1.2, 1.2)
                ax6.set_ylim(-1.3, 1.1)
                ax7.set_ylim(0, 1.2)
                ax8.set_ylim(-200, 200)
                ax8.set_yticks(np.arange(-180, 190, step=60))
                
                # set xticks
                ax0.xaxis.set_ticks_position('top')
                ax5.xaxis.set_ticks_position('top')
                ax6.xaxis.set_ticks_position('top')
                ax5.yaxis.set_ticks_position('left')
                ax6.yaxis.set_ticks_position('left')
                ax7.yaxis.set_ticks_position('left')
                ax8.yaxis.set_ticks_position('left')
                
                ax1.set_xticks([])
                ax2.set_xticks([])
                ax3.set_xticks([])
                ax4.set_xticks([])
                ax6.set_xticklabels([])
                ax7.set_xticks([])
                ax8.set_xticks([])
                
                ax0.set_yticks([])
                ax1.set_yticks([])
                ax2.set_yticks([])
                ax3.set_yticks([])
                ax4.set_yticks([])        
                
                #remove axis margins
                ax1.margins(x=0)
                ax2.margins(x=0)
                ax3.margins(x=0)
                ax4.margins(x=0)
                ax5.margins(x=0)
                ax6.margins(x=0)
                ax7.margins(x=0)
                ax8.margins(x=0)
                
                # remove some spines
                ax6.spines['bottom'].set_visible(False)
                ax7.spines['top'].set_visible(False)
                
                # set labels
                ax0.xaxis.set_label_position('top')
                ax1.xaxis.set_label_position('top')
                ax5.xaxis.set_label_position('top')
                ax5.yaxis.set_label_position('left')
                ax6.yaxis.set_label_position('left')
                ax7.yaxis.set_label_position('left')
                ax8.yaxis.set_label_position('left')
                
                ax0.set_xlabel('Number', fontsize=12, labelpad=6)
                ax5.set_xlabel('Time (s)', fontsize=12, labelpad=8)
                ax5.set_ylabel('Amp.', fontsize=12)
                ax6.set_ylabel('Amp.', fontsize=12)
                ax7.set_ylabel('CC', fontsize=12, labelpad=15)
                ax8.set_ylabel('Shifted (°)', fontsize=12)
                
                # set zero lines
                ax7.axhline(ccThreshold, linewidth=0.8, linestyle='--', color='gray')
                
                # set span
                ax0.axhspan(leftBoundry1Z, rightBoundry1Z, facecolor='0.5', alpha=0.2, color='black')
                ax5.axhspan(leftBoundry1Z, rightBoundry1Z, facecolor='0.5', alpha=0.1, color='black')
                    
                # plot figure number
                ax5.set_title( "a)", x=-0.1, fontsize=16, color='black', loc='left' )
                ax5.text( 0.45, 0.7, "Z", fontsize=12, color='black' )
                
                #show
                plt.tight_layout()
                # plt.savefig( "{0}/{1}_First2Steps_Z.png".format( outfilePath, nameEachStation[ist][0] ), dpi=360 )
                plt.savefig("{0}/{1}_First2Steps_Z.pdf".format(outfilePath, nameEachStation[ist][0]))
                #plt.savefig( "{0}/{1}_First2Steps_Z.svg".format( outfilePath, nameEachStation[ist][0] ), dpi=360 )
                plt.show    
        
                #%%####################################################################
                # plot waveform, templates, cc, depth-phase matches, phase-shifted angles
                #######################################################################
                ##### Radial component
                #-- prepare data for plotting
                DT       = DTGlobalR[ist][0]
                tempR0   = template0GlobalR[ist][0] / max( np.fabs( template0GlobalR[ist][0] )) # phase-shift = 0 deg
                tempR60  = template60GlobalR[ist][0] / max( np.fabs(template60GlobalR[ist][0]))# phase-shift = 60 deg
                tempR120 = template120GlobalR[ist][0] / max( np.fabs(template120GlobalR[ist][0]))# phase-shift = 120 deg
                tempR170 = template170GlobalR[ist][0] / max( np.fabs(template170GlobalR[ist][0]))# phase-shift = 180 deg
                dataR    = waveformGlobalR[ist][0] / max( np.fabs( waveformGlobalR[ist][0] ))
                tempR00  = template0GlobalR[ist][0] / max( np.fabs( waveformGlobalR[ist][0] ))
                bigAmpR  = bigAmpGlobalR[ist][0] * normWaveformGlobalR[ist][0]
                stNorR   = normWaveformGlobalR[ist][0]
                orgCcR   = peaksCurveGlobalR[ist][0]
                finalCcR = finalCcGlobalR[ist][0]
                peaksPts = finalpeaksPtsGlobalR[ist][0]
                corrLengR= corrLengGlobalR[ist][0]
                bigAmpTime       = bigAmpTimeGlobalR[ist][0]
                phaShiftAngR     = phaShiftAngGlobalR[ist][0]
                phaShiftAngTimeR = phaShiftAngTimeGlobalR[ist][0] 
                temRtBegNor      = temBegNorGlobalR[ist][0]
                leftBoundry1R    = leftBoundry1GlobalR[ist][0]
                rightBoundry1R   = rightBoundry1GlobalR[ist][0]
                wantedTimeLengR  = wantedTimeLengGlobalR[ist][0]
                finalArrR        = depthCandidateArrGlobalR[ist][idx]
                finalPhaOrgNameR = depthCandidatePhaOrgNameGlobalR[ist][idx]
                
                # set figure layout
                fig = plt.figure( constrained_layout=True, figsize=(8,4))
                fig.subplots_adjust(hspace=0.5)
                fig.subplots_adjust(wspace=0.05)
                gs0 = fig.add_gridspec(1, 2, width_ratios=[8,1] )
                gs00 = gs0[1].subgridspec(6,1)
                gs01 = gs0[0].subgridspec(10,1)
                
                
                ax0 = fig.add_subplot(gs00[0:2, 0])
                ax1 = fig.add_subplot(gs00[2, 0])
                ax2 = fig.add_subplot(gs00[3, 0])
                ax3 = fig.add_subplot(gs00[4, 0])
                ax4 = fig.add_subplot(gs00[5, 0])
                ax5 = fig.add_subplot(gs01[0:3, 0:])
                ax6 = fig.add_subplot(gs01[3:6, 0:])
                ax7 = fig.add_subplot(gs01[6, 0:])
                ax8 = fig.add_subplot(gs01[7:10, 0:])
                
                # plot data
                t1R = np.arange( 0, len(tempR0), 1)*DT
                t2R = np.arange( 0, len(dataR),  1)*DT
                tRcc= np.arange( 0, corrLengR,   1)*DT
                
                ax0.hist( histR, density=True, bins=11, orientation='horizontal')
                ax1.plot( t1R, tempR0, color='orange' )
                ax2.plot( t1R, tempR60 )
                ax3.plot( t1R, tempR120 )
                ax4.plot( t1R, tempR170 )
                ax5.plot( t2R, dataR ,linewidth=0.7)
                ax6.plot( t2R, stNorR, color='lightgray' ,linewidth=0.7)
                ax6.plot( t2R, bigAmpR ,linewidth=0.7)
                ax7.plot( tRcc, orgCcR, color='lightgray',linewidth=0.7)
                ax7.plot( tRcc, finalCcR ,linewidth=0.7)
                
                # add templates
                tTempR = np.arange( 0, len(tempR0), 1)* DT+temRtBegNor
                ax5.plot( tTempR, tempR00, color='orange',linewidth=1 )
                # add texts
                ax0.text( 2,   -0.4, r'$\mu-\sigma$', fontsize=10, rotation=0 )
                ax0.text( 2,    0.3, r'$\mu+\sigma$', fontsize=10, rotation=0 )
                ax1.text( 0.01, 0.45, '0 °', fontsize=10, color='black')
                ax2.text( 0.01, 0.45, '60 °', fontsize=10, color='black')
                ax3.text( 0.01, 0.45, '120 °', fontsize=10, color='black')
                ax4.text( 0.01, 0.45, '170 °', fontsize=10, color='black')
                
                # add matched phase and arrival time
                # find the phases sharing same arrival
                uniFinalArrR = list(set(finalArrR))
                for i in range( len( uniFinalArrR ) ):
                    phaNameForPlot = []
                    count = 1
                    for j in range( len( finalPhaOrgNameR ) ):
                        if finalArrR[j] == uniFinalArrR[i]:
                            if (len(phaNameForPlot)) > 0:
                                phaNameForPlot.append( "{0}{1}".format( finalPhaOrgNameR[j], count ) )
                                if finalPhaOrgNameR[j] == "s":
                                    phaNameForPlot[-1] = "s"
                                count += 1
                            else:
                                phaNameForPlot.append( "{0}".format( finalPhaOrgNameR[j] ) )
                        
                    nameAmpOffset = -0.7
                    ax6.axvline( uniFinalArrR[i], ymin=0, ymax=0.5, linewidth=1, color='black', linestyle='--')
                    for k in range( len( phaNameForPlot ) ):
                        if k>=0 and k < len( phaNameForPlot ) -1:
                            ax6.text( uniFinalArrR[i]-0.3, nameAmpOffset+0.5*k, "{0} + ".format( phaNameForPlot[k] ),
                                        fontsize=11, color='black', rotation=90)
                        else:
                            ax6.text( uniFinalArrR[i]-0.3, nameAmpOffset+0.5*k, "{0}".format( phaNameForPlot[k] ),
                                        fontsize=11, color='black', rotation=90)
                        
                ax7.plot( bigAmpTime, finalCcR[ peaksPts ], "o",
                        color='black', markersize=4, zorder=101)
                
                # plot phase-shifting angles
                ax8.plot( tRcc, finalCcR, alpha=0 ) # just use its time axis
                ax8.scatter( phaShiftAngTimeR, phaShiftAngR,
                            s=20, color='black', zorder=101 )
                
                #plot CC and phase-shifting angle of matched phases 
                for i in range( len( uniFinalArrR ) ):
                    for j in range( len( phaShiftAngR ) ):
                        if uniFinalArrR[i] == phaShiftAngTimeR[j]:
                                ax8.scatter( uniFinalArrR[i], phaShiftAngR[j],
                                            s=20, color='red', zorder=101 )
                    for j in range( len( peaksPts ) ):
                        if uniFinalArrR[i] == bigAmpTime[j]:
                                ax7.plot( uniFinalArrR[i], finalCcR[ peaksPts[j] ], "o",
                                        color='red',markersize=4, zorder=101)
                #set grid
                ax8.grid(True, linestyle='--', linewidth=0.5)
                
                # set title
                ax1.tick_params(axis='both', which='major', labelsize=10)
                #
                ax0.set_xscale('log')
                
                #set lim
                ax0.set_xlim(1e0, 1e3)
                ax0.set_ylim(-1.1, 1.1)
                ax1.set_ylim(-1.2, 1.2)
                ax2.set_ylim(-1.2, 1.2)
                ax3.set_ylim(-1.2, 1.2)
                ax4.set_ylim(-1.2, 1.2)
                ax5.set_ylim(-1.2, 1.2)
                ax6.set_ylim(-1.3, 1.1)
                ax7.set_ylim(0, 1.2)
                ax8.set_ylim(-200, 200)
                ax8.set_yticks(np.arange(-180, 190, step=60))
                
                # set xticks
                ax0.xaxis.set_ticks_position('top')
                ax5.xaxis.set_ticks_position('top')
                ax6.xaxis.set_ticks_position('top')
                ax5.yaxis.set_ticks_position('left')
                ax6.yaxis.set_ticks_position('left')
                ax7.yaxis.set_ticks_position('left')
                ax8.yaxis.set_ticks_position('left')
                
                ax1.set_xticks([])
                ax2.set_xticks([])
                ax3.set_xticks([])
                ax4.set_xticks([])
                ax6.set_xticklabels([])
                ax7.set_xticks([])
                ax8.set_xticks([])
                
                ax0.set_yticks([])
                ax1.set_yticks([])
                ax2.set_yticks([])
                ax3.set_yticks([])
                ax4.set_yticks([])        
                
                #remove axis margins
                ax1.margins(x=0)
                ax2.margins(x=0)
                ax3.margins(x=0)
                ax4.margins(x=0)
                ax5.margins(x=0)
                ax6.margins(x=0)
                ax7.margins(x=0)
                ax8.margins(x=0)
                
                # remove some spines
                ax6.spines['bottom'].set_visible(False)
                ax7.spines['top'].set_visible(False)
                
                # set labels
                ax0.xaxis.set_label_position('top')
                ax1.xaxis.set_label_position('top')
                ax5.xaxis.set_label_position('top')
                ax5.yaxis.set_label_position('left')
                ax6.yaxis.set_label_position('left')
                ax7.yaxis.set_label_position('left')
                ax8.yaxis.set_label_position('left')
                
                ax0.set_xlabel('Number', fontsize=12, labelpad=6)
                ax5.set_xlabel('Time (s)', fontsize=12, labelpad=8)
                ax5.set_ylabel('Amp.', fontsize=12)
                ax6.set_ylabel('Amp.', fontsize=12)
                ax7.set_ylabel('CC', fontsize=12, labelpad=15)
                ax8.set_ylabel('Shifted (°)', fontsize=12)
                
                # set zero lines
                ax7.axhline(ccThreshold, linewidth=0.8, linestyle='--', color='gray')
                
                # set span
                ax0.axhspan(leftBoundry1R, rightBoundry1R, facecolor='0.5', alpha=0.2, color='black')
                ax5.axhspan(leftBoundry1R, rightBoundry1R, facecolor='0.5', alpha=0.1, color='black')
                    
                # plot figure number
                ax5.set_title( "b)", x=-0.1, fontsize=16, color='black', loc='left' )
                ax5.text( 0.45, 0.7, "R", fontsize=12, color='black' )
                
                #show
                plt.tight_layout()
                # plt.savefig( "{0}/{1}_First2Steps_R.png".format( outfilePath, nameEachStation[ist][0] ), dpi=360 )
                #plt.savefig( "{0}/{1}_First2Steps_R.svg".format( outfilePath, nameEachStation[ist][0] ), dpi=360 )
                plt.savefig("{0}/{1}_First2Steps_R.pdf".format(outfilePath, nameEachStation[ist][0]))
                plt.show  
        
                #%%####################################################################
                # plot wavefrom, templates, cc, depth-phase matches, phase-shifted angles 
                #######################################################################
                ##### Transverse component
                #-- prepare data for plotting
                DT       = DTGlobalT[ist][0]
                tempT0   = template0GlobalT[ist][0] / max( np.fabs( template0GlobalT[ist][0] )) # phase-shift = 0 deg
                tempT60  = template60GlobalT[ist][0] / max( np.fabs(template60GlobalT[ist][0]))# phase-shift = 60 deg
                tempT120 = template120GlobalT[ist][0] / max( np.fabs(template120GlobalT[ist][0]))# phase-shift = 120 deg
                tempT170 = template170GlobalT[ist][0] / max( np.fabs(template170GlobalT[ist][0]))# phase-shift = 180 deg
                dataT    = waveformGlobalT[ist][0] / max( np.fabs( waveformGlobalT[ist][0] ))
                tempT00  = template0GlobalT[ist][0] / max( np.fabs( waveformGlobalT[ist][0] ))
                bigAmpT  = bigAmpGlobalT[ist][0] * normWaveformGlobalT[ist][0]
                stNorT   = normWaveformGlobalT[ist][0]
                orgCcT   = peaksCurveGlobalT[ist][0]
                finalCcT = finalCcGlobalT[ist][0]
                peaksPts = finalpeaksPtsGlobalT[ist][0]
                corrLengT= corrLengGlobalT[ist][0]
                bigAmpTime       = bigAmpTimeGlobalT[ist][0]
                phaShiftAngT     = phaShiftAngGlobalT[ist][0]
                phaShiftAngTimeT = phaShiftAngTimeGlobalT[ist][0] 
                temTtBegNor      = temBegNorGlobalT[ist][0]
                leftBoundry1T    = leftBoundry1GlobalT[ist][0]
                rightBoundry1T   = rightBoundry1GlobalT[ist][0]
                wantedTimeLengT  = wantedTimeLengGlobalT[ist][0]
                finalArrT        = depthCandidateArrGlobalT[ist][idx]
                finalPhaOrgNameT = depthCandidatePhaOrgNameGlobalT[ist][idx]
                
                # set figure layout
                fig = plt.figure( constrained_layout=True, figsize=(8,4))
                fig.subplots_adjust(hspace=0.5)
                fig.subplots_adjust(wspace=0.05)
                gs0 = fig.add_gridspec(1, 2, width_ratios=[8,1] )
                gs00 = gs0[1].subgridspec(6,1)
                gs01 = gs0[0].subgridspec(10,1)
                
                
                ax0 = fig.add_subplot(gs00[0:2, 0])
                ax1 = fig.add_subplot(gs00[2, 0])
                ax2 = fig.add_subplot(gs00[3, 0])
                ax3 = fig.add_subplot(gs00[4, 0])
                ax4 = fig.add_subplot(gs00[5, 0])
                ax5 = fig.add_subplot(gs01[0:3, 0:])
                ax6 = fig.add_subplot(gs01[3:6, 0:])
                ax7 = fig.add_subplot(gs01[6, 0:])
                ax8 = fig.add_subplot(gs01[7:10, 0:])
                
                # plot data
                t1T = np.arange( 0, len(tempT0), 1)*DT
                t2T = np.arange( 0, len(dataT),  1)*DT
                tTcc= np.arange( 0, corrLengT,   1)*DT
                
                ax0.hist( histT, density=True, bins=11, orientation='horizontal')
                ax1.plot( t1T, tempT0, color='orange' )
                ax2.plot( t1T, tempT60 )
                ax3.plot( t1T, tempT120 )
                ax4.plot( t1T, tempT170 )
                ax5.plot( t2T, dataT ,linewidth=0.7)
                ax6.plot( t2T, stNorT, color='lightgray' ,linewidth=0.7)
                ax6.plot( t2T, bigAmpT ,linewidth=0.7)
                ax7.plot( tTcc, orgCcT, color='lightgray',linewidth=0.7)
                ax7.plot( tTcc, finalCcT ,linewidth=0.7)
                
                # add templates
                tTempT = np.arange( 0, len(tempT0), 1)*DT+temTtBegNor
                ax5.plot( tTempT, tempT00, color='orange' ,linewidth=1)
                # add texts
                ax0.text( 2,   -0.4, r'$\mu-\sigma$', fontsize=10, rotation=0 )
                ax0.text( 2,    0.3, r'$\mu+\sigma$', fontsize=10, rotation=0 )
                ax1.text( 0.01, 0.45, '0 °', fontsize=10, color='black')
                ax2.text( 0.01, 0.45, '60 °', fontsize=10, color='black')
                ax3.text( 0.01, 0.45, '120 °', fontsize=10, color='black')
                ax4.text( 0.01, 0.45, '170 °', fontsize=10, color='black')
                
                # add matched phase and arrival time
                # find the phases sharing same arrival
                uniFinalArrT = list(set(finalArrT))
                for i in range( len( uniFinalArrT ) ):
                    phaNameForPlot = []
                    count = 1
                    for j in range( len( finalPhaOrgNameT ) ):
                        if finalArrT[j] == uniFinalArrT[i]:
                            if (len(phaNameForPlot)) > 0:
                                phaNameForPlot.append( "{0}{1}".format( finalPhaOrgNameT[j], count ) )
                                if finalPhaOrgNameT[j] == "s":
                                    phaNameForPlot[-1] = "s"
                                count += 1
                            else:
                                phaNameForPlot.append( "{0}".format( finalPhaOrgNameT[j] ) )
                        
                    nameAmpOffset = -0.7
                    ax6.axvline( uniFinalArrT[i], ymin=0, ymax=0.5, linewidth=1, color='black', linestyle='--')
                    for k in range( len( phaNameForPlot ) ):
                        if k>=0 and k < len( phaNameForPlot ) -1:
                            ax6.text( uniFinalArrT[i]-0.3, nameAmpOffset+0.5*k, "{0} + ".format( phaNameForPlot[k] ),
                                        fontsize=11, color='black', rotation=90)
                        else:
                            ax6.text( uniFinalArrT[i]-0.3, nameAmpOffset+0.5*k, "{0}".format( phaNameForPlot[k] ),
                                        fontsize=11, color='black', rotation=90)
                        
                ax7.plot( bigAmpTime, finalCcT[ peaksPts ],
                        "o", color='black', markersize=4, zorder=101)
                
                # plot phase-shifting angles
                ax8.plot( tTcc, finalCcT, alpha=0 ) # just use its time axis
                ax8.scatter( phaShiftAngTimeT, phaShiftAngT,
                            s=20, color='black', zorder=101 )
                
                #plot CC and phase-shifting angle of matched phases 
                for i in range( len( uniFinalArrT ) ):
                    for j in range( len( phaShiftAngT ) ):
                        if uniFinalArrT[i] == phaShiftAngTimeT[j]:
                                ax8.scatter( uniFinalArrT[i], phaShiftAngT[j],
                                            s=20, color='red', zorder=101 )
                    for j in range( len( peaksPts ) ):
                        if uniFinalArrT[i] == bigAmpTime[j]:
                                ax7.plot( uniFinalArrT[i], finalCcT[ peaksPts[j] ], "o",
                                        color='red',markersize=4, zorder=101)
                #set grid
                ax8.grid(True, linestyle='--', linewidth=0.5)
                
                # set title
                ax1.tick_params(axis='both', which='major', labelsize=10)
                #
                ax0.set_xscale('log')
                
                #set lim
                ax0.set_xlim(1e0, 1e3)
                ax0.set_ylim(-1.1, 1.1)
                ax1.set_ylim(-1.2, 1.2)
                ax2.set_ylim(-1.2, 1.2)
                ax3.set_ylim(-1.2, 1.2)
                ax4.set_ylim(-1.2, 1.2)
                ax5.set_ylim(-1.2, 1.2)
                ax6.set_ylim(-1.3, 1.1)
                ax7.set_ylim(0, 1.2)
                ax8.set_ylim(-200, 200)
                ax8.set_yticks(np.arange(-180, 190, step=60))
                
                # set xticks
                ax0.xaxis.set_ticks_position('top')
                ax5.xaxis.set_ticks_position('top')
                ax6.xaxis.set_ticks_position('top')
                ax5.yaxis.set_ticks_position('left')
                ax6.yaxis.set_ticks_position('left')
                ax7.yaxis.set_ticks_position('left')
                ax8.yaxis.set_ticks_position('left')
                
                ax1.set_xticks([])
                ax2.set_xticks([])
                ax3.set_xticks([])
                ax4.set_xticks([])
                ax6.set_xticklabels([])
                ax7.set_xticks([])
                ax8.set_xticks([])
                
                ax0.set_yticks([])
                ax1.set_yticks([])
                ax2.set_yticks([])
                ax3.set_yticks([])
                ax4.set_yticks([])        
                
                #remove axis margins
                ax1.margins(x=0)
                ax2.margins(x=0)
                ax3.margins(x=0)
                ax4.margins(x=0)
                ax5.margins(x=0)
                ax6.margins(x=0)
                ax7.margins(x=0)
                ax8.margins(x=0)
                
                # remove some spines
                ax6.spines['bottom'].set_visible(False)
                ax7.spines['top'].set_visible(False)
                
                # set labels
                ax0.xaxis.set_label_position('top')
                ax1.xaxis.set_label_position('top')
                ax5.xaxis.set_label_position('top')
                ax5.yaxis.set_label_position('left')
                ax6.yaxis.set_label_position('left')
                ax7.yaxis.set_label_position('left')
                ax8.yaxis.set_label_position('left')
                
                ax0.set_xlabel('Number', fontsize=12, labelpad=6)
                ax5.set_xlabel('Time (s)', fontsize=12, labelpad=8)
                ax5.set_ylabel('Amp.', fontsize=12)
                ax6.set_ylabel('Amp.', fontsize=12)
                ax7.set_ylabel('CC', fontsize=12, labelpad=15)
                ax8.set_ylabel('Shifted (°)', fontsize=12)
                
                # set zero lines
                ax7.axhline(ccThreshold, linewidth=0.8, linestyle='--', color='gray')
                
                # set span
                ax0.axhspan(leftBoundry1T, rightBoundry1T,  alpha=0.2, color='black')
                ax5.axhspan(leftBoundry1T, rightBoundry1T, alpha=0.1, color='black')
                    
                # plot figure number
                ax5.set_title( "c)", x=-0.1, fontsize=16, color='black', loc='left' )
                ax5.text( 0.45, 0.7, "T", fontsize=12, color='black' )
                
                #show
                plt.tight_layout()
                # plt.savefig( "{0}/{1}_First2Steps_T.png".format( outfilePath, nameEachStation[ist][0] ), dpi=360 )
                #plt.savefig( "{0}/{1}_First2Steps_T.svg".format( outfilePath, nameEachStation[ist][0] ), dpi=360 )
                plt.savefig("{0}/{1}_First2Steps_T.pdf".format(outfilePath, nameEachStation[ist][0]))
                plt.show      
    
       
    
        
    #-- calculte the final solution (median)
    numWinnerStep2 = len( depthWellBehavedStation )
    if( numWinnerStep2 > 1):
        finalDepthSolution = np.median(depthWellBehavedStation)
        finalDepthIdx = np.nanargmin(np.abs(depthWellBehavedStation - finalDepthSolution))
        Sta_best[wellBehavedStationId[finalDepthIdx]] = 1
    else:
        finalDepthSolution = depthWellBehavedStation[0]
        finalDepthIdx = 0
    #%%
    print( "finalDepthSolution = {0}".format( format( finalDepthSolution, ".2f" ) ), 'km' )    
         
    #%%############# PLOT ############################################
    if len(depthWellBehavedStation) > 0:
        numWinnerStep2 = len(depthWellBehavedStation)
        if numWinnerStep2 > 1:
            finalDepthSolution = np.median(depthWellBehavedStation)
            finalDepthIdx = np.nanargmin(np.abs(depthWellBehavedStation - finalDepthSolution))
            Sta_best[wellBehavedStationId[finalDepthIdx]] = 1
        else:
            finalDepthSolution = depthWellBehavedStation[0]
            finalDepthIdx = 0
        print("finalDepthSolution = {:.2f} km".format(finalDepthSolution))
    else:
        print("没有有效台站数据，跳过最终解计算和绘图。")
        # 可选择跳过后续绘图或直接退出
        # 这里可以 return 或 sys.exit()，但根据上下文，最好用 else 分支

    # 最终绘图
    if verboseFlag == 1 and len(depthWellBehavedStation) > 0 and len(rmsWellBehavedStation) > 0:
        # 确保 finalDepthIdx 有效
        if finalDepthIdx >= len(rmsWellBehavedStation):
            print("警告: finalDepthIdx 超出范围，重置为 0")
            finalDepthIdx = 0
            # -- plot the number of depth-phase matched of each assumed focal depth
            # and rms
            # set figure layout
            if len(depthWellBehavedStation) == 0 or len(rmsWellBehavedStation) == 0:
                print("没有有效台站数据，跳过 Step4 绘图。")
        # -- plot the number of depth-phase matched of each assumed focal depth
        # and rms
        # set figure layout
        fig = plt.figure( constrained_layout=True, figsize=(5,5))
        fig.subplots_adjust(hspace=0.4)
        fig.subplots_adjust(wspace=0.18)
        gs0 = fig.add_gridspec(1, 1 )
        gs00 = gs0[0].subgridspec(2,1)
        ax0 = fig.add_subplot(gs00[0, 0])
        ax1 = fig.add_subplot(gs00[1, 0])
        #-- ax0: the number of depth-phase matches
        #-- figure number
        t = scanDepthMembers
        ax0.plot(t, sumGlobal, linewidth=2., color="black")
        for ist in range(numSt):
            ax0.plot(t, totNumMatPhaGlobal[ist], linewidth=0.45, linestyle='-', color="grey")
        ax00 = ax0.twinx()  # instantiate a second axes that shares the same x-axis
        ax00.scatter(t, sumAvgArrTimeDiffResGlobal, s=15, marker='s',
                     facecolors='white', edgecolor='gray', zorder=100)
        ax00.scatter(prelimSolution_0, sumAvgArrTimeDiffResGlobal[np.int64(prelimSolution)-np.int64(scanDepthFrom)],
                     s=25, marker='s', facecolors='gray', edgecolor='gray', zorder=100)
        # ax00.axvline( np.array(prelimCandidatesGlobal).min(0)[1],
        #              linewidth=1.5, color='blue', linestyle='--')
        # ax00.axvline( np.array(prelimCandidatesGlobal).max(0)[1],
        #              linewidth=1.5, color='blue', linestyle='--')


        ax00.scatter(t, sumAvgArrTimeDiffResGlobal_2, s=15, marker='s',
                     facecolors='white', edgecolor='blue', zorder=100)
        ax00.scatter(prelimSolution, sumAvgArrTimeDiffResGlobal_2[np.int64(prelimSolution)-np.int64(scanDepthFrom)],
                     s=25, marker='s', facecolors='blue', edgecolor='blue', zorder=100)
        ax00.axvline( np.array(prelimCandidatesGlobal_2).min(0)[1],
                     linewidth=1.5, color='blue', linestyle='--')
        ax00.axvline( np.array(prelimCandidatesGlobal_2).max(0)[1],
                     linewidth=1.5, color='blue', linestyle='--')
        ax00.set_ylim(0, 0.4)

        # set labels
        ax0.set_ylabel('Number of matches', fontsize=12)
        ax0.set_xlabel('Depth (km)', fontsize=12)
        ax00.set_ylabel('Sum of differential\narrival time residuals (s)',
                        color='blue', fontsize=12)
        ax00.tick_params(axis='y', colors='blue')
        #set grid
        ax0.grid(True, linestyle='--', linewidth=0.25)
        ax0.axhline( thresholdMaxNumb,linewidth=1.5, color='black', linestyle='--')
        ax0.margins(x=0)
        ax0.set_xticks( np.arange(scanDepthFrom,scanDepthTo, step=2))                
        
        #-- ax1: rms curve
        minRms = np.min (rmsWellBehavedStation[finalDepthIdx])
        # minRms = np.min( rmsWellBehavedStation )
        maxRms = np.max( rmsWellBehavedStation )
        minDepthRange = np.min(  depthRangeWellBehavedStation )
        maxDepthRange = np.max(  depthRangeWellBehavedStation )
        
        #-- This commend is only for the synthetic example in Section 3.2 of DSA paper
        if velModel == "ak135_Section3.2":
            for iwin in range( numWinnerStep2 ):
                line, = ax1.plot(depthRangeWellBehavedStation[iwin], rmsWellBehavedStation[iwin],
                                 label='RTZ', color='black', linewidth=0.5 )
        # ax1.axvline( 13.5, linewidth=1.2, color='red', linestyle='-' )

        # rmsWellBehavedStation_avg = []
        # for idep in range(len(rmsWellBehavedStation[:][0] )):
        #     rmsWellBehavedStation_avg.append(np.min(rmsWellBehavedStation[idep][:]))
        # id
        # line, = ax1.plot(depthRangeWellBehavedStation[:][0], rmsWellBehavedStation_avg,
        #                      color='black', linewidth=0.5)

        # delta_depth = exactDepthRange[1] - exactDepthRange[0] #0.1 km
        # id_final_sta = []
        # for idep in range(len(rmsWellBehavedStation[:][0])):
        #     if np.fabs(depthWellBehavedStation[idep] - finalDepthSolution)<delta_depth:
        #         id_final_sta.append(idep)
        #
        # rmsWellBehavedStation_avg = []
        # for idep in range(len(rmsWellBehavedStation[:][0] )):
        #     tmp_rms = 0
        #     for id in range(len(id_final_sta)):
        #         tmp_rms = tmp_rms + rmsWellBehavedStation[idep][id_final_sta[id]]
        #         # print(rmsWellBehavedStation[idep][id_final_sta[id]])
        #     tmp_rms = rmsWellBehavedStation[idep][id_final_sta[0]] +rmsWellBehavedStation[idep][id_final_sta[1]]
        #     print(tmp_rms)
        #     rmsWellBehavedStation_avg.append(tmp_rms/len(id_final_sta))
        #
        # line, = ax1.plot(depthRangeWellBehavedStation[:][0], rmsWellBehavedStation_avg,
        #                      color='black', linewidth=0.5)

        # id_final_sta = np.argmin(np.fabs(depthWellBehavedStation - finalDepthSolution)) % only one station with final depth
        # id_final_sta = np.argmin(rmsWellBehavedStation) % len(rmsWellBehavedStation[:][0]) % only the station with min rms
        # line, = ax1.plot(depthRangeWellBehavedStation[:][0], rmsWellBehavedStation[:][id_final_sta],
        #                                       color='black', linewidth=0.5)

        ax1.text( finalDepthSolution-0.3, minRms*1.2,
                  "(x={0}, y={1})".format(
                  format( finalDepthSolution, ".1f"),
                  format( minRms, ".2f") ),
                  fontsize=12, color='black',  rotation=0, zorder=110)
        for ist in range(len(depthWellBehavedStation)):
            ax1.scatter( depthWellBehavedStation[ist], np.min(rmsWellBehavedStation[ist]), s =150,  marker='*',
                    facecolors='white', edgecolor='black', linewidths=0.55, zorder=100)


        ax1.scatter(finalDepthSolution, minRms, s=250, marker='*',
                    facecolors='black', edgecolor='black', zorder=100)

        #set xlim and ylim
        # ax1.set_xlim( minDepthRange, maxDepthRange )
        # ax1.set_xticks(np.arange(minDepthRange, maxDepthRange, step=0.4))
        ax1.set_xlim(prelimSolution-2, prelimSolution+2)
        ax1.set_xticks( np.arange( prelimSolution-2, prelimSolution+2, step=0.4) )
        ax1.set_ylim( 0, 0.4)
        # ax00.set_ylim(minRms, maxRms)

        # set labels
        ax1.set_ylabel('RMS (s)', fontsize=12)
        ax1.set_xlabel('Depth (km)', fontsize=12)
        #set grid
        ax1.grid(True, linestyle='--', linewidth=0.25)
        #-- figure number
        ax0.set_title( "a)", x=-0.2, fontsize=14, color='black', loc='left' )
        ax1.set_title( "b)", x=-0.2, fontsize=14, color='black', loc='left' )
        #save
        plt.tight_layout()
        # plt.savefig( "{0}/Steps3-4_Prelim{1}km_Final{2}km.png".format(
        #             outfilePath,
        #             format( prelimSolution, ".1f"),
        #             format( finalDepthSolution, '.1f') ),
        #             dpi=360 )
        # plt.savefig( "{0}/Steps3-4_Prelim{1}km_Final{2}km.svg".format(
        #             outfilePath,
        #             format( prelimSolution, ".1f"),
        #             format( finalDepthSolution, '.1f') ),
        #             dpi=360 )
        plt.savefig("{0}/Steps3-4_Prelim{1}km_Final{2}km.pdf".format(outfilePath,
                    format(prelimSolution, ".1f"),
                    format(finalDepthSolution, '.1f')))
        plt.show()
        
ist_to_pos = {ist: i for i, ist in enumerate(successful_ist)}
#%% output the initial time difference
df = pd.DataFrame()
df['station'] = [nameEachStation[ist][0] for ist in wellBehavedStationId]
df['lon'] = [Sta_stalon[ist_to_pos[ist]] for ist in wellBehavedStationId]
df['lat'] = [Sta_stalat[ist_to_pos[ist]] for ist in wellBehavedStationId]
df['dist'] = [Sta_distkm[ist_to_pos[ist]] for ist in wellBehavedStationId]
df['dt_R'] = [Sta_timediffR[ist_to_pos[ist]] for ist in wellBehavedStationId]
df['dt_T'] = [Sta_timediffT[ist_to_pos[ist]] for ist in wellBehavedStationId]
df['dt_Z'] = [Sta_timediffZ[ist_to_pos[ist]] for ist in wellBehavedStationId]
df['dt_final'] = [Sta_dt_final[ist] for ist in wellBehavedStationId]
df['n_match'] = [Sta_match[ist] for ist in wellBehavedStationId]
df['dep'] = [Sta_dep_final[ist] for ist in wellBehavedStationId]
df['best'] = [Sta_best[ist] for ist in wellBehavedStationId]
#%%       

outPath2 = str(outfilePath) + '/dt_cmp.csv'
df.to_csv(outPath2, index=False)
resultsFile.close()

#%% calculate computing time
stop = timeit.default_timer()
elapsedTime = stop - start
print('Elapsed time: ', format( elapsedTime, '.1f'),
  'sec = ', format( elapsedTime/60.0, '.1f'), 'min' )

import os
os.system('say "Bobo"')
os.system('say "your program has finished"')
os.system('say "come and have a check"')
