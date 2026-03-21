import numpy as np


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