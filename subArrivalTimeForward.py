from obspy.taup import TauPyModel
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