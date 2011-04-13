import numpy as np
import numpyimgproc as nim
import copy
import adskalman.adskalman as adskalman

import matplotlib.pyplot as plt

###############################################################################
# General Functions: Interpolating, Smooth, Etc.
###############################################################################

###
def normalize(array):
    normed_array = np.zeros([len(array), 1])
    for i in range(len(array)):
        normed_array[i] = np.linalg.norm(array[i])
    return normed_array
    

###
def count_movies(movie_dataset):
    
    total = len(movie_dataset.get_movie_keys(behavior='all'))
    landing_black_post_crash = len(movie_dataset.get_movie_keys(behavior='landing', posttype='black', crash=True))
    landing_black_post_nocrash = len(movie_dataset.get_movie_keys(behavior='landing', posttype='black', crash=False))
    landing_checkered_post_crash = len(movie_dataset.get_movie_keys(behavior='landing', posttype='checkered', crash=True))
    landing_checkered_post_nocrash = len(movie_dataset.get_movie_keys(behavior='landing', posttype='checkered', crash=False))
    flyby_black_post = len(movie_dataset.get_movie_keys(behavior='flyby', posttype='black'))
    flyby_checkered_post = len(movie_dataset.get_movie_keys(behavior='flyby', posttype='checkered'))
    
    print 'Movies in Dataset'
    print 'Total: ', total
    print
    print 'Landing - Crash - Black Post: ', landing_black_post_crash
    print 'Landing - No Crash - Black Post: ', landing_black_post_nocrash
    print 'Landing - Crash - Checkered Post: ', landing_checkered_post_crash
    print 'Landing - No Crash - Checkered Post: ', landing_checkered_post_nocrash
    print
    print 'Flyby - Black Post: ', flyby_black_post
    print 'Flyby - Checkered Post: ', flyby_checkered_post
    
    nmovies_dict = {'landing,nocrash,black': landing_black_post_nocrash, 
                    'landing,crash,black': landing_black_post_crash, 
                    'landing,nocrash,checkered': landing_checkered_post_nocrash,
                    'landing,crash,checkered': landing_checkered_post_crash,
                    'flyby,checkered': flyby_checkered_post,
                    'flyby,black': flyby_black_post}
                    
    return nmovies_dict
    
###
def interpolate_nan(Array):
    if True in np.isnan(Array):
        array = copy.copy(Array)
        for i in range(2,len(array)):
            if np.isnan(array[i]).any():
                array[i] = array[i-1]
        return array
    else:
        return Array
        
###
def kalman_smoother(data, F, H, Q, R, initx, initv, plot=False):
    os = H.shape[0]
    ss = F.shape[0]
    
    interpolated_data = np.zeros_like(data)
    
    for c in range(os):
        interpolated_data[:,c] = interpolate_nan(data[:,c])
        y = interpolated_data
        
    xsmooth,Vsmooth = adskalman.kalman_smoother(y,F,H,Q,R,initx,initv)
    
    if plot:
        plt.plot(xsmooth[:,0])
        plt.plot(y[:,0], '*')

    return xsmooth,Vsmooth
    
###
def smooth_centers(data, plot=False):
    ss = 4 # state size
    os = 2 # observation size
    F = np.array([[1,0,1,0], # process update
                     [0,1,0,1],
                     [0,0,1,0],
                     [0,0,0,1]],
                    dtype=np.float)
    H = np.array([[1,0,0,0], # observation matrix
                     [0,1,0,0]],
                    dtype=np.float)
    Q = 0.0001*np.eye(ss) # process noise
    R = 500*np.eye(os) # observation noise

    init_vel = np.mean(np.diff(data[0:20,:], axis=0), axis=0)

    initx = np.array([data[0,0], data[0,1], init_vel[0], init_vel[1]],dtype=np.float)
    initv = 0*np.eye(ss)

    xsmooth,Vsmooth = kalman_smoother(data, F, H, Q, R, initx, initv, plot=plot)
    
    return xsmooth

###
def smooth_orientation_2d(ori, plot=False):
    ss = 4 # state size
    os = 2 # observation size
    F = np.array([[1,0,1,0], # process update
                     [0,1,0,1],
                     [0,0,1,0],
                     [0,0,0,1]],
                    dtype=np.float)
    H = np.array([[1,0,0,0], # observation matrix
                     [0,1,0,0]],
                    dtype=np.float)
    Q = 0.0001*np.eye(ss) # process noise
    Q[2,2] = .001
    Q[3,3] = .001
    R = 100*np.eye(os) # observation noise

    init_vel = np.mean(np.diff(ori[0:20,:], axis=0), axis=0)

    initx = np.array([ori[0,0], ori[0,1], init_vel[0], init_vel[1]],dtype=np.float)
    initv = 0*np.eye(ss)

    xsmooth,Vsmooth = kalman_smoother(ori, F, H, Q, R, initx, initv, plot=plot)
    
    return xsmooth

###
def fix_orientation_2d(movieinfo, switching_threshold = 0.2, ratio_threshold=1.0, pixelspeed_threshold=100):
    ori = movieinfo.obj_longaxis
    Vel = movieinfo.smooth_vel
    
    vel = np.array([ Vel[i,:] / np.linalg.norm(Vel[i,:]) for i in range(len(Vel))])
    pixelspeed = np.array([ np.linalg.norm(Vel[i,:]) for i in range(len(Vel))])
    
    interpolated_data = np.zeros_like(ori)
    for c in range(ori.shape[1]):
        interpolated_data[:,c] = interpolate_nan(ori[:,c])
    ori = interpolated_data

    # flies don't spin around immediately, so generally body angle should be rouhgly the same from frame to frame, at least within 180 deg
    # use this principal to fix the orientation vectors
    # switching threshold sets how fast a fly is allowed to fly/move backwards
    dot_prev_ori = np.zeros(len(ori))
    dot_vel = np.zeros(len(vel))
    
    smooth_ori = np.zeros_like(ori)
    smooth_ori[0,:] = ori[0,:]
    for i in range(len(ori)):
        
        if i < 1:
            dot_vel[i] = np.dot(vel[i], ori[i])
            if dot_vel[i] < 0 and np.abs(dot_vel[i]) > .5:
                direction = -1
            else:
                direction = 1
            smooth_ori[i] = ori[i]*direction
            
            
        else:
            switchingthreshold = switching_threshold         
            dot_vel[i] = np.dot(vel[i], ori[i])
            direction = 1.
            if dot_vel[i] < 0 and np.abs(dot_vel[i]) > switchingthreshold:
                direction = -1
                
            smooth_ori[i] = ori[i]*direction
                
            if 1:
                dot_prev_ori[i] = np.dot(smooth_ori[i-1], smooth_ori[i])
                if dot_prev_ori[i] < 0: # not aligned with previous frame by > 90 deg
                    print i, direction
                    direction = -1
                    print i, direction 
                else:
                    direction = 1
                 
            smooth_ori[i] = smooth_ori[i]*direction
            
            ratio = movieinfo.smooth_ratio[i, 0] / movieinfo.smooth_ratio[i, 1]
            ratiop = movieinfo.smooth_ratio[i-1, 0] / movieinfo.smooth_ratio[i-1, 1]
            if np.isnan(ratio) or ratio < ratio_threshold:
                smooth_ori[i] = smooth_ori[i-1]
                
            #if np.linalg.norm(smooth_ori[i] - smooth_ori[i-1]) > 0.4:
            #    smooth_ori[i] = smooth_ori[i-1] 
        
    return smooth_ori
    
###
def remove_angular_rollover(A, max_change_acceptable):
    array = copy.copy(A)
    for i, val in enumerate(array):
        if i == 0:
            continue
        diff = array[i] - array[i-1]
        if np.abs(diff) > max_change_acceptable:
            factor = np.round(np.abs(diff)/(np.pi))  
            if iseven(factor):
                array[i] -= factor*np.pi*np.sign(diff)
    if len(A) == 2:
        return array[1]
    else:
        return array
        
###
def rotate_coordinates(old_coordinates, basis1, basis2):
    if len(old_coordinates.shape) == 1:
        b1 = basis1
        b2 = basis2
        n = np.zeros_like(old_coordinates)
        n[0] = n1 = np.dot(old_coordinates, b1)
        n[1] = np.dot(old_coordinates, b2)
        return n

    elif len(old_coordinates.shape) == 2:
    
        n = np.zeros_like(old_coordinates)
        for i in range(len(old_coordinates)):
            b1 = basis1[i]
            b2 = basis2[i]
            n[i,0] = np.dot(old_coordinates[i], b1)
            n[i,1] = np.dot(old_coordinates[i], b2)
        return n
        
###
def iseven(n):
    if int(n)/2.0 == int(n)/2:
        return True
    else:
        return False 
def isodd(n):
    if int(n)/2.0 == int(n)/2:
        return False
    else:
        return True
    
###############################################################################
# Image Plot stuff
###############################################################################

###
def strobe_from_movieinfo(movieinfo, interval=200, frames=None):

    bkgrd = movieinfo.background
    strobe_img = copy.copy(bkgrd)
    
    f = 0
    while f < len(movieinfo.frames):
        frame = movieinfo.frames[f]
        uimg = frame.uimg
        zero = frame.zero
        size = frame.uimg.shape
        
        blank = 255*np.ones_like(bkgrd)
        blank[ zero[0]:zero[0]+size[0], zero[1]:zero[1]+size[1] ] = uimg
        strobe_img = nim.darken(strobe_img, blank)
        
        f += interval
                
    return strobe_img
    
###############################################################################
# Processing Movie Data
###############################################################################

###
def process_movie_dataset(movie_dataset, reprocess=False):   

    keys = movie_dataset.get_movie_keys(behavior='all')
    for key in keys:
        movieinfo = movie_dataset.movies[key]
        if reprocess is False:
            try:
                tmp = movieinfo.scaled
            except:
                print 'need to process! ', movieinfo.id
                process_movieinfo(movieinfo)
        else:
            process_movieinfo(movieinfo)

###
def process_movieinfo(movieinfo):
    print 'processing: ', movieinfo.id

    try: fake = movieinfo.fake
    except: fake = False

    smooth_movieinfo(movieinfo)
    movieinfo.smooth_head_pos = movieinfo.smooth_centers + (movieinfo.smooth_ratio[:,0].reshape([len(movieinfo.smooth_ratio[:,0]),1]))*movieinfo.smooth_ori
    movieinfo.smooth_head_pos = interpolate_nan(movieinfo.smooth_head_pos)
    
    if not fake: calc_post_pos(movieinfo)
    
    calc_fly_coordinates(movieinfo)
    
    if not fake: calc_scale(movieinfo)
    else: movieinfo.scale = 1 
    calc_fly_coordinates_with_scale(movieinfo)
    
###
def reprocess_function(movie_dataset, function):
    keys = movie_dataset.get_movie_keys()
    for key in keys:
        movieinfo = movie_dataset.movies[key]
        function(movieinfo)
    
###
def smooth_movieinfo(movieinfo):
    try: fake = movieinfo.fake
    except: fake = False
    
    movieinfo.smooth_ratio = smooth_centers(movieinfo.obj_ratio)
    xsmooth = smooth_centers(movieinfo.obj_centers)
    movieinfo.smooth_centers = xsmooth[:,0:2]
    movieinfo.smooth_vel = xsmooth[:,2:4] * float(movieinfo.framerate)
    
    if not fake:
        calc_fly_heading(movieinfo)
    else:
        calc_fly_heading(movieinfo, pixelspeed_threshold=0)
        
    fixed_ori = fix_orientation_2d(movieinfo, switching_threshold = 0.2, ratio_threshold=1.4, pixelspeed_threshold=100)
    #remove_orientation_errors_due_short_major_axis(movieinfo, thresh=2, pixelspeed=70, plot=False)

    xsmooth = smooth_orientation_2d(fixed_ori)
    movieinfo.smooth_ori = xsmooth[:,0:2]
    for i in range(len(movieinfo.smooth_ori)):
        movieinfo.smooth_ori[i] = movieinfo.smooth_ori[i] / np.linalg.norm( movieinfo.smooth_ori[i] )
        
    movieinfo.smooth_ori_vel = xsmooth[:,2:4]

###
def remove_orientation_errors_due_short_major_axis(movieinfo, thresh=2, pixelspeed=1, plot=False):
    # do before smoothing / kalmanizing
    new_ori = copy.copy(movieinfo.smooth_ori) 
    for i in range(2,len(new_ori)):
    
        if np.linalg.norm(movieinfo.smooth_vel[i]) < pixelspeed:
            new_ori[i] = new_ori[i-1]
        
        ratio = movieinfo.smooth_ratio[i, 0] / movieinfo.smooth_ratio[i, 1]
        if ratio < thresh:
            new_ori[i] = new_ori[i-1]
    
    if plot:
        plt.plot(movieinfo.smooth_ori[:,0])
        plt.plot(new_ori[:,0], '*')
    
    movieinfo.smooth_ori = new_ori
    
    return new_ori    
    
###
def calc_post_pos(movieinfo):
    threshed_img = nim.threshold(movieinfo.background, 0, 35)
    blob = nim.find_biggest_blob(threshed_img)
    center, radius = nim.find_circle(blob, npts=200, navg=30, nstart=0, plot=False)
    movieinfo.post_pos = center
    movieinfo.post_radius_in_pixels = radius
    
###
def calc_fly_heading(movieinfo, pixelspeed_threshold=100):
    Vel = movieinfo.smooth_vel
    vel = np.array([ Vel[i,:] / np.linalg.norm(Vel[i,:]) for i in range(len(Vel))])
    pixelspeed = np.array([ np.linalg.norm(Vel[i,:]) for i in range(len(Vel))])
    heading = np.zeros_like(vel)
    for i in range(len(pixelspeed)):
        if pixelspeed[i] > pixelspeed_threshold:
            heading[i] = vel[i]
        else:
            heading[i] = heading[i-1]
    movieinfo.heading = heading
    
###
class Flycoord:
        pass
def calc_fly_coordinates(movieinfo):

    movieinfo.flycoord = Flycoord()
    post_pos = movieinfo.post_pos
    
    movieinfo.flycoord.angletopost = np.zeros([movieinfo.smooth_centers.shape[0], 1])
    movieinfo.flycoord.worldangle = np.zeros([movieinfo.smooth_centers.shape[0], 1])
    movieinfo.flycoord.slipangle = np.zeros([movieinfo.smooth_centers.shape[0], 1])
    movieinfo.flycoord.vel = np.zeros_like(movieinfo.smooth_centers)
    movieinfo.flycoord.vec_to_post = np.zeros_like(movieinfo.smooth_centers)
    movieinfo.flycoord.dist_to_post = np.zeros([movieinfo.smooth_centers.shape[0], 1])
    movieinfo.smooth_shortaxis = np.zeros_like(movieinfo.smooth_ori)
    
    # slipangle
    heading3vec = np.hstack( (movieinfo.heading, np.zeros([movieinfo.heading.shape[0], 1]) ) )
    ori3vec = np.hstack( (movieinfo.smooth_ori, np.zeros([movieinfo.smooth_ori.shape[0], 1]) ) )
    for i in range(len(ori3vec)):
        sinslipangle = (np.cross( heading3vec[i], ori3vec[i] ) / (np.linalg.norm(heading3vec[i])*np.linalg.norm(ori3vec[i]))).sum()
        cosslipangle = np.dot( heading3vec[i], ori3vec[i] ) / (np.linalg.norm(heading3vec[i])*np.linalg.norm(ori3vec[i]))
        movieinfo.flycoord.slipangle[i] = -1*np.sign(sinslipangle)*np.arccos(cosslipangle)
        
    # misc
    for i in range(len(movieinfo.smooth_centers)):
        movieinfo.smooth_shortaxis[i,0] = movieinfo.smooth_ori[i,1]
        movieinfo.smooth_shortaxis[i,1] = movieinfo.smooth_ori[i,0]*-1
        
        movieinfo.flycoord.worldangle[i] = np.arctan2(movieinfo.smooth_ori[i,1], movieinfo.smooth_ori[i,0]) # want angle between 0 and 360 deg
        if movieinfo.flycoord.worldangle[i] < 0:
            movieinfo.flycoord.worldangle[i] = np.pi*2+movieinfo.flycoord.worldangle[i]
        movieinfo.flycoord.worldangle[i] = remove_angular_rollover( [movieinfo.flycoord.worldangle[i-1], movieinfo.flycoord.worldangle[i]], .5 )
        
        vec_to_post = post_pos - movieinfo.smooth_centers[i]
        movieinfo.flycoord.dist_to_post[i] = np.linalg.norm(vec_to_post)
        movieinfo.flycoord.vec_to_post[i] = vec_to_post / np.linalg.norm(vec_to_post)
        
        cosangletopost = np.dot(vec_to_post, movieinfo.smooth_ori[i]) / ( np.linalg.norm(vec_to_post)*np.linalg.norm(movieinfo.smooth_ori[i]) )
        movieinfo.flycoord.angletopost[i] = np.arccos(cosangletopost)
            
    movieinfo.flycoord.vel = rotate_coordinates(movieinfo.smooth_vel, movieinfo.smooth_ori, movieinfo.smooth_shortaxis)
    movieinfo.flycoord.slipangle = movieinfo.flycoord.slipangle.reshape(len(movieinfo.flycoord.slipangle))
    
    #movieinfo.trajec.angle_subtended_by_post = 2*np.arcsin( movieinfo.trajec.stimulus.radius / (movieinfo.trajec.dist_to_stim_r+movieinfo.trajec.stimulus.radius) ).reshape([movieinfo.trajec.dist_to_stim_r.shape[0],1])
    
    y = np.sin(movieinfo.flycoord.worldangle)
    x = np.cos(movieinfo.flycoord.worldangle)
    smooth_ori_3vec = np.hstack( ( np.hstack( (x,y) ), np.zeros([len(x),1]) ) )
    vec_to_post_3vec = np.hstack( (movieinfo.flycoord.vec_to_post, np.zeros([len(movieinfo.flycoord.vec_to_post),1]) ) ) 
    
    signed_angle_to_post_smooth = np.sum(np.cross( vec_to_post_3vec, smooth_ori_3vec ), axis=1).reshape([vec_to_post_3vec.shape[0],1])
    mag_vec_to_post = np.array([ np.linalg.norm( vec_to_post_3vec[i,:] )  for i in range(vec_to_post_3vec.shape[0]) ]).reshape([vec_to_post_3vec.shape[0], 1])
    mag_ori = np.array([ np.linalg.norm( smooth_ori_3vec[i,:] )  for i in range(smooth_ori_3vec.shape[0]) ]).reshape([smooth_ori_3vec.shape[0], 1])
    sin_signed_angle_to_post = signed_angle_to_post_smooth / (mag_vec_to_post*mag_ori)
    sign_of_angle_to_post = np.sign(np.arcsin(sin_signed_angle_to_post))
     
    movieinfo.flycoord.signed_angletopost = -1*movieinfo.flycoord.angletopost*sign_of_angle_to_post

    # rotational velocity:
    tmp_angles = np.vstack( (movieinfo.flycoord.worldangle[0], movieinfo.flycoord.worldangle) )
    tmp_angles = tmp_angles.reshape([len(tmp_angles)])
    movieinfo.flycoord.worldangle_vel = np.diff(tmp_angles) * float(movieinfo.framerate)

###    
class Scaled:
        pass
def calc_fly_coordinates_with_scale(movieinfo):

    post_pos = movieinfo.post_pos
    scale = movieinfo.scale
    movieinfo.scaled = Scaled()
    
    movieinfo.post_radius = movieinfo.post_radius_in_pixels*movieinfo.scale
    
    movieinfo.scaled.dist_to_post = scale*movieinfo.flycoord.dist_to_post
    
    movieinfo.scaled.head_pos = (movieinfo.smooth_head_pos-post_pos)*movieinfo.scale
    movieinfo.scaled.dist_head_to_post = np.array([ np.linalg.norm(movieinfo.scaled.head_pos[i]) for i in range(len(movieinfo.scaled.head_pos)) ])
    
    movieinfo.scaled.signed_angletopost = remove_angular_rollover(movieinfo.flycoord.signed_angletopost, .5)
    movieinfo.scaled.slipangle = remove_angular_rollover(movieinfo.flycoord.slipangle, .5)
    
    movieinfo.scaled.angle_subtended_by_post = 2*np.arcsin( movieinfo.post_radius / (movieinfo.scaled.dist_head_to_post) ).reshape([movieinfo.scaled.dist_head_to_post.shape[0],1])
    # need to remove NAN's.. 
    where_dist_less_than_zero = (np.where( np.isnan(movieinfo.scaled.angle_subtended_by_post) == True )[0]).tolist()
    movieinfo.scaled.angle_subtended_by_post[where_dist_less_than_zero] = np.pi
    
    movieinfo.scaled.positions = (movieinfo.smooth_centers-post_pos)*movieinfo.scale
    movieinfo.scaled.velocities = movieinfo.smooth_vel*movieinfo.scale
    movieinfo.scaled.speed = np.array([np.linalg.norm(movieinfo.scaled.velocities[i]) for i in range(len(movieinfo.scaled.velocities))])
    
    movieinfo.scaled.angle_to_edge = np.abs(movieinfo.scaled.signed_angletopost)-np.abs(movieinfo.scaled.angle_subtended_by_post)/2.
    movieinfo.scaled.angle_to_lower_edge = (movieinfo.scaled.signed_angletopost - movieinfo.scaled.angle_subtended_by_post/2.).reshape([len(movieinfo.scaled.signed_angletopost)])
    movieinfo.scaled.angle_to_upper_edge = (movieinfo.scaled.signed_angletopost + movieinfo.scaled.angle_subtended_by_post/2.).reshape([len(movieinfo.scaled.signed_angletopost)])
    
    
    
###
def calc_scale(movieinfo, plot=False):
    nt, flydra_dist = interpolate_to_new_framerate(movieinfo, 200, movieinfo.trajec.epoch_time, movieinfo.trajec.dist_to_stim_r+movieinfo.trajec.stimulus.radius)
    nt, sa1_dist = interpolate_to_new_framerate(movieinfo, 200, movieinfo.timestamps, movieinfo.flycoord.dist_to_post)
    
    if movieinfo.behavior == 'landing': # ignore last bit
        frames_away_from_post = np.where(flydra_dist > .005)[0]
        flydra_dist = flydra_dist[frames_away_from_post]
        sa1_dist = sa1_dist[frames_away_from_post]
        nt = nt[frames_away_from_post]
        
    #sa1_dist -= movieinfo.post_pos #np.min(sa1_dist)
    #flydra_dist -= np.min(flydra_dist)
    
    scale_arr = flydra_dist / sa1_dist
    scale = np.median(scale_arr) # get problems with zeros in denominator / numerator if use mean
    
    movieinfo.scale = scale
    
    if plot:
        plt.plot(sa1_dist*scale)
        plt.plot(flydra_dist, '*')
        
###############################################################################
# Plotting Helper Functions
###############################################################################

###
def interpolate_to_new_framerate(movieinfo, framerate, old_timestamps, data):
    new_timestamps = np.arange(movieinfo.syncrange[0], movieinfo.syncrange[1], 1/float(framerate))
    try:
        data = data.reshape(len(data))
        new_data = np.interp(new_timestamps, old_timestamps, data)
        return new_timestamps, new_data
    except:
        new_data = np.zeros([len(new_timestamps),data.shape[1]])
        for c in range(data.shape[1]):
            col = data[:,c].reshape(len(data))
            new_data[:,c] = np.interp(new_timestamps, old_timestamps, col)
        return new_timestamps, new_data
    
    
###
def get_frames_until_landing(movieinfo):
    
    if movieinfo.landingframe is None:
        frames = (np.array(movieinfo.framenumbers)-movieinfo.framenumbers[0]).tolist()
        return frames
        
    else:
        frames = ( np.where(np.array(movieinfo.framenumbers) < (movieinfo.landingframe-movieinfo.startframe)) [0] ).tolist()
        return frames
        
###
def get_time_nearest_to_post(movieinfo):
    if movieinfo.landingframe is None:
        f = np.argmin(movieinfo.scaled.dist_to_post)
        t = movieinfo.timestamps[f]
        return t
    else:
        return movieinfo.landingtime 
    
    
###
def frame_to_timestamp(movieinfo, frames):   
    if type(frames) is list:
        frames = np.array(frames)
        framelist = (frames-movieinfo.startframe-movieinfo.framenumbers[0]).tolist()
    elif type(frames) is np.ndarray:
        framelist = (frames-movieinfo.startframe-movieinfo.framenumbers[0]).tolist()
    else:
        framelist = frames
    timestamps = movieinfo.timestamps[framelist]
    if len(timestamps) == 1:
        timestamps = timestamps[0]
    return timestamps

###
def get_frame_from_timestamp(movieinfo, timestamp, timestamps=None):
    if timestamps is None:
        return np.argmin( np.abs(movieinfo.timestamps - timestamp) )
    else:
        return np.argmin( np.abs(timestamps - timestamp) )

###
def get_angle_to_nearest_edge(obj_pos, obj_ori, post_pos, post_radius):
    
    vec_to_post = post_pos - obj_pos
    dist_pos_to_post = np.linalg.norm(vec_to_post)
    obj_ori /= np.linalg.norm(obj_ori)    
    
    worldangle = np.arctan2(obj_ori[1], obj_ori[0]) # want angle between 0 and 360 deg
    if worldangle < 0:
        worldangle = np.pi*2+worldangle
    # remove angular rollover?    
    
    obj_ori_3vec = np.hstack( ( obj_ori, 0) ) 
    vec_to_post_3vec = np.hstack( (vec_to_post, 0 ) ) 
    
    signed_angle_to_post = np.sum(np.cross( vec_to_post_3vec, obj_ori_3vec ) )
    sin_signed_angle_to_post = signed_angle_to_post / (dist_pos_to_post)
    sign_of_angle_to_post = np.sign(np.arcsin(sin_signed_angle_to_post))
    
    cosangletopost = np.dot(vec_to_post, obj_ori) / dist_pos_to_post 
    angletopost = np.arccos(cosangletopost)
     
    signed_angletopost = -1*angletopost*sign_of_angle_to_post
    
    angle_subtended_by_post = 2*np.arcsin( post_radius / (dist_pos_to_post) )
    # need to remove NAN's.. 
    if np.isnan(angle_subtended_by_post): angle_subtended_by_post = np.pi
    angle_to_edge = np.abs(signed_angletopost)-np.abs(angle_subtended_by_post)/2.
    
    return worldangle, angle_to_edge    
    
###############################################################################
# General ANALYSIS functions
###############################################################################

###
def get_indices_during_fixation(movieinfo, fixation_threshold_degrees=5, fixation_duration_threshold=0.1):
    fixation_threshold_radians = fixation_threshold_degrees*np.pi / 180.

    fixation_duration_threshold_frames = int(fixation_duration_threshold * float(movieinfo.framerate))    
    fixation_lower = movieinfo.scaled.angle_to_lower_edge - movieinfo.scaled.slipangle
    fixation_upper = movieinfo.scaled.angle_to_upper_edge - movieinfo.scaled.slipangle  
    
    def get_continuous_data(fixation_lower):
    
        lower_fixation_angle_bool_tmp = np.abs(fixation_lower) < fixation_threshold_radians
        fixation_lower_diff = np.diff( np.hstack( (fixation_lower[0], fixation_lower) ) )
        lower_fixation_diff_bool_tmp = fixation_lower_diff < 0.0005
        lower_fixation_bool_tmp = lower_fixation_diff_bool_tmp #lower_fixation_angle_bool_tmp*
        lower_fixation_indices_tmp = np.where( lower_fixation_bool_tmp == True )[0].tolist()
        #lower_fixation_indices_tmp = np.where( np.abs(fixation_lower) < fixation_threshold_radians )[0].tolist()
        continuous_sequences = nim.find_blobs( lower_fixation_indices_tmp, sizerange=[fixation_duration_threshold_frames,np.inf], dilate=3)
        lower_fixation_indices = []
        for sequence in continuous_sequences:
            indices = np.where(sequence == 1)[0].tolist()
            lower_fixation_indices = np.hstack( (lower_fixation_indices, np.array(lower_fixation_indices_tmp)[indices]) )
        return lower_fixation_indices.tolist()
        
    upper_fixation_indices = get_continuous_data(fixation_upper)
    lower_fixation_indices = get_continuous_data(fixation_lower)
        
    return  upper_fixation_indices, lower_fixation_indices
   
###
def get_slipangle_during_fixation(movieinfo, fixation_threshold_degrees=5, fixation_duration_threshold=0.1):
    upper_fixation_indices, lower_fixation_indices = get_indices_during_fixation(movieinfo, fixation_threshold_degrees, fixation_duration_threshold)
    slipangle_during_upper_fixation = movieinfo.scaled.slipangle[upper_fixation_indices]
    slipangle_during_lower_fixation = movieinfo.scaled.slipangle[lower_fixation_indices]
    slipangle_during_fixation = np.hstack( (slipangle_during_upper_fixation, slipangle_during_lower_fixation) )
    
    return slipangle_during_fixation
    
###
def calc_pitch_estimate(movieinfo):
        
    real_fly_short_axis = .001
    real_fly_long_axis = .0025


    theta_arr = np.linspace(5*np.pi/180., 85*np.pi/180., 100)
    projected_real_fly_long_axis = real_fly_long_axis * np.cos(theta_arr)
    real_fly_ratio = real_fly_short_axis / projected_real_fly_long_axis
        
    
    fly_ratio = movieinfo.smooth_ratio[:, 1] / movieinfo.smooth_ratio[:, 0]
    movieinfo.scaled.pitchangle = np.zeros([len(fly_ratio), 1])
    for i, r in enumerate(fly_ratio):
        movieinfo.scaled.pitchangle[i] = np.interp(r, real_fly_ratio, theta_arr)
    
    #movieinfo.scaled.pitchangle = np.arccos( movieinfo.smooth_ratio[:, 0]*movieinfo.scale*2 / real_fly_long_axis )
    
###
def get_speed_during_fixation(movieinfo, fixation_threshold_degrees=5, fixation_duration_threshold=0.1):
    upper_fixation_indices, lower_fixation_indices = get_indices_during_fixation(movieinfo, fixation_threshold_degrees, fixation_duration_threshold)
    
    speed_during_upper_fixation = movieinfo.scaled.speed[upper_fixation_indices]
    speed_during_lower_fixation = movieinfo.scaled.speed[lower_fixation_indices]
    speed_during_fixation = np.hstack( (speed_during_upper_fixation, speed_during_lower_fixation) )
    print movieinfo.id, len(speed_during_fixation)
    return speed_during_fixation
    
###
def get_trajectory_list_from_short_tuples(tuples):
    
    year_month_base = '201011'
    movie_name_base = '_C001H001S00'
    
    keys = []
    
    for tup in tuples:
        key = year_month_base + str(tup[0]) + movie_name_base + str(tup[1])
        keys.append(key)
        
    return keys
def get_fixation_keys():
    tuples = [(10,38), (10,32), (10,26), (11,38), (11,16), (13,17), (11,43), (11,44), (10,38), (10,32), (10,26), (10,26), (10,14), (13,20), (10,47), (10,23), (10, 11), (13,10), (13,20)]
    keys =  get_trajectory_list_from_short_tuples(tuples)
    return keys        
    
    
###############################################################################
# General ANALYSIS functions
###############################################################################
    
###
def calc_fixation_for_flydra_trajectory(trajec):
    
    pos = trajec.positions[:,0:2]
    ori = trajec.velocities[:,0:2]    
    trajec.angletoedge = np.zeros_like(trajec.speed)
    for i in range(len(ori)):
        ori[i,:] /= np.linalg.norm(ori[i,:])
        worldangle, angle_to_edge = get_angle_to_nearest_edge(pos[i], ori[i], np.array([0,0]), trajec.stimulus.radius)
        trajec.angletoedge[i] = angle_to_edge
def calc_fixation_for_flydra_dataset(dataset):
    for k, trajec in dataset.trajecs.items():
        calc_fixation_for_flydra_trajectory(trajec)
def calc_flydra_fixation_for_movieinfo(movieinfo):
    calc_fixation_for_flydra_trajectory(movieinfo.trajec)
    
###
def get_flydra_indices_during_fixation(trajec, fixation_threshold_degrees=5, fixation_duration_threshold=0.1):
    fixation_threshold_radians = fixation_threshold_degrees*np.pi / 180.
    fixation_duration_threshold_frames = int(fixation_duration_threshold * float(trajec.fps))    
    
    def get_continuous_data(fixation):
        fixation_angle_bool_tmp = np.abs(fixation) < fixation_threshold_radians
        dist_bool = trajec.dist_to_stim_r > 0.01
        altitude_bool_high = trajec.positions[:,2] < -.01
        altitude_bool_low = trajec.positions[:,2] > -.1
        fixation_bool_tmp = fixation_angle_bool_tmp*altitude_bool_high*altitude_bool_low*dist_bool
        #fixation_diff = np.diff( np.hstack( (fixation[0], fixation) ) )
        #fixation_diff_bool_tmp = fixation_diff < 0.0005
        #fixation_bool_tmp = fixation_angle_bool_tmp#*diff_bool_tmp
        #fixation_indices_tmp = np.where( fixation_angle_bool_tmp == True )[0].tolist()
        #fixation_indices_tmp = np.where( np.abs(fixation) < fixation_threshold_radians )[0].tolist()
        continuous_sequences = nim.find_blobs( fixation_bool_tmp, sizerange=[fixation_duration_threshold_frames,np.inf], dilate=False)
        fixation_indices = []
        for sequence in continuous_sequences:
            indices = np.where(sequence == 1)[0].tolist()
            fixation_indices = np.hstack( (fixation_indices, indices) )
        return fixation_indices.tolist()
        
    fixation_indices = get_continuous_data(trajec.angletoedge)
        
    return  fixation_indices
    
###
def get_flydra_indices_not_fixation(trajec, fixation_threshold_degrees=5, fixation_duration_threshold=0.1):
    fixation_threshold_radians = fixation_threshold_degrees*np.pi / 180.
    fixation_duration_threshold_frames = int(fixation_duration_threshold * float(trajec.fps))    
    
    def get_continuous_data(fixation):
        fixation_angle_bool_tmp = np.abs(fixation) > fixation_threshold_radians
        #fixation_angle_bool_tmp_2 = np.abs(fixation) < 20*np.pi / 180.
        dist_bool = trajec.dist_to_stim_r > 0.01
        altitude_bool_high = trajec.positions[:,2] < -.01
        altitude_bool_low = trajec.positions[:,2] > -.1
        fixation_bool_tmp = fixation_angle_bool_tmp*altitude_bool_high*dist_bool
        #fixation_diff = np.diff( np.hstack( (fixation[0], fixation) ) )
        #fixation_diff_bool_tmp = fixation_diff < 0.0005
        #fixation_bool_tmp = fixation_angle_bool_tmp#*diff_bool_tmp
        #fixation_indices_tmp = np.where( fixation_angle_bool_tmp == True )[0].tolist()
        #fixation_indices_tmp = np.where( np.abs(fixation) < fixation_threshold_radians )[0].tolist()
        continuous_sequences = nim.find_blobs( fixation_bool_tmp, sizerange=[fixation_duration_threshold_frames,np.inf], dilate=False)
        fixation_indices = []
        for sequence in continuous_sequences:
            indices = np.where(sequence == 1)[0].tolist()
            fixation_indices = np.hstack( (fixation_indices, indices) )
        return fixation_indices.tolist()
        
    fixation_indices = get_continuous_data(trajec.angletoedge)
        
    return  fixation_indices
    
###
def get_flydra_speed_during_fixation_means(trajec, fixation_threshold_degrees=5, fixation_duration_threshold=0.1):
    fixation_indices = get_flydra_indices_during_fixation(trajec, fixation_threshold_degrees, fixation_duration_threshold)
    
    if len(fixation_indices) > 0:
        diffarr = np.diff( np.hstack( (fixation_indices[0], fixation_indices) ) )
        continuous_sequences = nim.find_blobs( diffarr )
        
        mean = []
        std = []
        for sequence in continuous_sequences:
            indices = np.where(sequence == 1)[0].tolist()
            mean.append (np.mean(trajec.speed[indices]))
            std.append( np.std(trajec.speed[indices]))
            
            
        
        return mean, std
    else:
        return [], []
    
###
def get_flydra_speed_during_fixation(trajec, fixation_threshold_degrees=5, fixation_duration_threshold=0.1):
    fixation_indices = get_flydra_indices_during_fixation(trajec, fixation_threshold_degrees, fixation_duration_threshold)
    return trajec.speed[fixation_indices]
def get_flydra_speed_not_fixation(trajec, fixation_threshold_degrees=5, fixation_duration_threshold=0.1):
    fixation_indices = get_flydra_indices_not_fixation(trajec, fixation_threshold_degrees, fixation_duration_threshold)
    return trajec.speed[fixation_indices]
    
###
def get_bouts_of_constant_speed(movieinfo, accel_threshold=.005, duration_threshold=0.2): 
    duration_threshold_frames = duration_threshold*float(movieinfo.framerate)
    
    
    #speed_diff = np.diff( np.hstack( (movieinfo.scaled.speed[0], movieinfo.scaled.speed) ) )
    #speed_diff_bool_tmp = np.abs(speed_diff) < accel_threshold
    
    
    nt, alt_speed = interpolate_to_new_framerate(movieinfo, 5000., movieinfo.trajec.epoch_time, movieinfo.trajec.velocities[:,2])
    nt, speed = interpolate_to_new_framerate(movieinfo, 5000., movieinfo.timestamps, movieinfo.scaled.speed)
    
    speed_diff = np.diff( np.hstack( (speed[0], speed) ) )
    speed_diff_bool_tmp = np.abs(speed_diff) < accel_threshold
    
    alt_speed_diff = np.diff( np.hstack( (alt_speed[0], alt_speed) ) )
    alt_speed_diff_bool_tmp = np.abs(alt_speed_diff) < .005
    
    
    bool_tmp = alt_speed_diff_bool_tmp*speed_diff_bool_tmp
    
    continuous_sequences = nim.find_blobs( bool_tmp, sizerange=[duration_threshold_frames,np.inf], dilate=1)
    
        
    calc_pitch_estimate(movieinfo)
    
    pitch = []
    speed = []
    for i, sequence in enumerate(continuous_sequences):
        indices = np.where(sequence == 1)[0].tolist()
        
        if np.sum(sequence) > 0:
            print movieinfo.id, len(sequence)
        
        r1 = np.median(movieinfo.scaled.speed[indices])
        r2 = np.median(movieinfo.scaled.pitchangle[indices])
        
        if not np.isnan(r1) and not np.isnan(r2):
            pitch.append(r2)
            speed.append(r1)
            
    return pitch, speed


def get_keys(movie_dataset, behavior, subbehavior, crash=False):
    
    if type(behavior) is not list:
        behavior = [behavior]
    if behavior is None:
        behavior = ['landing', 'flyby']
    
    allkeys = movie_dataset.get_movie_keys(behavior='all')
    keys = []
    for key in allkeys:
        movieinfo = movie_dataset.movies[key]
        
        if 'crash' in movieinfo.subbehavior or 'wingcrash' in movieinfo.subbehavior:
            iscrash = True
        else:
            iscrash = False
            
        if subbehavior in movieinfo.subbehavior and movieinfo.behavior in behavior and iscrash == crash:
            keys.append(key)
        
    return keys

    
        
    
    
    
    
