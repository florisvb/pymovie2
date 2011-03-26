import numpy as np
import numpyimgproc as nim
import copy
import adskalman.adskalman as adskalman

import matplotlib.pyplot as plt

###############################################################################
# General Functions: Interpolating, Smooth, Etc.
###############################################################################

###
def interpolate(Array, values):
    # this function will run through the array, and replace any EXACT instances of the values given with linear interpolations from the array
    array = copy.copy(Array)
    if type(values) is not list:
        values = [values]
    for i in range(1,len(array)):
        if array[i] in values:
            future_val = values[0]
            future_i = i
            while future_val in values:
                future_i += 1
                if future_i >= len(array):
                    future_i = i
                    break
                future_val = array[future_i]
            delta_val = (array[future_i] - array[i-1]) / float(future_i- (i-1) )
            array[i] = array[i-1] + delta_val
    return array
    
###
def kalman_smoother(data, F, H, Q, R, initx, initv, interpvals=0):
    os = H.shape[0]
    ss = F.shape[0]
    
    data = np.nan_to_num(data)
    interpolated_data = np.zeros_like(data)
    
    for c in range(os):
        interpolated_data[:,c] = interpolate(data[:,c], interpvals)
    y = interpolated_data
    
    xsmooth,Vsmooth = adskalman.kalman_smoother(y,F,H,Q,R,initx,initv)

    return xsmooth,Vsmooth
    
###
def smooth_centers(data, interpvals=0):
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
    R = 100*np.eye(os) # observation noise

    initx = np.array([data[0,0], data[0,1], data[1,0]-data[0,0], data[1,1]-data[0,1]],dtype=np.float)
    initv = 0*np.eye(ss)

    xsmooth,Vsmooth = kalman_smoother(data, F, H, Q, R, initx, initv, interpvals=interpvals)
    
    return xsmooth

###
def smooth_orientation_2d(ori, interpvals=0):
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
    Q[2,2] = .000001
    Q[3,3] = .000001
    R = np.eye(os) # observation noise

    initx = np.array([ori[0,0], ori[0,1], ori[1,0]-ori[0,0], ori[1,1]-ori[0,1]],dtype=np.float)
    initv = 0*np.eye(ss)

    xsmooth,Vsmooth = kalman_smoother(ori, F, H, Q, R, initx, initv, interpvals=interpvals)
    
    return xsmooth

###
def fix_orientation_2d(ori, vel, switching_threshold = 0.2, interpvals=0):

    interpolated_data = np.zeros_like(ori)
    for c in range(ori.shape[1]):
        interpolated_data[:,c] = interpolate(ori[:,c], interpvals)
    ori = interpolated_data

    # flies don't spin around immediately, so generally body angle should be rouhgly the same from frame to frame, at least within 180 deg
    # use this principal to fix the orientation vectors
    # switching threshold sets how fast a fly is allowed to fly/move backwards
    dot_prev_ori = np.zeros(len(ori))
    dot_vel = np.zeros(len(vel))
    
    smooth_ori = np.zeros_like(ori)
    smooth_ori[0,:] = ori[0,:]
    for i in range(1,len(ori)):
        
        if i < 1:
            switchingthreshold = 0
        else:
            switchingthreshold = switching_threshold         
    
        dot_prev_ori[i] = np.dot(smooth_ori[i-1], ori[i])
        dot_vel[i] = np.dot(vel[i], ori[i])
        
        direction = 1.

        if dot_vel[i] < 0 and np.abs(dot_vel[i]) > switchingthreshold:
            direction = -1
        else:
            if dot_prev_ori[i] < 0: # not aligned with previous frame by > 90 deg
                direction = -1
                
                if dot_vel[i] < 0: # orientation not aligned with velocity
                    if np.abs(dot_vel[i]) > switchingthreshold:
                        direction = -1
                if dot_vel[i] > 0: # orientation is aligned with velocity, but not with prev ori
                    if np.abs(dot_vel[i]) > switchingthreshold:
                        direction = 1
                    
                            
        smooth_ori[i] = ori[i]*direction
        
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
    while f <= len(movieinfo.frames):
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
def process_movieinfo(movieinfo):
    
    smooth_movieinfo(movieinfo)
    calc_fly_coordinates(movieinfo)
    calc_scale(movieinfo)
    calc_fly_coordinates_with_scale(movieinfo)
    
    
###
def smooth_movieinfo(movieinfo):
    
    xsmooth = smooth_centers(movieinfo.obj_centers, interpvals=0)
    movieinfo.smooth_centers = xsmooth[:,0:2]
    movieinfo.smooth_vel = xsmooth[:,2:4]

    fixed_ori = fix_orientation_2d(movieinfo.obj_longaxis, movieinfo.smooth_vel, switching_threshold = 0.2, interpvals=0)
    xsmooth = smooth_orientation_2d(fixed_ori, interpvals=0)
    movieinfo.smooth_ori = xsmooth[:,0:2]
    for i in range(len(movieinfo.smooth_ori)):
        movieinfo.smooth_ori[i] = movieinfo.smooth_ori[i] / np.linalg.norm( movieinfo.smooth_ori[i] )
        
    movieinfo.smooth_ori_vel = xsmooth[:,2:4]

###
def calc_fly_coordinates(movieinfo, post_pos = [512, 512]):

    class Flycoord:
        pass
        
    movieinfo.flycoord = Flycoord()
    post_pos = np.array(post_pos)
    
    movieinfo.flycoord.heading = np.zeros_like(movieinfo.smooth_centers)
    movieinfo.flycoord.angletopost = np.zeros([movieinfo.smooth_centers.shape[0], 1])
    movieinfo.flycoord.worldangle = np.zeros([movieinfo.smooth_centers.shape[0], 1])
    movieinfo.flycoord.slipangle = np.zeros([movieinfo.smooth_centers.shape[0], 1])
    movieinfo.flycoord.vel = np.zeros_like(movieinfo.smooth_centers)
    movieinfo.flycoord.vec_to_post = np.zeros_like(movieinfo.smooth_centers)
    movieinfo.flycoord.dist_to_post = np.zeros([movieinfo.smooth_centers.shape[0], 1])
    movieinfo.smooth_shortaxis = np.zeros_like(movieinfo.smooth_ori)
    
    for i in range(len(movieinfo.smooth_centers)):
        movieinfo.smooth_shortaxis[i,0] = movieinfo.smooth_ori[i,1]
        movieinfo.smooth_shortaxis[i,1] = movieinfo.smooth_ori[i,0]*-1
        
        movieinfo.flycoord.worldangle[i] = np.arctan2(movieinfo.smooth_ori[i,1], movieinfo.smooth_ori[i,0]) # want angle between 0 and 360 deg
        if movieinfo.flycoord.worldangle[i] < 0:
            movieinfo.flycoord.worldangle[i] = np.pi*2+movieinfo.flycoord.worldangle[i]
        movieinfo.flycoord.worldangle[i] = remove_angular_rollover( [movieinfo.flycoord.worldangle[i-1], movieinfo.flycoord.worldangle[i]], .5 )
            
        movieinfo.flycoord.heading[i] = movieinfo.smooth_vel[i] / np.linalg.norm( movieinfo.smooth_vel[i] )
        
        heading3vec = np.hstack( (movieinfo.flycoord.heading[i], [0]) )
        ori3vec = np.hstack( (movieinfo.smooth_ori[i], [0]) )
        
        sinslipangle = (np.cross( heading3vec, ori3vec ) / (np.linalg.norm(heading3vec)*np.linalg.norm(ori3vec))).sum()
        movieinfo.flycoord.slipangle[i] = np.arcsin(sinslipangle)
        
        vec_to_post = post_pos - movieinfo.smooth_centers[i]
        movieinfo.flycoord.dist_to_post[i] = np.linalg.norm(vec_to_post)
        movieinfo.flycoord.vec_to_post[i] = vec_to_post / np.linalg.norm(vec_to_post)
        
        cosangletopost = np.dot(vec_to_post, movieinfo.smooth_ori[i]) / ( np.linalg.norm(vec_to_post)*np.linalg.norm(movieinfo.smooth_ori[i]) )
        movieinfo.flycoord.angletopost[i] = np.arccos(cosangletopost)
            
    movieinfo.flycoord.vel = rotate_coordinates(movieinfo.smooth_vel, movieinfo.smooth_ori, movieinfo.smooth_shortaxis)
    movieinfo.flycoord.slipangle = movieinfo.flycoord.slipangle.reshape(len(movieinfo.flycoord.slipangle))
    
    movieinfo.trajec.angle_subtended_by_post = 2*np.arcsin( movieinfo.trajec.stimulus.radius / (movieinfo.trajec.dist_to_stim_r+movieinfo.trajec.stimulus.radius) ).reshape([movieinfo.trajec.dist_to_stim_r.shape[0],1])
    
    y = np.sin(movieinfo.flycoord.worldangle)
    x = np.cos(movieinfo.flycoord.worldangle)
    #print y.shape, x.shape, np.hstack( (x,y) ).shape, np.zeros([len(x),1]).shape
    smooth_ori_3vec = np.hstack( ( np.hstack( (x,y) ), np.zeros([len(x),1]) ) )
    vec_to_post_3vec = -1*np.hstack( (movieinfo.flycoord.vec_to_post, np.zeros([len(movieinfo.flycoord.vec_to_post),1]) ) ) # -1* is because essentially we just subtract the vector to the post from (0,0)
    
    signed_angle_to_post_smooth = np.sum(np.cross( vec_to_post_3vec, smooth_ori_3vec ), axis=1).reshape([vec_to_post_3vec.shape[0],1])
    mag_vec_to_post = np.array([ np.linalg.norm( vec_to_post_3vec[i,:] )  for i in range(vec_to_post_3vec.shape[0]) ]).reshape([vec_to_post_3vec.shape[0], 1])
    mag_ori = np.array([ np.linalg.norm( smooth_ori_3vec[i,:] )  for i in range(smooth_ori_3vec.shape[0]) ]).reshape([smooth_ori_3vec.shape[0], 1])
    sin_signed_angle_to_post = signed_angle_to_post_smooth / (mag_vec_to_post*mag_ori)
    
    movieinfo.flycoord.signed_angletopost = np.arcsin(sin_signed_angle_to_post)

###    
def calc_fly_coordinates_with_scale(movieinfo):

    class Scaled:
        pass
        
    post_pos = np.array([512, 512])
    scale = movieinfo.scale
    movieinfo.scaled = Scaled()

    movieinfo.scaled.dist_to_post = scale*movieinfo.flycoord.dist_to_post
    movieinfo.scaled.signed_angletopost = movieinfo.flycoord.signed_angletopost
    
    movieinfo.scaled.angle_subtended_by_post = 2*np.arcsin( movieinfo.trajec.stimulus.radius / (movieinfo.scaled.dist_to_post+movieinfo.trajec.stimulus.radius) ).reshape([movieinfo.scaled.dist_to_post.shape[0],1])
    
    movieinfo.scaled.positions = (movieinfo.smooth_centers-post_pos)*movieinfo.scale
    movieinfo.scaled.velocities = movieinfo.smooth_vel*movieinfo.scale
    
    movieinfo.scaled.angle_to_edge = np.abs(movieinfo.scaled.signed_angletopost)-np.abs(movieinfo.scaled.angle_subtended_by_post)/2.
    movieinfo.scaled.angle_to_lower_edge = (np.abs(movieinfo.scaled.signed_angletopost) - movieinfo.scaled.angle_subtended_by_post/2.).reshape([len(movieinfo.scaled.signed_angletopost)])
    movieinfo.scaled.angle_to_upper_edge = (np.abs(movieinfo.scaled.signed_angletopost) + movieinfo.scaled.angle_subtended_by_post/2.).reshape([len(movieinfo.scaled.signed_angletopost)])
    
    
    
###
def calc_scale(movieinfo, plot=False):
    nt, flydra_dist = interpolate_to_new_framerate(movieinfo, 200, movieinfo.trajec.epoch_time, movieinfo.trajec.dist_to_stim_r)
    nt, sa1_dist = interpolate_to_new_framerate(movieinfo, 200, movieinfo.timestamps, movieinfo.flycoord.dist_to_post)
    
    if movieinfo.behavior == 'landing': # ignore last bit
        frames_away_from_post = np.where(flydra_dist > .007)[0]
        flydra_dist = flydra_dist[frames_away_from_post]
        sa1_dist = sa1_dist[frames_away_from_post]
        nt = nt[frames_away_from_post]
        
    sa1_dist -= np.min(sa1_dist)
    flydra_dist -= np.min(flydra_dist)
    
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
    data = data.reshape(len(data))
    new_data = np.interp(new_timestamps, old_timestamps, data)
    return new_timestamps, new_data
    

        
    
    
    
