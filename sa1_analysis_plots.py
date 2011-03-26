import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Arrow

import sa1_analysis as sa1 

def plot_expansion(movieinfo):
    fig = plt.figure()
    ax1 = fig.add_axes([.1, .1, .8, .8])
    
    upper = movieinfo.scaled.angle_to_upper_edge      
    lower = movieinfo.scaled.angle_to_lower_edge   
    time = movieinfo.timestamps    
    ax1.fill_between(time, upper, lower)
    
    if movieinfo.landingframe is not None:
        ax1.vlines(movieinfo.landingtime, -2, 2, color='r')
        pass
    #ax1.hlines(0, -1, 1, color='r')
    
def plot_trajectory(movieinfo):
    fig = plt.figure(None)
    ax0 = fig.add_axes([.1,.1,.8,.8])
    post_pos = np.array([512,512])
    ax0.plot(movieinfo.scaled.positions[:,0], movieinfo.scaled.positions[:,1])
    ax0.plot(movieinfo.trajec.positions[:,0], movieinfo.trajec.positions[:,1], '.')
    
    ax0.set_xlim([-.1, .1])
    ax0.set_ylim([-.1, .1])
    ax0.set_aspect('equal')
    
    strobe = sa1.strobe_from_movieinfo(movieinfo, interval=200)
    ax0.imshow(strobe.T, plt.get_cmap('gray'), origin='lower', extent = [-512*movieinfo.scale, 512*movieinfo.scale, -512*movieinfo.scale, 512*movieinfo.scale])
    
    # plot body orientation vector
    interval = 50
    i = 0
    while i < len(movieinfo.framenumbers):
        center = movieinfo.scaled.positions[i]
        long_axis = movieinfo.smooth_ori[i]
        
        factor = .001
                
        dx = long_axis[0]*factor
        dy = long_axis[1]*factor
        
        arrow = Arrow(center[0], center[1], dx, dy, width=.0001, color='red')
        ax0.add_artist(arrow)
        
        i += interval        
        
    
    
    return ax0
