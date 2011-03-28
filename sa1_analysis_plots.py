import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
import matplotlib.patches as patches
import matplotlib.backends.backend_pdf as pdf

import sys
sys.path.append('/home/floris/src/analysis')

import sa1_analysis as sa1 
import colorline

def plot_expansion(movieinfo, ax=None, figure=None, plot_dist=True):

    if ax is None:
        fig = plt.figure(figure)
        ax = fig.add_axes([.1, .1, .8, .8])
        
    slipangle = movieinfo.scaled.slipangle
    upper = movieinfo.scaled.angle_to_upper_edge
    lower = movieinfo.scaled.angle_to_lower_edge
    
    frames = sa1.get_frames_until_landing(movieinfo)
    time_nearest_to_post = sa1.get_time_nearest_to_post(movieinfo)
    #print time_nearest_to_post
    
    
    time = movieinfo.timestamps - time_nearest_to_post
    
    ax.fill_between(time[frames], lower[frames], upper[frames])
    
    t = np.array([time[frames][0], time[frames][-1]])
    plt.plot(t, np.zeros_like(t), linewidth=3, color='black')
    
    # legs
    if 1: #movieinfo.legextensionrange is not None:
        last_timestamp = movieinfo.timestamps[frames[-1]]
        legextensiontimerange = sa1.frame_to_timestamp(movieinfo, movieinfo.legextensionrange)
        legextensiontimerange[1] = np.min([legextensiontimerange[1], last_timestamp])
        plt.plot(legextensiontimerange-time_nearest_to_post, np.zeros_like(legextensiontimerange), linewidth=3, color='red')
    
    cl = colorline.Colorline(colormap='jet', norm = None, hide_colorbar=True, ax0=ax)
    cl.colorline(time[frames], -1*slipangle[frames], movieinfo.scaled.speed[frames])
    
    if plot_dist:
        dist = movieinfo.scaled.dist_head_to_post[frames]-movieinfo.post_radius
        scalefactor = 50#2/np.max(dist)
        plt.plot(time[frames], dist*scalefactor, color='black')
    
    ax.vlines(0, -np.pi, np.pi, color='r', linestyle='--')
    return ax
    
def plot_expansion_for_dataset(movie_dataset, figure=None, behavior=None, posttype=None, crash=False):  
    fig = plt.figure(figure)
    
    keys = movie_dataset.get_movie_keys(behavior=behavior, posttype=posttype)
    
    if behavior == 'landing':
        if crash is False:
            filtered_keys = []
            for key in keys:
                if 'crash' in movie_dataset.movies[key].subbehavior or 'wingcrash'  in movie_dataset.movies[key].subbehavior:
                    pass
                else:
                    filtered_keys.append(key)
        else:
            filtered_keys = []
            for key in keys:
                if 'crash' in movie_dataset.movies[key].subbehavior or 'wingcrash'  in movie_dataset.movies[key].subbehavior:
                    filtered_keys.append(key)
                else:
                    pass
                    
        keys = filtered_keys
            
    
    
    nmovies = len(keys)+1
    ncols = 2
    nrows = int(np.ceil(nmovies/2))
    
    n = 0
    print keys
    for key in keys:
        print key
        n += 1
        movie = movie_dataset.movies[key]
        ax = fig.add_subplot(nrows,ncols,n)
        ax = plot_expansion(movie, ax, figure=fig)
        if behavior != 'landing':
            ax.set_xlim([-.5,.5])
        else:
            ax.set_xlim([-.5, 0])
        ax.set_ylim([-2, 2])
        
        ax.set_yticklabels([])
        #ax.set_xticklabels([])
        title = movie.id + movie.subbehavior[0]
        ax.set_title(title)
    
def plot_trajectory(movieinfo, figure=None):
    fig = plt.figure(figure)
    ax0 = fig.add_axes([.1,.1,.8,.8])
    post_pos = movieinfo.post_pos
    frames = sa1.get_frames_until_landing(movieinfo)
    
    ax0.plot(movieinfo.scaled.positions[frames,0], movieinfo.scaled.positions[frames,1])
    ax0.plot(movieinfo.trajec.positions[:,0], movieinfo.trajec.positions[:,1], '.')
    
    ax0.set_xlim([-.1, .1])
    ax0.set_ylim([-.1, .1])
    ax0.set_aspect('equal')
    
    strobe = sa1.strobe_from_movieinfo(movieinfo, interval=200)
    ax0.imshow(strobe.T, plt.get_cmap('gray'), origin='lower', extent = [-512*movieinfo.scale, 512*movieinfo.scale, -512*movieinfo.scale, 512*movieinfo.scale])
    
    # plot body orientation vector
    interval = 50
    i = 0
    while i < frames[-1]:
        ax0.plot(movieinfo.scaled.head_pos[i,0],movieinfo.scaled.head_pos[i,1], '.', color='black', zorder=10)
        center = movieinfo.scaled.positions[i]
        long_axis = movieinfo.smooth_ori[i]
        
        factor = .001
                
        dx = long_axis[0]*factor
        dy = long_axis[1]*factor
        
        arrow = Arrow(center[0], center[1], dx, dy, width=.0001, color='red')
        ax0.add_artist(arrow)
        
        i += interval        
        
    moviecenter = np.array([512,512])
    c = -1*(moviecenter-post_pos)*movieinfo.scale
    circle = patches.Circle( (c[0], c[1]), radius=movieinfo.post_radius, facecolor='none', edgecolor='green')
    ax0.add_artist(circle)
    
    title = movieinfo.id + ' ' + movieinfo.behavior + ' ' + movieinfo.subbehavior[0]
    ax0.set_title(title)
    
    return ax0

###############################################################################
# PDF's
###############################################################################

def pdf_trajectory_plots(movie_dataset, filename='sa1_trajectory_plots.pdf', behavior=None, posttype=None, scale=10):
    
    plt.close('all')
    pp =  pdf.PdfPages(filename)
    keys = movie_dataset.get_movie_keys(behavior=behavior, posttype=posttype)
    
    for f, key in enumerate(keys):
        movieinfo = movie_dataset.movies[key]
        if not np.isnan(movieinfo.scale):
            print 'plotting: ', key
            # As many times as you like, create a figure fig, then either:
            ax0 = plot_trajectory(movieinfo, figure=f)
            
            plt.Figure.set_figsize_inches(ax0.figure, [2*scale,1*scale])
            plt.Figure.set_dpi(ax0.figure, 72)
            
            pp.savefig(f)
            plt.close(f)
    
    # Once you are done, remember to close the object:
    pp.close()
    print 'closed'
    
###############################################################################
# Rarely Used (one time use?) Functions 
###############################################################################
    
###
def plot_post_centers(movie_dataset):
    
    keys = movie_dataset.get_movie_keys()
    centers = np.zeros([len(keys), 2])
    
    for i, key in enumerate(keys):
        movieinfo = movie_dataset.movies[key]
        centers[i, :] = movieinfo.post_pos
        
    plt.plot(centers[:,0])
    plt.plot(centers[:,1])
    
def plot_post_radii(movie_dataset):
    
    keys = movie_dataset.get_movie_keys()
    radii = np.zeros([len(keys), 1])
    
    for i, key in enumerate(keys):
        movieinfo = movie_dataset.movies[key]
        radii[i] = movieinfo.post_radius_in_pixels
        
    plt.plot(radii[:])
    
    
    
