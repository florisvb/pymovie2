import numpy as np
import copy

import sys
sys.path.insert(0, '/usr/local/lib/python2.6/dist-packages')
import matplotlib
print 'using matplotlib version: ', matplotlib.__version__

import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
import matplotlib.patches as patches
import matplotlib.backends.backend_pdf as pdf
from matplotlib.colors import colorConverter
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('/home/floris/src/analysis')

import sa1_analysis as sa1 
import colorline
import numpyimgproc as nim

###########################################################################
#                               Functions                                 #
###########################################################################

def adjust_spines(ax,spines, color='black', spine_locations={}):
    if type(spines) is not list:
        spines = [spines]
    spine_locations_dict = {'top': 10, 'right': 10, 'left': 10, 'bottom': 10}
    for key in spine_locations.keys():
        spine_locations_dict[key] = spine_locations[key]
        
    if 'none' in spines:
        for loc, spine in ax.spines.iteritems():
            spine.set_color('none') # don't draw spine
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        return
    
    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            spine.set_position(('outward',spine_locations_dict[loc])) # outward by x points
            spine.set_color(color)
            #spine.set_smart_bounds(True)
            if loc == 'bottom':
                spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
        
        
def plot_remove_jumps(ax, x, y, thresh=[1,1], linestyle='-', color='blue', linewidth=1):
    
    if ax is None:
        fig = plt.figure(None)
        ax = fig.add_axes([.1,.1,.8,.8])
    
    dx = np.abs(np.diff(x))
    jump_indices_x = np.where(dx >= thresh[0])[0]
    
    dy = np.abs(np.diff(y))
    jump_indices_y = np.where(dy >= thresh[1])[0]
    
    jump_indices = np.hstack(( jump_indices_x, jump_indices_y ))
    jump_indices = np.sort(jump_indices).tolist()
    jump_indices.insert(0,-1)
    jump_indices.append(-1)
    
    for i in range(1,len(jump_indices)):
        ind = [jump_indices[i-1]+1, jump_indices[i]]
        ax.plot(x[ind[0]:ind[1]], y[ind[0]:ind[1]], linestyle=linestyle, color=color, linewidth=linewidth)
        

def plot_show_backwards_flight(ax, x, y, linestyle=['-', '--'], color=['gray', 'gray'], thresh=[1,1], linewidth=1):

    forwards = np.where( nim.threshold(y, -np.pi/2., np.pi/2.) == True)[0].tolist()
    backwards = np.where( nim.threshold(y, -np.pi/2., np.pi/2.) == False)[0].tolist()
    
    ybackwards = -1*copy.copy(y)
    ybackwards[ np.where(ybackwards < -np.pi/2.)[0].tolist() ] += np.pi
    ybackwards[ np.where(ybackwards >  np.pi/2.)[0].tolist() ] -= np.pi
    
    plot_remove_jumps(ax, x[forwards], y[forwards], thresh=thresh, linestyle=linestyle[0], color=color[0], linewidth=linewidth)
    plot_remove_jumps(ax, x[backwards], ybackwards[backwards], thresh=thresh, linestyle=linestyle[1], color=color[1], linewidth=linewidth)     
        
###########################################################################
#                               Plotting                                 
###########################################################################

def plot_expansion(movieinfo, ax=None, figure=None, plot_dist=True, plot_altitude=True, plot_slipangle=True, show_dist_spine=True, show_time_spine=True, show_angle_spine=True, show_colorbar_spine=False, flip=False, label_lines=True, print_post_tags=False, label_font_size=9):
    frames = sa1.get_frames_until_landing(movieinfo)
    
    ## SET UP ALL AXES ##
    if ax is None:
        fig = plt.figure(figure)
        ax = fig.add_axes([.1, .1, .8, .8])
    ax.set_axis_bgcolor('white')
    if 1:
        divider = make_axes_locatable(ax)
        divider.set_anchor('E')
        cax = divider.append_axes("right", size="2%", pad=0.)
    if plot_dist:
        distax = ax.twinx()
    if plot_altitude:
        altax = ax.twinx()
    ## DONE WITH AXES  ##
    
    if flip:
        mean_angletopost = np.mean(movieinfo.scaled.signed_angletopost[frames])
        if mean_angletopost < 0: flip = -1
        else: flip = 1
    else:
        flip = 1
    if label_lines: label_x_offset = 0.01

    # simple variable names        
    slipangle = movieinfo.flycoord.slipangle*flip
    upper = movieinfo.scaled.angle_to_upper_edge*flip
    lower = movieinfo.scaled.angle_to_lower_edge*flip
    time_nearest_to_post = sa1.get_time_nearest_to_post(movieinfo)
    frame_nearest_to_post = sa1.get_frame_from_timestamp(movieinfo, time_nearest_to_post)
    time = movieinfo.timestamps - time_nearest_to_post
    
    # plot FILL BETWEEN
    ax.fill_between(time[frames], lower[frames], upper[frames], facecolor='black')
    
    # plot LEG EXTENSION
    if movieinfo.legextensionrange is not None:
        last_timestamp = movieinfo.timestamps[frames[-1]]
        legextensiontimerange = sa1.frame_to_timestamp(movieinfo, movieinfo.legextensionrange)
        legextensiontimerange[1] = np.min([legextensiontimerange[1], last_timestamp])
        ax.plot(legextensiontimerange-time_nearest_to_post, np.zeros_like(legextensiontimerange), linewidth=2, color='red')

    # plot SLIP ANGLE    
    if plot_slipangle:
        plot_show_backwards_flight(ax, time[frames], slipangle[frames], linestyle=['-', '--'], color=['gray', 'gray'], thresh=[2/float(movieinfo.framerate), np.pi/2], linewidth=2)
    
    # plot COLOR DISTANCE SPEED
    if plot_dist:
        dist = movieinfo.scaled.dist_head_to_post[frames]-movieinfo.post_radius
        clim = (0,0.3)
        cl = colorline.Colorline(colormap='jet', norm=clim, hide_colorbar=True, ax0=distax)
        cl.colorline(time[frames], dist, movieinfo.scaled.speed[frames], linewidth=2, norm=clim)
        distax.set_frame_on(True)
        distax.set_axis_bgcolor('none')
        distylim = [0, .15]
        distax.set_ylim(distylim) 
        if show_dist_spine:
            distyticks = np.linspace(distylim[0], distylim[-1], 4, endpoint=True)
            adjust_spines(distax,['right'], color='green', spine_locations={'right': 0})
            dround = [np.int(np.round(distyticks[i]*(10**3))) for i in range(len(distyticks))]
            s = [str(dround[i]) for i in range(len(dround))]
            distax.set_yticks(distyticks)
            distax.set_yticklabels(s)
            distax.set_ylabel('dist scale, mm')
        else:
            if show_time_spine:
                adjust_spines(distax,['bottom'], spine_locations={'bottom': 20})
            else:
                adjust_spines(distax,['none'])
        if label_lines:
            norm = plt.Normalize(0,.3)
            cmap = plt.get_cmap('jet')
            color = cmap( norm(movieinfo.scaled.speed[frames[-1]]) )
            while np.sum(color[0:3]) > 1.5: # make darker for readability
                color = [i/2. for i in color]
                color[-1] = 1.0
            distax.text(time[frames][-1]+label_x_offset, dist[-1], 'dist to post,', fontdict={'fontsize': label_font_size, 'color': color}, withdash=False, horizontalalignment='left', verticalalignment='bottom')
            distax.text(time[frames][-1]+label_x_offset, dist[-1], 'color: speed', fontdict={'fontsize': label_font_size, 'color': color}, withdash=False, horizontalalignment='left', verticalalignment='top')
            
    # plot ALTITUDE
    if plot_altitude:
        nt, flydra_altitude = sa1.interpolate_to_new_framerate(movieinfo, 200, movieinfo.trajec.epoch_time, movieinfo.trajec.positions[:,2])
        nt -= time_nearest_to_post
        altax.plot(nt, flydra_altitude, color='green', linewidth=2)
        altax.set_frame_on(True)
        altax.set_axis_bgcolor('none')
        altax.set_ylim(-.15,0)
        if show_time_spine:
            adjust_spines(altax,['bottom'], spine_locations={'bottom': 20})
        else:
            adjust_spines(altax,['none'])
        if label_lines:
            altax.text(time[frames][-1]+label_x_offset, flydra_altitude[-1], 'altitude', fontdict={'fontsize': label_font_size, 'color': 'green'}, withdash=False, horizontalalignment='left', verticalalignment='center')
        
    # plot GUIDE LINES
    ax.vlines(0, -np.pi, np.pi, color='black', linestyle='-')
    ax.hlines(0, -0.5, .5, color='red', linestyle='dotted')
    ax.hlines(-np.pi/2., -0.5, 0.5, color='gray', linestyle='dotted', linewidth=2)
    ax.hlines(np.pi/2., -0.5, 0.5, color='gray', linestyle='dotted')
    ax.hlines(np.pi, -0.5, 0.5, color='gray', linestyle='dotted', linewidth=2)
    
    # plot POST EDGE GUIDE LINES
    norm = plt.Normalize(0,.3)
    cmap = plt.get_cmap('jet')
    color = cmap( norm(movieinfo.scaled.speed[frame_nearest_to_post]) )
    while np.sum(color[0:3]) > 1.5: # make darker for readability
        color = [i/2. for i in color]
        color[-1] = 1.0
    ax.hlines(-np.pi/2., -0.03, 0.03, color=color, linestyle='-', zorder=100, linewidth=4)
    # post top
    ax.hlines(np.pi, -0.03, 0.03, color='green', linestyle='-', zorder=100, linewidth=4)
    
    if 'crash' in movieinfo.subbehavior or 'wingcrash' in movieinfo.subbehavior:
        el = patches.Ellipse( (0,0), .02, .2, facecolor='red', edgecolor='red', zorder=20)
        ax.add_artist(el)
    
    # primary AX parameters
    axylim = [-np.pi/2., np.pi]
    ax.set_ylim(axylim)
    if movieinfo.behavior == 'landing':
        axxlim = [-.5, 0]
    else:
        axxlim = [-.5, .5]
    ax.set_xlim(axxlim) 
    
    if show_angle_spine:
        spines = ['left']
    else:
        spines = []
    if show_time_spine:
        spines.append('bottom')
    adjust_spines(ax,spines, spine_locations={'bottom': 20})
    if show_time_spine:
        xticks = np.linspace(axxlim[0], axxlim[-1], 5, endpoint=True)
        ax.set_xticks(xticks)
        ax.set_xlabel('time, sec')
    if show_angle_spine:
        yticks = [-np.pi/2., 0, np.pi/2, np.pi]
        ax.set_yticks(yticks)
        ax.set_yticklabels([-90, 0, 90, 180])
        ax.set_ylabel('angle on retina, deg')
        
    # print LABELS / LEGEND
    if label_lines:
        y_vis = lower[-1] + (upper[-1] - lower[-1])/2.
        # post img
        ax.text(time[frames][-1]+label_x_offset, y_vis, '2D post img', fontdict={'fontsize': label_font_size, 'color': 'black'}, withdash=False, horizontalalignment='left', verticalalignment='bottom')
        ax.text(time[frames][-1]+label_x_offset, y_vis, 'on flies retina', fontdict={'fontsize': label_font_size, 'color': 'black'}, withdash=False, horizontalalignment='left', verticalalignment='top')
        # slipangle
        ax.text(time[frames][0]-label_x_offset, slipangle[0], 'flight angle rel.', fontdict={'fontsize': label_font_size, 'color': 'gray'}, withdash=False, horizontalalignment='right', verticalalignment='bottom')
        ax.text(time[frames][0]-label_x_offset, slipangle[0], 'to orientation', fontdict={'fontsize': label_font_size, 'color': 'gray'}, withdash=False, horizontalalignment='right', verticalalignment='top')
    
    if 1: # COLORBAR
        cticks = [0, .1, .2, .3]
        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('jet'), norm=plt.Normalize(clim[0], clim[1]), orientation='vertical', boundaries=None, ticks=cticks, drawedges=False)
        cax.set_ylabel('speed, m/s')
        
        if show_colorbar_spine:
            cax.set_visible(True)
        else:
            cax.set_visible(False)
    
    # print POST TAGS
    if print_post_tags:
            norm = plt.Normalize(0,.3)
            cmap = plt.get_cmap('jet')
            color = cmap( norm(movieinfo.scaled.speed[frame_nearest_to_post]) )
            while np.sum(color[0:3]) > 1.5: # make darker for readability
                color = [i/2. for i in color]
                color[-1] = 1.0
            ax.text(0, -np.pi/2., 'post edge', fontdict={'fontsize': label_font_size, 'color': color}, withdash=False, horizontalalignment='center', verticalalignment='top')
            ax.text(0, np.pi, 'post top', fontdict={'fontsize': label_font_size, 'color': 'green'}, withdash=False, horizontalalignment='center')
        
    # package axes and return
    axes = [ax]
    if plot_dist:
        axes.append(distax)
    if plot_altitude:
        axes.append(altax)
    return axes

###
def plot_expansion_for_dataset(movie_dataset, figure=None, behavior=None, posttype=None, crash=False, nmovies=None, firstmovie=0):  
    fig = plt.figure(figure)
    fig.set_facecolor('white')
    
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
            
    nkeys = len(keys)
    if nmovies is None:
        nmovies = nkeys
    elif nmovies > nkeys:
        nmovies = nkeys
    
    ncols = 3
    nrows = int(np.ceil( (nmovies)/float(ncols)))
    left_side_panel = np.arange(1,nmovies+1,ncols)
    right_side_panel = np.arange(ncols,nmovies+2,ncols)
    
    n = 0
    print keys
    label = True
    for n in range(nmovies):
        key = keys[n+firstmovie]
        print key
        n += 1
        movie = movie_dataset.movies[key]
        ax = fig.add_subplot(nrows,ncols,n)
        
        if n==right_side_panel[-1]:
            axes = plot_expansion(movie, ax, figure=fig, show_time_spine=True, show_angle_spine=False, show_dist_spine=True, flip=True, show_colorbar_spine=False, label_lines=True, print_post_tags=True)
            flycon = plt.imread('/home/floris/src/pymovie2/flycon_topview_flight.png')
            axes[0].imshow(flycon, extent=[-.5,-.35,-.6,.6], aspect='auto', zorder=200)
        elif n==right_side_panel[-2]:
            axes = plot_expansion(movie, ax, figure=fig, show_time_spine=False, show_angle_spine=False, show_dist_spine=False, show_colorbar_spine=True, flip=True, label_lines=False)
        elif n==left_side_panel[-1]:
            axes = plot_expansion(movie, ax, figure=fig, show_time_spine=False, show_angle_spine=True, show_dist_spine=False, flip=True, label_lines=False)
        else:
            axes = plot_expansion(movie, ax, figure=fig, show_time_spine=False, show_angle_spine=False, show_dist_spine=False, flip=True, label_lines=False)
        axes[0].text(-.5, -np.pi/2, movie.id, fontdict={'fontsize': 8}, withdash=False)
            
    plt.draw()
    plt.show()
    return fig
    
def plot_trajectory(movieinfo, figure=None):
    fig = plt.figure(figure)
    ax0 = fig.add_axes([.1,.1,.8,.8])
    post_pos = movieinfo.post_pos
    frames = sa1.get_frames_until_landing(movieinfo)
    
    ax0.plot(movieinfo.scaled.positions[frames,0], movieinfo.scaled.positions[frames,1])
    ax0.plot(movieinfo.trajec.positions[:,0], movieinfo.trajec.positions[:,1], '.', markersize=2)
    
    ax0.set_xlim([-.1, .1])
    ax0.set_ylim([-.1, .1])
    ax0.set_aspect('equal')
    
    strobe = sa1.strobe_from_movieinfo(movieinfo, interval=200)
    ax0.imshow(strobe.T, plt.get_cmap('gray'), origin='lower', extent = [-1*movieinfo.post_pos[0]*movieinfo.scale, (1024-movieinfo.post_pos[0])*movieinfo.scale, -1*movieinfo.post_pos[1]*movieinfo.scale, (1024-movieinfo.post_pos[1])*movieinfo.scale])
    
    # plot body orientation vector
    interval = 50
    i = 0
    while i < frames[-1]:
        ax0.plot(movieinfo.scaled.head_pos[i,0],movieinfo.scaled.head_pos[i,1], '.', color='black', zorder=10, markersize=2)
        center = movieinfo.scaled.positions[i]
        long_axis = movieinfo.smooth_ori[i]
        
        factor = .001
                
        dx = long_axis[0]*factor
        dy = long_axis[1]*factor
        
        arrow = Arrow(center[0], center[1], dx, dy, width=.0001, color='red')
        ax0.add_artist(arrow)
        
        i += interval        
        
    circle = patches.Circle( (0,0), radius=movieinfo.post_radius, facecolor='none', edgecolor='green')
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
    
    
    
