import numpy as np
import copy

import sys
#sys.path.insert(0, '/usr/local/lib/python2.6/dist-packages')
import matplotlib
print 'using matplotlib version: ', matplotlib.__version__

import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
import matplotlib.patches as patches
import matplotlib.backends.backend_pdf as pdf
from matplotlib.colors import colorConverter
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mpl_toolkits.axes_grid1 import ImageGrid

sys.path.append('/home/floris/src/analysis')
import analysis_plot as ap

import sa1_analysis as sa1 
import colorline
import numpyimgproc as nim
import sa1_analysis_classification as sac

###########################################################################
#                               Functions                                 #
###########################################################################

def adjust_spines(ax,spines, color='black', spine_locations={}, smart_bounds=True):
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
            if loc == 'bottom' and smart_bounds:
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

def plot_expansion(movieinfo, ax=None, figure=None, plot_dist=True, plot_altitude=True, plot_slipangle=True, plot_angular_vel=False, plot_classification=False, show_dist_spine=True, show_time_spine=True, show_angle_spine='left', show_colorbar_spine=False, flip=False, label_lines=True, print_post_tags=False, label_font_size=9, lollipop=False, show_lollipop_spines=None, flycon=None, movie_id_text=True, time_limits=None, plot_post_edges=True, linewidth=2, align_method='time'):
    print time_limits
    frames = sa1.get_frames_until_landing(movieinfo)
    try: fake = movieinfo.fake
    except: fake = False
    
    if flycon is True:
        flycon='/home/floris/src/pymovie2/flycon_topview_flight.png'
    
    ## SET UP ALL AXES ##
    if ax is None:
        fig = plt.figure(figure)
        fig.set_facecolor('white')
        ax = fig.add_axes([.1, .1, .8, .8])
    ax.set_axis_bgcolor('white')
    if 1:
        colorbar_pad = 0
        colorbar_size = "3%"
        divider = make_axes_locatable(ax)
        divider.set_anchor('E')
        cax = divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
    if plot_classification:
        class_pad = 0.1
        class_size = "8%"
        classax = divider.append_axes("top", size=class_size, pad=class_pad)
    if lollipop:
        if show_angle_spine == 'left':
            print 'angle spine on left: '
            extra_loli_pad = 1
        else: extra_loli_pad = 0.
        divider.set_anchor('E')
        loli_pad = 0.1 + extra_loli_pad
        loli_size = "25%"
        loliax = divider.append_axes("left", size=loli_size, pad=loli_pad)
    if 1:
        distax = ax.twinx()
        divider = make_axes_locatable(distax)
        divider.set_anchor('E')
        distcax = divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
        distcax.set_visible(False)
        if lollipop:
            distloliax = divider.append_axes("left", size=loli_size, pad=loli_pad)
            distloliax.set_visible(False)
        if plot_classification:
            distclassax = divider.append_axes("top", size=class_size, pad=class_pad)
            distclassax.set_visible(False)
    if plot_altitude:
        altax = ax.twinx()
        divider = make_axes_locatable(altax)
        divider.set_anchor('E')
        altcax = divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
        altcax.set_visible(False)
        if lollipop:
            altloliax = divider.append_axes("left", size=loli_size, pad=loli_pad)
            altloliax.set_visible(False)
        if plot_classification:
            altclassax = divider.append_axes("top", size=class_size, pad=class_pad)
            altclassax.set_visible(False)
        
    ## DONE WITH AXES  ##
    if label_lines: label_x_offset = 0.01
    if flip:
        mean_angletopost = np.mean(movieinfo.scaled.signed_angletopost[frames])
        if mean_angletopost < 0: flip = -1
        else: flip = 1
    else:
        flip = 1

    # Lollipops
    if lollipop:
        plot_lollipops(movieinfo, ax=loliax, figure=None, scaletofit=True, show_spines=show_lollipop_spines, interval=200, flip=flip)
        loliax.set_axis_bgcolor('none')
    
    # Classification:
    if plot_classification:
        sac.classify(movieinfo)
        classaxes = sac.plot_classified(movieinfo, ax=classax)
    

    # simple variable names        
    slipangle = movieinfo.flycoord.slipangle*flip
    upper = movieinfo.scaled.angle_to_upper_edge*flip
    lower = movieinfo.scaled.angle_to_lower_edge*flip
    time_nearest_to_post = sa1.get_time_nearest_to_post(movieinfo)
    frame_nearest_to_post = sa1.get_frame_from_timestamp(movieinfo, time_nearest_to_post)
    
    if align_method == 'time':
        align_time = time_nearest_to_post
        align_frame = frame_nearest_to_post
        
    if align_method == 'state change':
        try:
            align_frame = np.where(movieinfo.deviation_from_initial_state > 0.5)[0][0]
        except:
            align_frame = frames[-1]
        align_time = movieinfo.timestamps[align_frame]
        
    time = movieinfo.timestamps - align_time
    
    # plot FILL BETWEEN
    ax.fill_between(time[frames], lower[frames], upper[frames], facecolor='black')
    
    # plot LEG EXTENSION
    if movieinfo.legextensionrange is not None:
        last_timestamp = movieinfo.timestamps[frames[-1]]
        legextensiontimerange = sa1.frame_to_timestamp(movieinfo, movieinfo.legextensionrange)
        legextensiontimerange[1] = np.min([legextensiontimerange[1], last_timestamp])
        ax.plot(legextensiontimerange-align_time, np.zeros_like(legextensiontimerange), linewidth=linewidth, color='red')

    # plot SLIP ANGLE    
    if plot_slipangle:
        plot_show_backwards_flight(ax, time[frames], slipangle[frames], linestyle=['-', '--'], color=['gray', 'gray'], thresh=[2/float(movieinfo.framerate), np.pi/2], linewidth=linewidth)
        #ax.plot(time[frames], movieinfo.deviation_from_initial_state[frames], 'blue')
        #ax.plot(movieinfo.trajec.epoch_time-align_time, movieinfo.trajec.integrated_expansion)
        
    # plot ANGULAR VELOCITY
    if plot_angular_vel:
        tmp = np.abs(movieinfo.flycoord.worldangle_vel[frames])
        tmp /= 20.
        #ax.plot(time[frames], tmp, color='blue')
    
    # plot COLOR DISTANCE SPEED
    if plot_dist:
        dist = movieinfo.scaled.dist_head_to_post[frames]-movieinfo.post_radius
        clim = (0,0.3)
        cl = colorline.Colorline(colormap='jet', norm=clim, hide_colorbar=True, ax0=distax)
        cl.colorline(time[frames], dist, movieinfo.scaled.speed[frames], linewidth=linewidth, norm=clim)
        if label_lines:
            norm = plt.Normalize(0,.3)
            cmap = plt.get_cmap('jet')
            color = cmap( norm(movieinfo.scaled.speed[frames[-1]]) )
            while np.sum(color[0:3]) > 1.5: # make darker for readability
                color = [i/2. for i in color]
                color[-1] = 1.0
            distax.text(time[frames][-1]+label_x_offset, dist[frames][-1], 'dist to post,', fontdict={'fontsize': label_font_size, 'color': color}, withdash=False, horizontalalignment='left', verticalalignment='bottom')
            distax.text(time[frames][-1]+label_x_offset, dist[frames][-1], 'color: speed', fontdict={'fontsize': label_font_size, 'color': color}, withdash=False, horizontalalignment='left', verticalalignment='top')
    
    distax.set_frame_on(True)
    distax.set_axis_bgcolor('none')
    distylim = [0, .15]
    distax.set_ylim(distylim)
    
    if show_dist_spine and not fake and (plot_dist or plot_altitude):
        distyticks = np.linspace(distylim[0], distylim[-1], 4, endpoint=True)
        adjust_spines(distax,['right'], color='green', spine_locations={'right': 5})
        dround = [np.int(np.round(distyticks[i]*(10**3))) for i in range(len(distyticks))]
        s = [str(dround[i]) for i in range(len(dround))]
        distax.set_yticks(distyticks)
        distax.set_yticklabels([])
        distax.set_yticklabels(s, color='green', fontsize=label_font_size)
        distax.set_ylabel('dist scale, mm', fontdict={'fontsize': label_font_size, 'color': 'green'})
    else:
        if show_time_spine:
            adjust_spines(distax,['none'])#adjust_spines(distax,['bottom'])#, spine_locations={'bottom': 5})
        else:
            adjust_spines(distax,['none'])
    # plot ALTITUDE
    if plot_altitude and not fake:
        nt, flydra_altitude = sa1.interpolate_to_new_framerate(movieinfo, 200, movieinfo.trajec.epoch_time, movieinfo.trajec.positions[:,2])
        
        if movieinfo.behavior == 'landing':
            flydra_frame_of_landing = sa1.get_frame_from_timestamp(movieinfo, time_nearest_to_post, nt)
        else: flydra_frame_of_landing = -1
        nt -= align_time
        
        altax.plot(nt[:flydra_frame_of_landing], flydra_altitude[:flydra_frame_of_landing], color='green', linewidth=linewidth)
        altax.set_frame_on(True)
        altax.set_axis_bgcolor('none')
        altax.set_ylim(-.15,0)
        if show_time_spine:
            adjust_spines(altax,['none'])#adjust_spines(altax,['bottom'])#, spine_locations={'bottom': 5})
        else:
            adjust_spines(altax,['none'])
        if label_lines:
            altax.text(nt[0]-label_x_offset, flydra_altitude[flydra_frame_of_landing], 'altitude', fontdict={'fontsize': label_font_size, 'color': 'green'}, withdash=False, horizontalalignment='right', verticalalignment='center')
    if fake:
        adjust_spines(altax,['none'])
        
    # plot GUIDE LINES
    ax.vlines(0, -np.pi, np.pi, color='black', linestyle='-')
    ax.hlines(0, -0.5, .5, color='red', linestyle='dotted')
    ax.hlines(-np.pi/2., -0.5, 0.5, color='gray', linestyle='dotted', linewidth=linewidth)
    ax.hlines(np.pi/2., -0.5, 0.5, color='gray', linestyle='dotted')
    ax.hlines(np.pi, -0.5, 0.5, color='gray', linestyle='dotted', linewidth=linewidth)
    
    # plot POST EDGE GUIDE LINES
    if plot_post_edges is True:
        norm = plt.Normalize(0,.3)
        cmap = plt.get_cmap('jet')
        color = cmap( norm(movieinfo.scaled.speed[align_frame]) )
        while np.sum(color[0:3]) > 1.5: # make darker for readability
            color = [i/2. for i in color]
            color[-1] = 1.0
        ax.hlines(-np.pi/2., -0.03, 0.03, color=color, linestyle='-', zorder=100, linewidth=linewidth*2)
        # post top
        ax.hlines(np.pi, -0.03, 0.03, color='green', linestyle='-', zorder=100, linewidth=linewidth*2)
    
    if 'crash' in movieinfo.subbehavior or 'wingcrash' in movieinfo.subbehavior:
        el = patches.Ellipse( (0,0), .02, .2, facecolor='red', edgecolor='red', zorder=20)
        ax.add_artist(el)
    
    # primary AX parameters
    axylim = [-np.pi/2., np.pi]
    ax.set_ylim(axylim)
    if time_limits is None:
        if movieinfo.behavior == 'landing':
            axxlim = [-.5, .2]
        else:
            axxlim = [-.3, .4]
    else:
        axxlim = time_limits
    ax.set_xlim(axxlim) 
    
    if show_angle_spine is not False:
        if show_angle_spine is True:
            spines = ['left']
        else:
            spines = [show_angle_spine]
    else:
        spines = []
    if show_time_spine:
        spines.append('bottom')
        
    spine_locations={'bottom': 10}
    if 'right' in spines:
        spine_locations.setdefault('right', 5)
    adjust_spines(ax,spines, spine_locations=spine_locations)
    if show_time_spine:
        xticks = np.linspace(axxlim[0], axxlim[-1], int(np.ceil((axxlim[-1]-axxlim[0])/0.1))+1, endpoint=True)
        xtickstrings = ['' for tick in xticks]
        for i, tick in enumerate(xticks): 
            tickstring = ''
            if tick<0: tickstring += '-'
            stick = str(tick)
            stick = stick.lstrip('-0')
            tickstring += stick[0:2]
            if tick > -0.00001 and tick < 0.00001: tickstring = '0'
            xtickstrings[i] = tickstring
        
        ax.set_xticks(xticks)
        print xticks
        ax.set_xticklabels(xtickstrings, fontsize=label_font_size)
        print xtickstrings
        ax.set_xlabel('time, sec', fontdict={'fontsize': label_font_size})
    if show_angle_spine is not False:
        ax.set_ylabel('angle, deg', fontdict={'fontsize': label_font_size})
        yticks = [-np.pi/2., 0, np.pi/2, np.pi]
        ax.set_yticks(yticks)
        ax.set_yticklabels([-90, 0, 90, 180], fontsize=label_font_size)
        if show_angle_spine == 'right':
            ax.yaxis.set_label_position('right')
        
        
    # print LABELS / LEGEND
    if label_lines:
        y_vis = lower[frames][-1] + (upper[frames][-1] - lower[frames][-1])/2.
        # post img
        ax.text(time[frames][-1]+label_x_offset, y_vis, '2D post img', fontdict={'fontsize': label_font_size, 'color': 'black'}, withdash=False, horizontalalignment='left', verticalalignment='bottom')
        ax.text(time[frames][-1]+label_x_offset, y_vis, 'on flies retina', fontdict={'fontsize': label_font_size, 'color': 'black'}, withdash=False, horizontalalignment='left', verticalalignment='top')
        # slipangle
        ax.text(time[frames][0]-label_x_offset, slipangle[0], 'flight angle rel.', fontdict={'fontsize': label_font_size, 'color': 'gray'}, withdash=False, horizontalalignment='right', verticalalignment='bottom')
        ax.text(time[frames][0]-label_x_offset, slipangle[0], 'to orientation', fontdict={'fontsize': label_font_size, 'color': 'gray'}, withdash=False, horizontalalignment='right', verticalalignment='top')
    
    if show_colorbar_spine: # COLORBAR
        cticks = [0, .1, .2, .3]
        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('jet'), norm=plt.Normalize(clim[0], clim[1]), orientation='vertical', boundaries=None, ticks=cticks, drawedges=False)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(label_font_size)

        cax.set_ylabel('speed, m/s', fontsize=label_font_size)
    if show_colorbar_spine:
        cax.set_visible(True)
    else:
        cax.set_visible(False)
    
    # print POST TAGS
    if print_post_tags:
            norm = plt.Normalize(0,.3)
            cmap = plt.get_cmap('jet')
            color = cmap( norm(movieinfo.scaled.speed[align_frame]) )
            while np.sum(color[0:3]) > 1.5: # make darker for readability
                color = [i/2. for i in color]
                color[-1] = 1.0
            ax.text(0, -np.pi/2., 'post edge', fontdict={'fontsize': label_font_size, 'color': color}, withdash=False, horizontalalignment='center', verticalalignment='top')
            ax.text(0, np.pi, 'post top', fontdict={'fontsize': label_font_size, 'color': 'green'}, withdash=False, horizontalalignment='center')
        
    if flycon is not None and flycon is not False:
        print flycon
        flycon = plt.imread(flycon)
        ax.imshow(flycon, extent=[-.5,-.35,-.6,.6], aspect='auto', zorder=200)
    if movie_id_text:
        txt = movieinfo.id + ': ' + movieinfo.posttype# + str(movieinfo.change_in_behavior)
        ax.text(-.5, -np.pi/2, txt, fontdict={'fontsize': 8}, withdash=False)
        
    # package axes and return
    print ax.get_xticks()
    
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
    for m in range(nmovies):
        key = keys[m+firstmovie]
        print key
        n += 1
        movie = movie_dataset.movies[key]
        ax = fig.add_subplot(nrows,ncols,n)
        
        if n==right_side_panel[-1]:
            axes = plot_expansion(movie, ax, figure=fig, show_time_spine=True, show_angle_spine=False, show_dist_spine=True, flip=True, show_colorbar_spine=False, label_lines=True, print_post_tags=True, flycon=None)
        elif n==right_side_panel[-2]:
            axes = plot_expansion(movie, ax, figure=fig, show_time_spine=False, show_angle_spine=False, show_dist_spine=False, show_colorbar_spine=True, flip=True, label_lines=False)
        elif n==left_side_panel[-1]:
            axes = plot_expansion(movie, ax, figure=fig, show_time_spine=False, show_angle_spine=True, show_dist_spine=False, flip=True, label_lines=False)
        else:
            axes = plot_expansion(movie, ax, figure=fig, show_time_spine=False, show_angle_spine=False, show_dist_spine=False, flip=True, label_lines=False)
        
            
    #plt.draw()
    #plt.show()
    return fig

###
def plot_trajectory(movieinfo, figure=None, show_post=True, show_flydra=True):
    fig = plt.figure(figure)
    ax0 = fig.add_axes([.1,.1,.8,.8])
    post_pos = movieinfo.post_pos
    frames = sa1.get_frames_until_landing(movieinfo)
    
    ax0.plot(movieinfo.scaled.positions[frames,0], movieinfo.scaled.positions[frames,1])
    if show_flydra:
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
        
    if show_post:
        circle = patches.Circle( (0,0), radius=movieinfo.post_radius, facecolor='none', edgecolor='green')
        ax0.add_artist(circle)
    
    title = movieinfo.id + ' ' + movieinfo.behavior + ' ' + movieinfo.subbehavior[0]
    ax0.set_title(title)
    
    
    
    return ax0
    
###
def plot_lollipops(movieinfo, ax=None, figure=None, show_spines=['left', 'bottom'], scaletofit=True, interval=100, flip=False):
    print 'plotting lollipos'
    frames = sa1.get_frames_until_landing(movieinfo)
    try: fake = movieinfo.fake
    except: fake = False
    
    if type(flip) is bool:
        pass
    else:
        if flip == 1: flip = False
        else: flip = True
        
    
    ## SET UP ALL AXES ##
    if ax is None:
        fig = plt.figure(figure)
        fig.set_facecolor('white')
        ax = fig.add_axes([.1, .1, .8, .8])
    ax.set_axis_bgcolor('white')
    ## DONE WITH AXES  ##
    
    # trajectory
    #ax.plot(movieinfo.scaled.positions[frames,0], movieinfo.scaled.positions[frames,1], color='gray')
    
    # post
    if not fake:
        post_pos = (movieinfo.post_pos - np.array([512,512]))*movieinfo.scale
    else:
        post_pos = movieinfo.post_pos
    if flip:
        post_pos = post_pos[::-1]
    if movieinfo.posttype == 'black':
        facecolor = 'black'
    elif movieinfo.posttype == 'checkered':
        facecolor = 'gray'
    circle = patches.Circle( post_pos, radius=movieinfo.post_radius, facecolor=facecolor, edgecolor='none')
    ax.add_artist(circle)
    
    # heads
    i = 0
    while i < len(frames):
        head_pos = movieinfo.scaled.head_pos[frames[i]]
        center = movieinfo.scaled.positions[frames[i]]
        if flip:
            head_pos = head_pos[::-1]
            center = center[::-1]
        dx, dy = (head_pos - center)*5
                
        arrow = Arrow(head_pos[0], head_pos[1], -1*dx, -1*dy, width=.0001, color='red', linewidth=1)
        ax.add_artist(arrow)
        
        try:
            err = movieinfo.deviation_from_initial_state[i]
        except:
            err = 0
        if err > 0.5:
            facecolor = 'blue'
        else:
            facecolor = 'black'
        
        head = patches.Circle( head_pos, radius=0.001, facecolor=facecolor, edgecolor='none')
        ax.add_artist(head)
        
        i += interval    
    
        if flip:
            xindex = 1
            yindex = 0
        else:   
            xindex = 0
            yindex = 1
        xmin = np.min(movieinfo.scaled.positions[frames][:,xindex])
        xmax = np.max(movieinfo.scaled.positions[frames][:,xindex])
        ymin = np.min(movieinfo.scaled.positions[frames][:,yindex])
        ymax = np.max(movieinfo.scaled.positions[frames][:,yindex])
    
    if scaletofit:
        # compare to post
        plotbuffer = 0.1*movieinfo.post_radius
        xmin = np.min( [xmin, post_pos[0]-movieinfo.post_radius-plotbuffer] )
        xmax = np.max( [xmax, post_pos[0]+movieinfo.post_radius+plotbuffer] )
        ymin = np.min( [ymin, post_pos[1]-movieinfo.post_radius-plotbuffer] )
        ymax = np.max( [ymax, post_pos[1]+movieinfo.post_radius+plotbuffer] )

        # now make pretty
        xmin = np.float32( np.floor( 100.*xmin ) / 100. )
        xmax = np.float32( np.ceil( 100.*xmax ) / 100. )
        ymin = np.float32( np.floor( 100.*ymin ) / 100. )
        ymax = np.float32( np.ceil( 100.*ymax ) / 100. )

        xlim = (xmin, xmax)
        ylim = (ymin, ymax)
    else:
        xlim = (-.08, .08)
        ylim = (-.08, .08)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    
    if show_spines is not None:
        adjust_spines(ax,show_spines, smart_bounds=False)
        
        if 'bottom' in show_spines:
            xticks = np.arange(xlim[0], xlim[1]+.03, 0.03)
            print xticks
            ax.set_xticks(xticks)
            ax.set_xlabel('x position, m')
        
        if 'left' in show_spines:
            yticks = np.arange(ylim[0], ylim[1]+.03, 0.03)
            ax.set_yticks(yticks)
            ax.set_ylabel('y position, m')
        
    else:
        adjust_spines(ax, 'none')    
        
        
    #plt.show()
    return ax
    
###
def plot_lollipop_and_expansion(movieinfo, ax, figure=None, plot_dist=True, plot_altitude=True, plot_slipangle=True, plot_angular_vel=False, plot_classification=False, show_dist_spine=True, show_time_spine=True, show_angle_spine='left', show_colorbar_spine=False, flip=False, label_lines=True, print_post_tags=False, label_font_size=9, lollipop=True, show_lollipop_spines=None, flycon=None, movie_id_text=True, time_limits=None, plot_post_edges=True):
    try:
        tmp = movieinfo.scaled
    except:
        sa1.process_movieinfo(movieinfo)
    expaxes = plot_expansion(movieinfo, ax=ax, figure=figure, plot_dist=plot_dist, plot_altitude=plot_altitude, plot_slipangle=plot_slipangle, plot_angular_vel=plot_angular_vel, plot_classification=plot_classification, show_dist_spine=show_dist_spine, show_time_spine=show_time_spine, show_angle_spine=show_angle_spine, show_colorbar_spine=show_colorbar_spine, flip=flip, label_lines=label_lines, print_post_tags=print_post_tags, label_font_size=label_font_size, lollipop=lollipop, show_lollipop_spines=show_lollipop_spines, flycon=flycon, movie_id_text=movie_id_text, time_limits=time_limits, plot_post_edges=plot_post_edges)
    return expaxes
    
#sap.plot_lollipop_and_expansion_new_figure(movieinfo, plot_dist=True, show_dist_spine=False, plot_altitude=True, show_colorbar_spine=True, plot_slipangle=True, label_lines=False, movie_id_text=False, plot_post_edges=True, flip=True, lollipop=True, label_font_size=14)
def plot_lollipop_and_expansion_new_figure(movieinfo, figure=None, plot_dist=True, plot_altitude=True, plot_slipangle=True, plot_angular_vel=False, plot_classification=False, show_dist_spine=True, show_time_spine=True, show_angle_spine='left', show_colorbar_spine=False, flip=False, label_lines=True, print_post_tags=False, label_font_size=16, lollipop=False, show_lollipop_spines=None, flycon=None, movie_id_text=True, time_limits=None, plot_post_edges=True, scale=3):
    fig = plt.figure()
    fig.set_facecolor('white')
    ax1 = fig.add_axes([.1, .25, .75, .6])
    #ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122)
    
    axes = plot_lollipop_and_expansion(movieinfo, ax1, figure=figure, plot_dist=plot_dist, plot_altitude=plot_altitude, plot_slipangle=plot_slipangle, plot_angular_vel=plot_angular_vel, plot_classification=plot_classification, show_dist_spine=show_dist_spine, show_time_spine=show_time_spine, show_angle_spine=show_angle_spine, show_colorbar_spine=show_colorbar_spine, flip=flip, label_lines=label_lines, print_post_tags=print_post_tags, label_font_size=label_font_size, lollipop=lollipop, show_lollipop_spines=show_lollipop_spines, flycon=flycon, movie_id_text=movie_id_text, time_limits=time_limits, plot_post_edges=plot_post_edges)    
    
    fig.set_size_inches(2*scale,1*scale)
    fig.savefig('lol_exp_fig', dpi=200)
    return fig
    
###
def plot_lollipop_and_expansion_for_dataset(movie_dataset, figure=None, behavior=None, posttype=None, firstmovie=0, nmovies=None, columns=1, crash=False, save_as_pdf=False, keys=None, label_font_size=None):
    if save_as_pdf is not False:
        plt.close('all')
    
    fig = plt.figure(figure)
    fig.set_facecolor('white')
    
    if behavior == 'fake': 
        posttype='black'
        fake=True
    else:
        fake=False
        
    if keys is None:
        keys = movie_dataset.get_movie_keys(fake=fake, behavior=behavior, posttype=posttype, crash=crash)

    nkeys = len(keys)
    print 'n keys: ', nkeys
    if nmovies is None:
        nmovies = nkeys
    elif nmovies > nkeys:
        nmovies = nkeys
    
    ncols = columns
    nrows = int(np.ceil( (nmovies)/float(ncols)))
    print nrows
    left_side_panel = np.arange(1,nmovies+1,ncols)
    right_side_panel = np.arange(ncols,nmovies+2,ncols)
    
    keys_to_plot = keys[firstmovie:firstmovie+nmovies]
    print firstmovie, firstmovie+nmovies, len(keys)
    print 'n keys to plot: ', len(keys_to_plot)
    
    # find key where the movie stops the soonest
    last_times = np.zeros([len(keys_to_plot)])
    for i, key in enumerate(keys_to_plot):
        print key
        movieinfo = movie_dataset.movies[key]
        
        try:
            tmp = movieinfo.scaled
        except:
            sa1.process_movieinfo(movieinfo)
        
        frames = sa1.get_frames_until_landing(movieinfo)
        time_nearest_to_post = sa1.get_time_nearest_to_post(movieinfo)
        frame_nearest_to_post = sa1.get_frame_from_timestamp(movieinfo, time_nearest_to_post)
        time = movieinfo.timestamps - time_nearest_to_post
        last_times[i] =  1
        a = (time[frames])[-1]
    earliest_ending_key = keys[np.argmin(last_times)]
        
    n = 0
    print keys
    label = True
    for m, key in enumerate(keys_to_plot):
        print key
        n += 1
        movie = movie_dataset.movies[key]
        ax = fig.add_subplot(nrows,ncols,n)
        
        if n > nmovies-1: show_spines=True
        else: show_spines=False
        
        try:
            if n == right_side_panel[-2]:
                show_angle_spine = 'right'
            else:
                show_angle_spine = False
        except:
            show_angle_spine = False
            
        try:
            if n == right_side_panel[-3]:
                show_dist_spine = True
            else:
                show_dist_spine = False
        except:
            show_dist_spine=False
            
        if key == earliest_ending_key:
            label_lines = True
        else:
            label_lines = False
            
        
        if show_spines:
            axes = plot_lollipop_and_expansion(movie, ax, show_lollipop_spines=None, show_dist_spine=show_dist_spine, show_time_spine=True, show_angle_spine=show_angle_spine, show_colorbar_spine=True, print_post_tags=True, label_lines=label_lines, flycon=None, label_font_size=label_font_size)
        else:
            axes = plot_lollipop_and_expansion(movie, ax, show_lollipop_spines=None, show_dist_spine=show_dist_spine, show_time_spine=False, show_angle_spine=show_angle_spine, show_colorbar_spine=False, print_post_tags=False, label_lines=label_lines, label_font_size=label_font_size)
            
    if behavior is None or type(behavior) is list:
        behavior = 'mixed behaviors'
    if posttype is None or type(posttype) is list:
        posttype = 'mixed type'
        
    if behavior == 'landing' and crash is True:
        behavior += ' crash'
    elif behavior == 'landing':
        behavior += ' no crash'
        
    if save_as_pdf is not False:
        if figure is None:
            fig_numbers = [x.num for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()] 
            figure = fig_numbers[-1]
        scale = 10
        pp =  pdf.PdfPages(save_as_pdf)
        fig.set_size_inches(2*scale,1*scale)
        fig.set_dpi(72)
        pp.savefig(figure)
        plt.close(figure)
        
        # Once you are done, remember to close the object:
        pp.close()
        
    return fig
    
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
    
###
def pdf_lollipop_and_expansion(movie_dataset, filename='sa1_ethogragraphical_plots.pdf', scale=10):
    
    nmovies_dict = sa1.count_movies(movie_dataset)
    nmovies_per_page = 8
    
    plt.close('all')
    pp =  pdf.PdfPages(filename)
    
    f = -1
    
    for key in nmovies_dict.keys():
        key_parsed = key.split(',')
        behavior = key_parsed[0]
        posttype = key_parsed[-1]
        if behavior == 'landing':
            if key_parsed[1] == 'crash':
                crash = True
            else:
                crash = False
        else:
            crash = False
        movies_plotted = 0
        
        while movies_plotted < nmovies_dict[key]:
            f += 1
            if movies_plotted > 0:
                firstmovie = movies_plotted-1
            else:
                firstmovie = 0
            print '*'*80
            print behavior + ' ' + str(crash) + ' ' + posttype + ' nmovies: ', nmovies_dict[key], movies_plotted, firstmovie
            fig = plot_lollipop_and_expansion_for_dataset(movie_dataset, figure=f, behavior=behavior, posttype=posttype, crash=crash, nmovies=nmovies_per_page, columns=2, firstmovie=firstmovie)
            fig.set_size_inches(2*scale,1*scale)
            fig.set_dpi(72)
            pp.savefig(f)
            plt.close(f)
            movies_plotted += nmovies_per_page
            
            
    
    # Once you are done, remember to close the object:
    pp.close()
    print 'closed'
    
###
def pdf_random(movie_dataset, filename='random_plots.pdf', scale=10):
    
    keys = movie_dataset.get_movie_keys()
    nmovies_per_page = 32
    
    plt.close('all')
    pp =  pdf.PdfPages(filename)
    
    f = 0
    
    fig = plot_lollipop_and_expansion_for_dataset(movie_dataset, figure=f, nmovies=nmovies_per_page, columns=4, keys=keys)
    fig.set_size_inches(2*scale,1*scale)
    fig.set_dpi(72)
    
    pp.savefig(f)
    plt.close(f)
    pp.close()
    
    
    
###
def pdf_lollipop_and_expansion_for_landing_subbehaviors(movie_dataset, filename='sa1_ethogragraphical_plots_landing_subbehaviors.pdf', scale=10):
    
    nmovies_per_page = 8
    
    plt.close('all')
    pp =  pdf.PdfPages(filename)
    
    f = -1
    
    
    fixation_keys = sa1.get_keys(movie_dataset, 'landing', 'fixation')
    centered_keys = sa1.get_keys(movie_dataset, 'landing', 'centered')
    hover_keys_raw = sa1.get_keys(movie_dataset, 'landing', 'hover')
    hover_keys = []
    for key in hover_keys_raw:
        if key not in centered_keys and key not in fixation_keys:
            hover_keys.append(key)
    etf_keys = sa1.get_keys(movie_dataset, 'landing', 'etf')
    
    keys_to_plot = [fixation_keys, centered_keys, hover_keys, etf_keys]
    keys_dict = ['fixation', 'centered', 'hover', 'end of tunnel fixation and turn']
    
    print '*'*80
    print keys_to_plot
    print '*'*80
    
    for k, keys in enumerate(keys_to_plot):
        movies_plotted = 0
        
        while movies_plotted < len(keys):
            f += 1
            if movies_plotted > 0:
                firstmovie = movies_plotted
            else:
                firstmovie = 0
            
            fig = plot_lollipop_and_expansion_for_dataset(movie_dataset, figure=f, nmovies=nmovies_per_page, columns=2, firstmovie=firstmovie, keys=keys)
            fig.set_size_inches(2*scale,1*scale)
            fig.set_dpi(72)
            
            fig_title = keys_dict[k]
            fig.text(0.5,.98,fig_title, horizontalalignment='center', verticalalignment='top', weight='heavy')
    
    
            pp.savefig(f)
            plt.close(f)
            movies_plotted += nmovies_per_page
            
            
    
    # Once you are done, remember to close the object:
    pp.close()
    print 'closed'
    
###
def pdf_lollipop_and_expansion_for_flyby_subbehaviors(movie_dataset, filename='sa1_ethogragraphical_plots_flyby_subbehaviors.pdf', scale=10):
    
    nmovies_per_page = 8
    
    plt.close('all')
    pp =  pdf.PdfPages(filename)
    
    f = -1
    
    
    fixation_keys = sa1.get_keys(movie_dataset, 'flyby', 'fixation')
    centered_keys = sa1.get_keys(movie_dataset, 'flyby', 'centered')
    hover_keys_raw = sa1.get_keys(movie_dataset, 'flyby', 'hover')
    hover_keys = []
    for key in hover_keys_raw:
        if key not in centered_keys and key not in fixation_keys:
            hover_keys.append(key)
    etf_keys = sa1.get_keys(movie_dataset, 'flyby', 'etf')
    
    keys_to_plot = [fixation_keys, centered_keys, hover_keys, etf_keys]
    keys_dict = ['fixation', 'centered', 'hover', 'end of tunnel fixation and turn']
    
    for k, keys in enumerate(keys_to_plot):
        movies_plotted = 0
        
        while movies_plotted < len(keys):
            f += 1
            if movies_plotted > 0:
                firstmovie = movies_plotted
            else:
                firstmovie = 0
            print '*'*80
            fig = plot_lollipop_and_expansion_for_dataset(movie_dataset, figure=f, nmovies=nmovies_per_page, columns=2, firstmovie=firstmovie, keys=keys)
            fig.set_size_inches(2*scale,1*scale)
            fig.set_dpi(72)
            
            fig_title = keys_dict[k]
            fig.text(0.5,.98,fig_title, horizontalalignment='center', verticalalignment='top', weight='heavy')
    
    
            pp.savefig(f)
            plt.close(f)
            movies_plotted += nmovies_per_page
            
            
    
    # Once you are done, remember to close the object:
    pp.close()
    print 'closed'

    
###
def pdf_lollipop_and_expansion_for_crash_subbehaviors(movie_dataset, filename='sa1_ethogragraphical_plots_crash_subbehaviors.pdf', scale=10):
    
    nmovies_per_page = 8
    
    plt.close('all')
    pp =  pdf.PdfPages(filename)
    
    f = -1
    
    
    fixation_keys = sa1.get_keys(movie_dataset, 'landing', 'fixation', crash=True)
    centered_keys = sa1.get_keys(movie_dataset, 'landing', 'centered', crash=True)
    hover_keys_raw = sa1.get_keys(movie_dataset, 'landing', 'hover', crash=True)
    hover_keys = []
    for key in hover_keys_raw:
        if key not in centered_keys and key not in fixation_keys:
            hover_keys.append(key)
    etf_keys = sa1.get_keys(movie_dataset, 'landing', 'etf', crash=True)
    
    keys_to_plot = [fixation_keys, centered_keys, hover_keys, etf_keys]
    keys_dict = ['fixation', 'centered', 'hover', 'end of tunnel fixation and turn']
    
    for k, keys in enumerate(keys_to_plot):
        movies_plotted = 0
        
        while movies_plotted < len(keys):
            f += 1
            if movies_plotted > 0:
                firstmovie = movies_plotted
            else:
                firstmovie = 0
            print '*'*80
            fig = plot_lollipop_and_expansion_for_dataset(movie_dataset, figure=f, nmovies=nmovies_per_page, columns=2, firstmovie=firstmovie, keys=keys)
            fig.set_size_inches(2*scale,1*scale)
            fig.set_dpi(72)
            
            fig_title = keys_dict[k]
            fig.text(0.5,.98,fig_title, horizontalalignment='center', verticalalignment='top', weight='heavy')
    
    
            pp.savefig(f)
            plt.close(f)
            movies_plotted += nmovies_per_page
            
            
    
    # Once you are done, remember to close the object:
    pp.close()
    print 'closed'
    
    
###
def pdf_lollipop_and_expansion_keys(movie_dataset, filename='sa1_ethogragraphical_plots_of_fixation_keys.pdf', scale=10, keys=None):
    if keys is None:
        keys = movie_dataset.get_movie_keys()
    '''
    for key in keys:
        movieinfo = movie_dataset.movies[key]
        try:
            isscaled = movieinfo.scaled
        except:
            print key
            del()
    '''        
    
    plt.close('all')
    pp =  pdf.PdfPages(filename)
    
    nmovies_per_page = 8
    npages = np.int(np.ceil(len(keys) / nmovies_per_page))
    firstmovie = 0
    for page in range(npages):
        
            lastmovie = np.min([len(keys), firstmovie+nmovies_per_page])
            if lastmovie <= firstmovie:
                break
            print keys[firstmovie:lastmovie]
            fig = plot_lollipop_and_expansion_for_dataset(movie_dataset, figure=page, nmovies=nmovies_per_page, columns=2, firstmovie=firstmovie, keys=keys)
            fig.set_size_inches(2*scale,1*scale)
            fig.set_dpi(72)
            pp.savefig(page)
            plt.close(page)
            firstmovie += nmovies_per_page
            
            
    
    # Once you are done, remember to close the object:
    pp.close()
    print 'closed'
    
    
def fig_of_classifications_for_candidacy(movie_dataset):

#sap.plot_lollipop_and_expansion_new_figure(movieinfo, plot_dist=True, show_dist_spine=False, plot_altitude=True, show_colorbar_spine=True, plot_slipangle=True, label_lines=False, movie_id_text=False, plot_post_edges=True, flip=True, lollipop=True, label_font_size=14)


    tuples = [(20101030, 19), (20101110,26), (20101111,22), (20101030,35), (20101111,47), (20101101,8), (20101110,43), (20101111,05)]
    keys =  sa1.get_trajectory_list_from_short_tuples(tuples)


    fig = plt.figure(None)
    fig.set_facecolor('white')
    f = 0
    
    for i in range(len(keys)):
        f += 1
        
        ax = fig.add_subplot(4,2,f)
        
        show_dist_spine=False
        show_time_spine=False
        show_colorbar_spine=False
        print_post_tags=False
        label_lines=False
        show_angle_spine=False
        
        if f == 8:
            show_time_spine=True
            show_colorbar_spine=True
        if f == 6:
            show_angle_spine='right'
        if f == 4:
            show_dist_spine=True
        if f == 1:
            label_lines=True
        if f == 8:
            print_post_tags=True
        
        movieinfo = movie_dataset.movies[keys[i]] 
        
        label_lines=False
        print_post_tags=False
        
        plot_lollipop_and_expansion(movieinfo, ax, plot_dist=True, plot_altitude=True, plot_slipangle=True, plot_angular_vel=False, plot_classification=False, show_dist_spine=show_dist_spine, show_time_spine=show_time_spine, show_angle_spine=show_angle_spine, show_colorbar_spine=show_colorbar_spine, flip=True, label_lines=label_lines, print_post_tags=print_post_tags, label_font_size=9, lollipop=True, show_lollipop_spines=None, flycon=None, movie_id_text=False, time_limits=None, plot_post_edges=True)
    
        if f == 1:
            ax.text(-0.65, 0.5, 'fixation', fontdict={'fontsize': 9, 'color': 'black'}, withdash=False, horizontalalignment='right', verticalalignment='center', rotation='vertical')
        if f == 3:
            ax.text(-0.65, 0.5, 'centered', fontdict={'fontsize': 9, 'color': 'black'}, withdash=False, horizontalalignment='right', verticalalignment='center', rotation='vertical')
        if f == 5:
            ax.text(-0.65, 0.5, 'hovering', fontdict={'fontsize': 9, 'color': 'black'}, withdash=False, horizontalalignment='right', verticalalignment='center', rotation='vertical')
        if f == 7:
            ax.text(-0.65, 0.5, 'fixation switch', fontdict={'fontsize': 9, 'color': 'black'}, withdash=False, horizontalalignment='right', verticalalignment='center', rotation='vertical')
            
        if f == 1:
            ax.text(0, 4, 'flyby trajectories', fontdict={'fontsize': 9, 'color': 'black'}, withdash=False, horizontalalignment='center', verticalalignment='center', rotation='horizontal')
        if f == 2:
            ax.text(0, 4, 'landing trajectories', fontdict={'fontsize': 9, 'color': 'black'}, withdash=False, horizontalalignment='center', verticalalignment='center', rotation='horizontal')  
            
    fig.set_size_inches(6.5,5)
    return fig
        
        
        
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
    
###
def plot_slipangle_during_fixation_histogram(movie_dataset, fixation_threshold_degrees=7, fixation_duration_threshold=0.05):
    fig = plt.figure()
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)
    
    slipangle_during_fixation = []
    
    keys = movie_dataset.get_movie_keys()
    for i, key in enumerate(keys):
        movieinfo = movie_dataset.movies[key]
        new_data = sa1.get_slipangle_during_fixation(movieinfo, fixation_threshold_degrees, fixation_duration_threshold)
        slipangle_during_fixation = np.hstack( (slipangle_during_fixation, new_data) )
    
    ax.hist(slipangle_during_fixation*180./np.pi, bins=np.linspace(-90,90,100,endpoint=True))
    spines = ['left', 'bottom']
    adjust_spines(ax,spines, color='black')
    
    ax.set_xlabel('slip angle during fixation, degrees')
    ax.set_ylabel('number of data points')
    titlestring = 'slip angle during fixation, min of ' + str(fixation_duration_threshold) + ' sec, and < ' + str(fixation_threshold_degrees) + ' deg error'
    ax.set_title(titlestring)
    
    #plt.show()
    
    return ax
    
###
def plot_speed_during_fixation_histogram(movie_dataset, fixation_threshold_degrees=5, fixation_duration_threshold=0.1, keys=None):
    fig = plt.figure()
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)
    
    speed_during_fixation = []
    
    if keys is None:
        keys = movie_dataset.get_movie_keys(behavior=['landing', 'flyby'], crash=False)
    fixation_keys = []
    for i, key in enumerate(keys):
        movieinfo = movie_dataset.movies[key]
        #new_data = sa1.get_speed_during_fixation(movieinfo, fixation_threshold_degrees, fixation_duration_threshold)
        #speed_during_fixation = np.hstack( (speed_during_fixation, new_data) )
        sa1.calc_flydra_fixation_for_movieinfo(movieinfo)
        new_data = sa1.get_flydra_speed_during_fixation(movieinfo.trajec, fixation_threshold_degrees, fixation_duration_threshold)
        speed_during_fixation = np.hstack( (speed_during_fixation, new_data) )
        if len(new_data) > 0:
            print key, len(new_data)
            fixation_keys.append(key)
    
    
    ax.hist(speed_during_fixation, bins=np.linspace(0,.6,100,endpoint=True))
    spines = ['left', 'bottom']
    adjust_spines(ax,spines, color='black')
    
    ax.set_xlabel('speed during fixation, m/s', fontsize=14)
    ax.set_ylabel('number of data points', fontsize=14)
    titlestring = 'speed during fixation, min of ' + str(fixation_duration_threshold) + ' sec, and < ' + str(fixation_threshold_degrees) + ' deg error'
    #ax.set_title(titlestring, fontsize=14)
    
    #plt.show()
    
    return ax, fixation_keys
    
###
def plot_speed_not_fixation_histogram(movie_dataset, fixation_threshold_degrees=5, fixation_duration_threshold=0.00, keys=None):
    fig = plt.figure()
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)
    
    speed_during_fixation = []
    
    if keys is None:
        keys = movie_dataset.get_movie_keys(behavior=['landing', 'flyby'], crash=False)
    #fixation_keys = []
    for i, key in enumerate(keys):
        movieinfo = movie_dataset.movies[key]
        #new_data = sa1.get_speed_during_fixation(movieinfo, fixation_threshold_degrees, fixation_duration_threshold)
        #speed_during_fixation = np.hstack( (speed_during_fixation, new_data) )
        new_data = sa1.get_flydra_speed_not_fixation(movieinfo.trajec, fixation_threshold_degrees, fixation_duration_threshold)
        speed_during_fixation = np.hstack( (speed_during_fixation, new_data) )
        #if len(new_data) > 100:
        #    fixation_keys.append(key)
    
    
    ax.hist(speed_during_fixation, bins=np.linspace(0,.6,100,endpoint=True))
    spines = ['left', 'bottom']
    adjust_spines(ax,spines, color='black')
    
    ax.set_xlabel('speed when not fixating, m/s', fontsize=14)
    ax.set_ylabel('number of data points', fontsize=14)
    titlestring = 'speed when NOT fixating, min of ' + str(fixation_duration_threshold) + ' sec, and < ' + str(fixation_threshold_degrees) + ' deg error'
    #ax.set_title(titlestring)
    
    #plt.show()
    
    return ax#, fixation_keys
    
    
###
def plot_speed_vs_pitch(movie_dataset, keys=None): 


    fig = plt.figure()
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)
    
    
    if keys is None:
        keys = movie_dataset.get_movie_keys()
    for i, key in enumerate(keys):
        movieinfo = movie_dataset.movies[key]
        frames = sa1.get_frames_until_landing(movieinfo)
        print movieinfo.scaled.speed.shape, movieinfo.scaled.pitchangle.shape
        ax.plot(movieinfo.flycoord.vel[frames,0]*movieinfo.scale, movieinfo.scaled.pitchangle[frames], '*')
    
    
    #plt.show()
    
    return ax



###
def plot_flydra_speed_during_fixation_histogram(dataset, fixation_threshold_degrees=5, fixation_duration_threshold=0.1):
    fig = plt.figure()
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)
    
    speed_during_fixation = []
    
    keys = dataset.trajecs.keys()
    for i, key in enumerate(keys):
        trajec = dataset.trajecs[key]
        if trajec.behavior == 'landing' and trajec.post_type=='checkered':
            new_data = sa1.get_flydra_speed_during_fixation(trajec, fixation_threshold_degrees, fixation_duration_threshold)
            speed_during_fixation = np.hstack( (speed_during_fixation, new_data) )
    
    
    ax.hist(speed_during_fixation, bins=150)
    spines = ['left', 'bottom']
    adjust_spines(ax,spines, color='black')
    
    ax.set_xlabel('speed during fixation, degrees')
    ax.set_ylabel('number of data points')
    titlestring = 'speed during fixation, min of ' + str(fixation_duration_threshold) + ' sec, and < ' + str(fixation_threshold_degrees) + ' deg error'
    ax.set_title(titlestring)
    
    #plt.show()
    
    return ax
###
def plot_flydra_speed_not_fixation_histogram(dataset, fixation_threshold_degrees=5, fixation_duration_threshold=0.1):
    fig = plt.figure()
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)
    
    speed_during_fixation = []
    
    keys = dataset.trajecs.keys()
    for i, key in enumerate(keys):
        trajec = dataset.trajecs[key]
        if trajec.behavior == 'landing' and trajec.post_type=='checkered':
            new_data = sa1.get_flydra_speed_not_fixation(trajec, fixation_threshold_degrees, fixation_duration_threshold)
            speed_during_fixation = np.hstack( (speed_during_fixation, new_data) )
    
    
    ax.hist(speed_during_fixation, bins=150)
    spines = ['left', 'bottom']
    adjust_spines(ax,spines, color='black')
    
    ax.set_xlabel('speed during fixation, degrees')
    ax.set_ylabel('number of data points')
    titlestring = 'speed during fixation, min of ' + str(fixation_duration_threshold) + ' sec, and < ' + str(fixation_threshold_degrees) + ' deg error'
    ax.set_title(titlestring)
    
    #plt.show()
    
    return ax
    

###
def plot_flydra_speed_during_fixation_means(dataset, fixation_threshold_degrees=5, fixation_duration_threshold=0.07):
    fig = plt.figure()
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)
    
    
    keys = dataset.trajecs.keys()
    for i, key in enumerate(keys):
        trajec = dataset.trajecs[key]
        mean, std = sa1.get_flydra_speed_during_fixation_means(trajec, fixation_threshold_degrees, fixation_duration_threshold)
        
        for m in range(len(mean)):
            ax.plot(mean[m], i, '*')
            ax.hlines(i, mean[m]-std[m], mean+std[m], color='black', linestyle='-')
    
    spines = ['left', 'bottom']
    adjust_spines(ax,spines, color='black')
    
    ax.set_xlabel('speed during fixation, degrees')
    ax.set_ylabel('number of data points')
    titlestring = 'speed during fixation, min of ' + str(fixation_duration_threshold) + ' sec, and < ' + str(fixation_threshold_degrees) + ' deg error'
    ax.set_title(titlestring)
    
    #plt.show()
    
    return ax

###
def plot_speed_vs_pitch_no_transients(movie_dataset):
    fig = plt.figure()
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)
    
    accel_threshold=.008
    duration_threshold=0.25
    
    keys = movie_dataset.get_movie_keys()
    pitch = []
    speed = []
    for i, key in enumerate(keys):
        movieinfo = movie_dataset.movies[key]
        newpitch, newspeed = sa1.get_bouts_of_constant_speed(movieinfo, accel_threshold=accel_threshold, duration_threshold=duration_threshold)    
        #ax.plot(speed, np.array(pitch)*180/np.pi, '*')
        pitch.extend(newpitch)
        speed.extend(newspeed)
    ax.plot(speed, np.array(pitch)*180/np.pi, '*')
            
    spines = ['left', 'bottom']
    adjust_spines(ax,spines, color='black')
    
    ax.set_xlabel('mean speed, m/s')
    ax.set_ylabel('mean pitch angle estimate, deg')
    titlestring = 'pitch angle vs. speed for bouts of constant speed for trajectory snippets longer than ' + str(duration_threshold) + ' sec'
    ax.set_title(titlestring)
        
        
        
    speed = np.array( speed )
    pitch = np.array( pitch )
        
    
    
    
    return pitch, speed
    









###############################################################################
# Averaging Stuff
###############################################################################




def fill_between_average_test(movie_dataset):

    #fixation_keys = sa1.get_keys(movie_dataset, 'flyby', 'fixation')
    fixation_keys = sa1.get_keys(movie_dataset, 'landing', 'fixation')
    centered_keys = sa1.get_keys(movie_dataset, 'landing', 'centered', crash=True)
    '''
    fixation_keys = sa1.get_keys(movie_dataset, 'landing', 'fixation', crash=True)
    
    hover_keys_raw = sa1.get_keys(movie_dataset, 'landing', 'hover', crash=True)
    hover_keys = []
    for key in hover_keys_raw:
        if key not in centered_keys and key not in fixation_keys:
            hover_keys.append(key)
    etf_keys = sa1.get_keys(movie_dataset, 'landing', 'etf', crash=True)
    '''
    etf_keys = sa1.get_keys(movie_dataset, 'landing', 'etf', crash=False)
    
    align_method = 'state change'
    flip = 1
    
    fig = plt.figure()
    ax = fig.add_axes([.1,.1,.8,.8])
    
    for key in centered_keys:
        movieinfo = movie_dataset.movies[key]
        sa1.calc_flydra_post_angles(movieinfo)
        frames = sa1.get_frames_until_landing(movieinfo)
        # simple variable names        
        
            
        if align_method == 'state change':
            try:
                #align_frame = np.where(movieinfo.deviation_from_initial_state > 0.5)[0][0]
                align_frame = np.where(movieinfo.trajec.dist_to_stim_r < 0.025)[0][0]
            except:
                align_frame = frames[-1]
            align_time = movieinfo.timestamps[align_frame]
            
        time = movieinfo.trajec.epoch_time - align_time
        align_frame_flydra = sa1.get_frame_from_timestamp_flydra(movieinfo, align_time)
        first_frame_flydra = sa1.get_frame_from_timestamp_flydra(movieinfo, align_time-0.2)
        # decide whether to flip or not
        avg_upper = np.mean(movieinfo.trajec.angle_to_upper_edge[first_frame_flydra:align_frame_flydra])
        avg_lower = np.mean(movieinfo.trajec.angle_to_lower_edge[first_frame_flydra:align_frame_flydra])
        flip = np.sign(np.mean([avg_upper, avg_lower]))
        
        upper = movieinfo.trajec.angle_to_upper_edge*flip
        lower = movieinfo.trajec.angle_to_lower_edge*flip
        
        # plot FILL BETWEEN
        ax.fill_between(time, lower, upper, facecolor='black', alpha=0.3)
        ax.plot(time, movieinfo.trajec.signed_angle_to_post*flip, color='black')
        
        small_subtension = np.where(movieinfo.trajec.angle_subtended_by_post < np.pi*5/180.)[0].tolist()
        large_subtension = np.where(movieinfo.trajec.angle_subtended_by_post >= np.pi*5/180.)[0].tolist()
        
        #ax.plot(time[small_subtension], movieinfo.trajec.dist_to_stim_r[small_subtension]*10, color='blue')
        #ax.plot(time[large_subtension], movieinfo.trajec.dist_to_stim_r[large_subtension]*10, color='red')
        ax.plot(time, movieinfo.trajec.dist_to_stim_r*30, color='blue', alpha=0.5)
        
        if movieinfo.legextensionrange is not None:
            ax.plot( [movieinfo.timestamps[movieinfo.legextensionrange[0]]-align_time, movieinfo.timestamps[movieinfo.legextensionrange[1]]-align_time], [0,0], color='red', alpha=0.3, linewidth=5)
            
        ax.plot(time, movieinfo.trajec.speed*(-10), color='green', alpha=0.5)

        ax.set_xlim([-0.4, 0.2])
        ax.set_ylim([-5,5])
        
        
        
        
###
def time_to_impact_vs_change_in_state(movie_dataset):

    landing_keys = movie_dataset.get_movie_keys(behavior = 'landing')
    flyby_keys = movie_dataset.get_movie_keys(behavior = 'flyby')
    
    plt.figure()
    for key in landing_keys:
        movieinfo = movie_dataset.movies[key]
        frames = sa1.get_frames_until_landing(movieinfo)
        try:
            align_frame = np.where(movieinfo.deviation_from_initial_state > 0.5)[0][0] # highspeed frame
        except:
            align_frame = frames[-1]
        align_time = movieinfo.timestamps[align_frame]
        align_frame_flydra = sa1.get_frame_from_timestamp_flydra(movieinfo, align_time)
        
        
        print align_frame
        vel_at_change = movieinfo.trajec.speed[align_frame_flydra-10]
        dist_at_change = movieinfo.trajec.dist_to_stim_r[align_frame_flydra-10]
        movieinfo.change_in_behavior = dist_at_change/vel_at_change
        
        
        
    
        
        '''
        x = mu + sigma*P.randn(1000,3)

n, bins, patches = P.hist(x, 10, normed=1, histtype='bar',
                            color=['crimson', 'burlywood', 'chartreuse'],
                            label=['Crimson', 'Burlywood', 'Chartreuse'])
                            
                            
        '''
        
        
    #plt.hist(time_to_impact, bins=40, range=(0,.5))
    #plt.title('landing')
        
        
    plt.figure()
    time_to_impact = []
    for key in flyby_keys:
        movieinfo = movie_dataset.movies[key]
        frames = sa1.get_frames_until_landing(movieinfo)
        try:
            align_frame = np.where(movieinfo.deviation_from_initial_state > 0.5)[0][0] # flydra time
        except:
            align_frame = frames[-1]
        align_time = movieinfo.timestamps[align_frame]
        align_frame_flydra = sa1.get_frame_from_timestamp_flydra(movieinfo, align_time)
        
        print align_frame
        vel_at_change = movieinfo.trajec.speed[align_frame_flydra-10]
        dist_at_change = movieinfo.trajec.dist_to_stim_r[align_frame_flydra-10]
        time_to_impact.append( dist_at_change/vel_at_change )
        movieinfo.change_in_behavior = dist_at_change/vel_at_change
        
    #plt.hist(time_to_impact, bins=40, range=(0,.5))
    #plt.title('flyby')
        
        
        
###
def plot_testing_behavior_change(movieinfo):
    
    #plt.plot(movieinfo.trajec.speed)
    
    ## kalman
    data = movieinfo.trajec.speed.reshape([len(movieinfo.trajec.speed),1])
    ss = 2 # state size
    os = 1 # observation size
    F = np.array([   [1,1], # process update
                     [0,1]],
                    dtype=np.float)
    H = np.array([   [0,1]], # observation matrix
                    dtype=np.float)
    Q = 1*np.eye(ss) # process noise
    R = 25*np.eye(os) # observation noise
    init_vel = data[0]
    initx = np.array([0, init_vel], dtype=np.float)
    initv = 0*np.eye(ss)
    xsmooth,Vsmooth = sa1.kalman_smoother(data, F, H, Q, R, initx, initv, plot=False)
    ## 
    
    #plt.plot(xsmooth[:,1])
    
    movieinfo.trajec.smooth_accel = sa1.diffa(xsmooth[:,1])
    
    #plt.plot(movieinfo.trajec.smooth_accel*20.)
    #plt.plot(movieinfo.trajec.dist_to_stim_r)
    
    
    movieinfo.trajec.world_velocity_angle = sa1.remove_angular_rollover(np.arctan2(movieinfo.trajec.velocities[:,1], movieinfo.trajec.velocities[:,0]), 3)
    ## kalman
    data = movieinfo.trajec.world_velocity_angle.reshape([len(movieinfo.trajec.world_velocity_angle),1])
    ss = 2 # state size
    os = 1 # observation size
    F = np.array([   [1,1], # process update
                     [0,1]],
                    dtype=np.float)
    H = np.array([   [1,0]], # observation matrix
                    dtype=np.float)
    Q = 1*np.eye(ss) # process noise
    R = 5*np.eye(os) # observation noise
    init_vel = data[0]
    initx = np.array([0, init_vel], dtype=np.float)
    initv = 0*np.eye(ss)
    xsmooth,Vsmooth = sa1.kalman_smoother(data, F, H, Q, R, initx, initv, plot=False)
    ## 
    plt.plot(xsmooth[:,1])
    plt.plot(movieinfo.trajec.world_velocity_angle)
    movieinfo.trajec.world_velocity_angle_smooth = xsmooth[:,0]
    movieinfo.trajec.world_velocity_angle_diff_smooth = xsmooth[:,1]
    plt.plot(sa1.diffa(xsmooth[:,1]))
    
    #plt.plot(movieinfo.trajec.orientation)
    
    plt.show()
    
    
    
    #plt.plot(movieinfo.trajec.accel_1d)
        
        
###        
def plot_flydra_trajecs(movie_dataset):
    landing_keys = movie_dataset.get_movie_keys(behavior='landing')
    fixation_keys = sa1.get_keys(movie_dataset, 'landing', 'fixation')
    centered_keys = sa1.get_keys(movie_dataset, 'landing', 'centered')
    hover_keys_raw = sa1.get_keys(movie_dataset, 'landing', 'hover')
    hover_keys = []
    for key in hover_keys_raw:
        if key not in centered_keys and key not in fixation_keys:
            hover_keys.append(key)
    etf_keys = sa1.get_keys(movie_dataset, 'landing', 'etf')    
    
    hover_trajecs = []
    for key in hover_keys:
        hover_trajecs.append(movie_dataset.movies[key].trajec)
    landing_trajecs = []
    for key in landing_keys:
        landing_trajecs.append(movie_dataset.movies[key].trajec)
    etf_trajecs = []
    for key in etf_keys:
        etf_trajecs.append(movie_dataset.movies[key].trajec)
    fixation_trajecs = []
    for key in fixation_keys:
        fixation_trajecs.append(movie_dataset.movies[key].trajec)
    centered_trajecs = []
    for key in centered_keys:
        centered_trajecs.append(movie_dataset.movies[key].trajec)
        
    crash_course_keys = get_crash_course_keys(movie_dataset)
    crash_course_trajecs = []
    for key in crash_course_keys:
        crash_course_trajecs.append(movie_dataset.movies[key].trajec)
        
    change_course_keys = get_change_course_keys(movie_dataset)
    change_course_trajecs = []
    for key in change_course_keys:
        change_course_trajecs.append(movie_dataset.movies[key].trajec)
        
    slow_course_keys = get_slow_course_keys(movie_dataset)
    slow_course_trajecs = []
    for key in slow_course_keys:
        slow_course_trajecs.append(movie_dataset.movies[key].trajec)
        
    ap.xy_trajectories(trajectory_objects = change_course_trajecs)
    
###
def get_crash_course_keys(movie_dataset, threshold = 0.2):
    landing_keys = movie_dataset.get_movie_keys(behavior='landing')
    crash_course_keys = []
    for key in landing_keys:
        movieinfo = movie_dataset.movies[key]
        if len(movieinfo.trajec.speed) >= 70:
            print movieinfo.trajec.behavior
            f = sa1.calc_frame_of_landing(movieinfo.trajec)
            print movieinfo.id, f
            avg_speed_before_landing = np.mean(movieinfo.trajec.speed[movieinfo.trajec.frame_of_landing-20:movieinfo.trajec.frame_of_landing])
            if avg_speed_before_landing > threshold:
                crash_course_keys.append(key)
    return crash_course_keys
###
def get_change_course_keys(movie_dataset, threshold = 10):
    landing_keys = movie_dataset.get_movie_keys(behavior='landing')
    change_course_keys = []
    for key in landing_keys:
        movieinfo = movie_dataset.movies[key]
        if len(movieinfo.trajec.speed) >= 70:
            f = sa1.calc_frame_of_landing(movieinfo.trajec)
            calc_change_in_course_prior_to_landing(movieinfo)
            if movieinfo.trajec.change_in_course > threshold:
                if movieinfo.id == '20101111_C001H001S0005':
                    print 'change in course ok'
                nframestocheck = np.min([len(movieinfo.trajec.speed[0:movieinfo.trajec.frame_of_landing])-1, 100])
                if f != 0 and f != -1 and f > nframestocheck:
                    max_speed_before_landing = np.max(movieinfo.trajec.speed[movieinfo.trajec.frame_of_landing-nframestocheck:movieinfo.trajec.frame_of_landing])
                    if max_speed_before_landing > 0.14:
                        change_course_keys.append(key)
    return change_course_keys
###
def get_slow_course_keys(movie_dataset, threshold = 0.14):
    landing_keys = movie_dataset.get_movie_keys(behavior='landing')
    slow_course_keys = []
    for key in landing_keys:
        movieinfo = movie_dataset.movies[key]
        if len(movieinfo.trajec.speed) >= 70:
            f = sa1.calc_frame_of_landing(movieinfo.trajec)
            if f != 0 and f != -1 and f > 100:
                max_speed_before_landing = np.max(movieinfo.trajec.speed[movieinfo.trajec.frame_of_landing-100:movieinfo.trajec.frame_of_landing])
                if max_speed_before_landing < threshold:
                    slow_course_keys.append(key)
    return slow_course_keys
###
def get_classified_keys(movie_dataset):
    
    crash_course_keys = get_crash_course_keys(movie_dataset)
    change_course_keys = get_change_course_keys(movie_dataset)
    slow_course_keys = get_slow_course_keys(movie_dataset)
    
    landing_keys = movie_dataset.get_movie_keys(behavior='landing')
    classified_keys = crash_course_keys + change_course_keys + slow_course_keys
    unclassified_keys = []
    for key in landing_keys:
        if key not in classified_keys:
            movieinfo = movie_dataset.movies[key]
            if len(movieinfo.trajec.speed) >= 70:
                unclassified_keys.append(key)
                
    return {'crash': crash_course_keys, 'change': change_course_keys, 'slow': slow_course_keys, 'unclassified': unclassified_keys}
###
def get_trajecs_from_keys(movie_dataset, keys):
    trajecs = []
    for key in keys:
        trajecs.append(movie_dataset.movies[key].trajec)
    return trajecs
###
def get_movies_from_keys(movie_dataset, keys):
    movies = []
    for key in keys:
        movies.append(movie_dataset.movies[key])
    return movies
###
def plot_flydra_trajecs_from_keys(movie_dataset, keys, print_obj_ids=False):
    trajecs = get_trajecs_from_keys(movie_dataset, keys)
    cl = ap.xy_trajectories(trajectory_objects = trajecs, print_obj_ids=print_obj_ids)
    return cl
###
def pdf_flydra_trajecs_of_classifications(movie_dataset, filename='classified_flydra_trajectories'):
    classified_keys = get_classified_keys(movie_dataset)
    
    page = -1
    plt.close('all')
    pp =  pdf.PdfPages(filename)
    
    for k, item in classified_keys.items():
        cl = plot_flydra_trajecs_from_keys(movie_dataset, item)
        title = str(k) + ' course'
        cl.ax0.set_title(title)
        cl.ax0.figure.set_size_inches(2*10,1*10)
        cl.ax0.figure.set_dpi(72)
        pp.savefig()
        plt.close('all')
    
    pp.close()
    print 'closed'
    
    
###
def calc_change_in_course_prior_to_landing(movieinfo, nframes=50):

    ## kalman
    data = movieinfo.trajec.speed.reshape([len(movieinfo.trajec.speed),1])
    ss = 2 # state size
    os = 1 # observation size
    F = np.array([   [1,1], # process update
                     [0,1]],
                    dtype=np.float)
    H = np.array([   [0,1]], # observation matrix
                    dtype=np.float)
    Q = 1*np.eye(ss) # process noise
    R = 25*np.eye(os) # observation noise
    init_vel = data[0]
    initx = np.array([0, init_vel], dtype=np.float)
    initv = 0*np.eye(ss)
    xsmooth,Vsmooth = sa1.kalman_smoother(data, F, H, Q, R, initx, initv, plot=False)
    ## 
    
    #plt.plot(xsmooth[:,1])
    
    movieinfo.trajec.smooth_accel = sa1.diffa(xsmooth[:,1])
    
    #plt.plot(movieinfo.trajec.smooth_accel*20.)
    #plt.plot(movieinfo.trajec.dist_to_stim_r)
    
    
    movieinfo.trajec.world_velocity_angle = sa1.remove_angular_rollover(np.arctan2(movieinfo.trajec.velocities[:,1], movieinfo.trajec.velocities[:,0]), 3)
    ## kalman
    data = movieinfo.trajec.world_velocity_angle.reshape([len(movieinfo.trajec.world_velocity_angle),1])
    ss = 2 # state size
    os = 1 # observation size
    F = np.array([   [1,1], # process update
                     [0,1]],
                    dtype=np.float)
    H = np.array([   [1,0]], # observation matrix
                    dtype=np.float)
    Q = 1*np.eye(ss) # process noise
    R = 5*np.eye(os) # observation noise
    init_vel = data[0]
    initx = np.array([0, init_vel], dtype=np.float)
    initv = 0*np.eye(ss)
    xsmooth,Vsmooth = sa1.kalman_smoother(data, F, H, Q, R, initx, initv, plot=False)
    ## 
    movieinfo.trajec.world_velocity_angle_smooth = xsmooth[:,0]
    movieinfo.trajec.world_velocity_angle_diff_smooth = xsmooth[:,1]
        
    frames = np.arange(movieinfo.trajec.frame_of_landing-nframes,movieinfo.trajec.frame_of_landing).tolist()
    mean_angle = np.mean(movieinfo.trajec.world_velocity_angle_smooth[frames])
    diff_from_mean = movieinfo.trajec.world_velocity_angle_smooth[frames]-mean_angle
    sum_diff_from_mean = np.sum( np.abs(diff_from_mean) )

    movieinfo.trajec.change_in_course = sum_diff_from_mean   
        
        
        
        
        
        
        
###
def normalize_dist_to_stim_r(trajec):
    trajec.dist_to_stim_r_normed = trajec.dist_to_stim_r - trajec.dist_to_stim_r[trajec.frame_of_landing]
def get_frame_at_distance(movieinfo, distance):
    normalize_dist_to_stim_r(movieinfo.trajec)
    frames = np.arange(0, movieinfo.trajec.frame_of_landing).tolist()
    dist_to_post = movieinfo.trajec.dist_to_stim_r_normed[frames]
    dist_crossovers = np.where( sa1.diffa(np.sign(dist_to_post - distance)) != 0 )[0]
    frame = dist_crossovers[-1]
    return frame
def get_speed_at_distance(movieinfo, distance):
    frame = get_frame_at_distance(movieinfo, distance)
    speed = movieinfo.trajec.speed[frame]
    return speed
def classify(movieinfo, dfar=0.06, dnear=0.005):

    speed_hi_threshold = 0.18
    speed_lo_threshold = 0.15

    if movieinfo.trajec.behavior == 'landing' and movieinfo.trajec.dist_to_stim_r[0] >= 0.06:
        movieinfo.trajec.speed_far = get_speed_at_distance(movieinfo, dfar)
        movieinfo.trajec.speed_near = get_speed_at_distance(movieinfo, dnear)
        
        if movieinfo.trajec.speed_near > speed_hi_threshold:
            movieinfo.trajec.classification = 'fast'
        elif movieinfo.trajec.speed_far < speed_lo_threshold:
            movieinfo.trajec.classification = 'slow'
        else:
            movieinfo.trajec.classification = 'mid'
###
def plot_classification(movie_dataset):
    landing_keys = movie_dataset.get_movie_keys(behavior='landing')
    for key in landing_keys:
        movieinfo = movie_dataset.movies[key]
        if movieinfo.trajec.behavior == 'landing' and movieinfo.trajec.dist_to_stim_r[0] >= 0.06:
            classify(movieinfo)
            
            if movieinfo.trajec.classification == 'fast':
                color = 'red'
            elif movieinfo.trajec.classification == 'slow':
                color = 'blue'
            elif movieinfo.trajec.classification == 'mid':
                color = 'black'
            else:
                color = 'green'
                
            plt.plot(movieinfo.trajec.speed_far, movieinfo.trajec.speed_near, 'o', color=color, )
    plt.xlabel('speed far (0.06m)')
    plt.ylabel('speed near (o.01m)')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
