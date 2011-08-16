import numpy
import numpy as np
from pyglet import media
import numpyimgproc as nim
import copy
import pickle

import load_movie_info

###
def save(data, filename):
    fd = open( filename, mode='w' )
    pickle.dump(data, fd)
    fd.close()
    return 1
def load(filename):
    fd = open( filename, mode='r')
    print 'loading data... from file'
    data = pickle.load(fd)
    return data

###
class Frame:
    def __init__(self):
        self.tmp = 0

###
class Movie:
    def __init__(self, filename=None):
        self.mask = None
        self.filename = filename
        if filename is not None:
            self.load_source(filename)
    
    def load_source(self, filename):
        print 'loading source: ', filename
        self.source = media.load(filename)
        try:
            self.source = media.load(filename)
            print '1'
            self.width = int(self.source.video_format.width)
            self.height = int(self.source.video_format.height)
            self.duration = self.source.duration
            print 'loaded movie.. height: ', self.height, ' | width: ', self.width
        except:
            print 'failed to open movie!'
            ValueError('failed to open movie!')
        self.calc_playback_rate()
            
    def get_next_frame(self):
        imdata = self.source.get_next_video_frame()
        a = numpy.frombuffer(imdata.data, numpy.uint8)
        a.shape = (imdata.height, imdata.width, 3)
        del(imdata)
        raw = (a[:,:,0]+a[:,:,1]+a[:,:,2])/3 #convert to rawchrome
        raw = copy.copy(raw)
        del(a)
        if self.mask is not None:
            raw += np.ones_like(raw)
            raw *= self.mask
            # delete rows that are all zeros:
            nz = (raw == 0).sum(1)
            raw = raw[nz == 0, :]
            # delete columns that are all zeros:
            nz = (raw == 0).sum(0)
            raw = raw[:, nz == 0]
            raw -= np.ones_like(raw)
        return raw
        
    def get_frame_at_timestamp(self, timestamp):
        self.seek(timestamp)
        raw = self.get_next_frame()
        return raw
            
    def seek(self, timestamp=0):
        self.source._seek(timestamp)
        
    def calc_playback_rate(self):
        self.seek(0)
        t = self.source.get_next_video_timestamp()
        self.framerate = 1/t
        
    def frame_to_timestamp(self, frame):
        timestamp = frame / self.framerate
        return timestamp
    def seek_to_frame(self, frame):
        timestamp = self.frame_to_timestamp(frame)
        self.seek(timestamp) 
    def get_frame(self, frame):
        timestamp = self.frame_to_timestamp(frame)
        raw = self.get_frame_at_timestamp(timestamp)
        return raw

###        
def get_uimg_list(movieinfo):
    return [movieinfo.frames[i].uimg for i in range(len(movieinfo.frames))]

###   
def process_movie(movieinfo, framerange=None, nframes=None):

    filename = movieinfo.path + movieinfo.id + '.avi'
    movie = Movie(filename)
    
    # make a mask to remove text and such of the top part of the movie
    mask = np.ones([movie.height, movie.width])
    mask[0:movie.height-movie.width, :] = 0
    movie.mask = mask
    tracking_mask = np.abs(nim.plot_circle(1024,1024,[512,512], 150)-1)
    
    first_frame = (movie.get_frame_at_timestamp(0)[::-1, :])
    last_frame = (movie.get_frame_at_timestamp(movie.duration-1)[::-1, :])
    background = nim.lighten(first_frame, last_frame)
    movieinfo.background = copy.copy(background)
    
    if framerange is None:
        framerange = movieinfo.framenumbers
    else:
        framerange = np.arange(framerange[0], framerange[1]+1, 1).tolist()
    if nframes is not None:
        framerange = framerange[0:nframes]
    nframes = len(framerange)
    
    movieinfo.obj_centers = np.zeros([nframes, 2])
    movieinfo.obj_longaxis = np.zeros([nframes, 2])
    movieinfo.obj_shortaxis = np.zeros([nframes, 2])
    movieinfo.obj_ratio = np.zeros([nframes, 2])
    movieinfo.frames = [Frame() for frame in framerange]
    
    print 'framerange = ', framerange[0]
    for f, framenumber in enumerate(framerange):
        frame = movieinfo.frames[f]
        
        print f, framenumber
        raw = (movie.get_frame(framenumber)[::-1, :])
        
        if f < 3:
            guess = None
            guess_radius = None
        else:
            try:
                vel_est = movieinfo.obj_centers[f-1,:] - movieinfo.obj_centers[f-2,:]
                guess = movieinfo.obj_centers[f-1,:] + vel_est
                guess_radius = 35
            except:
                guess = None
                guess_radius = None
                
        
        center, uimg, zero = nim.find_object_with_background_subtraction(raw, background, mask=tracking_mask, guess=guess, guess_radius=guess_radius, sizerange=[10,600], thresh=10, uimg_roi_radius=30, return_uimg=True, return_mimg=False)
        
        print '********pygmovie********', uimg.shape, zero
        
        ubackground = nim.extract_uimg(background, uimg.shape, zero)
        
        relative_center_of_body, longaxis, shortaxis, body, ratio = nim.find_ellipse(uimg, background=ubackground, threshrange=[150,254], sizerange=[10,600], dist_thresh=10, erode=False, check_centers=True)
        
        center_of_body = relative_center_of_body + zero
        #center_of_body = center
        
        frame.uimg = copy.copy(uimg)
        #frame.mimg = mimg
        frame.zero = zero
        frame.framenumber = framenumber
        
        movieinfo.obj_centers[f,:] = copy.copy(center_of_body)
        movieinfo.obj_longaxis[f,:] = copy.copy(longaxis)
        movieinfo.obj_shortaxis[f,:] = copy.copy(shortaxis)
        movieinfo.obj_ratio[f,:] = copy.copy(ratio)
        
        del(raw)
    del(movie)
    
    return    
    #return movieinfo        
    
###
def reprocess_uframes(movieinfo):
    print movieinfo.id
    
    framerange = movieinfo.framenumbers
    print framerange[0], framerange[-1]
    
    nframes = len(framerange)
    
    movieinfo.obj_centers = np.zeros([nframes, 2])
    movieinfo.obj_longaxis = np.zeros([nframes, 2])
    movieinfo.obj_shortaxis = np.zeros([nframes, 2])
    movieinfo.obj_ratio = np.zeros([nframes, 2])
    
    for f, framenumber in enumerate(framerange):
        print f
        frame = movieinfo.frames[f] 
        uimg = frame.uimg
        zero = frame.zero
        
        
        ubackground = nim.extract_uimg(movieinfo.background, uimg.shape, zero)
            
        relative_center_of_body, longaxis, shortaxis, body, ratio = nim.find_ellipse(uimg, background=ubackground, threshrange=[150,254], sizerange=[10,600], dist_thresh=10, erode=False, check_centers=True)
        
        center_of_body = relative_center_of_body + zero
        #center_of_body = center
        
        movieinfo.obj_centers[f,:] = center_of_body
        movieinfo.obj_longaxis[f,:] = longaxis
        movieinfo.obj_shortaxis[f,:] = shortaxis
        movieinfo.obj_ratio[f,:] = ratio
        
###
def reprocess_movie_dataset(movie_dataset):
    keys = movie_dataset.get_movie_keys(processed=True)
    for key in keys:
        movieinfo = movie_dataset.movies[key]
        reprocess_uframes(movieinfo) 
###
def batch_process_movies(saveas='movie_dataset_extended', movie_dataset=None):

    movie_dataset = load_movie_info.load(movie_dataset)
    movie_dataset_empty = load_movie_info.load()
    
    save(movie_dataset, saveas)
    del(movie_dataset)
    
    
    
    for k, movie in movie_dataset_empty.movies.items():  
    
        # check to make sure movie hasn't been processed already
        try:
            tmp = movie.frames
            print 'already processed, skipping: ', k
            del(tmp)
            continue
        except:
            pass
            
        if movie.infocus == 1:
            movieinfo = copy.copy(movie)
            process_movie(movieinfo)
            
            movie_dataset = load(saveas)
            movie_dataset.movies[k] = movieinfo 
            save(movie_dataset, saveas)
            del(movie_dataset)
    

    
    
    
    
    
    
