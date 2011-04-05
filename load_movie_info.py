import datetime
import time
import os
import pickle
import numpy as np

import sys
sys.path.append('/home/floris/src/analysis')
import sa1_analysis as sa1

class MovieDataset:
        def __init__(self):
            self.movies = {}
            
        def get_movie_keys(self, behavior=None, posttype=None, infocus=1, processed=True, error=False, abovepost=False, fake=False):
        
            if behavior == 'fake':
                fake = True
        
            if fake:
                print 'fake'
                return self.movies.keys()
        
            if behavior is None:
                behavior = ['landing', 'flyby']
            else:
                if type(behavior) is not list:
                    behavior = [behavior]
                
            if posttype is None:
                posttype = ['checkered', 'black']
            else:
                if posttype is not list:
                    posttype = [posttype]
                    
            movie_keys = []
            for key, movieinfo in self.movies.items():
                try:
                    tmp = movieinfo.obj_centers
                    del(tmp)
                    isprocessed = True
                except:
                    isprocessed = False
                    
                # for manually coded errors in processing
                try:
                    movieerror = movieinfo.error
                except:
                    movieerror = False
                    
                # check to make sure fly is below the level of the post for the duration of the SA1 video
                if isprocessed:
                    nt, flydra_altitude = sa1.interpolate_to_new_framerate(movieinfo, 100, movieinfo.trajec.epoch_time, movieinfo.trajec.positions[:,2])
                    maxaltitude = np.max(flydra_altitude)
                    if maxaltitude > 0:
                        isabovepost = True
                    else:
                        isabovepost = False
                else:
                    isabovepost = abovepost
                    
                if movieinfo.behavior in behavior and movieinfo.posttype in posttype and movieinfo.infocus == infocus and isprocessed == processed and movieerror == error and isabovepost == abovepost:
                    movie_keys.append(key)
            return movie_keys
            
        def get_movie(self, behavior=None, posttype=None):
            movie_keys = self.get_movie_keys(behavior=behavior, posttype=posttype)
            return self.movies[movie_keys[0]]
                
class MovieInfo:
        pass            
                
def load_movie_info_from_movie_list(movie_list, movie_dataset=None):
        
    if movie_dataset is None:
        movie_dataset = MovieDataset()
        
    #moviepath, moviename, posttype, behavior, subbehavior, startframe, touchdown, endingframe, legextensionrange, infocus
    movie_list_fid = open(movie_list, 'r')
    for line in movie_list_fid.readlines():
        entry = line.split(',')
        
        # clean entry
        for i, enum in enumerate(entry):
            entry[i] = entry[i].rstrip()
            entry[i] = entry[i].lstrip()
            try:
                entry[i] = entry[i].rstrip('\n')
            except:
                pass
            entry[i] = entry[i].rstrip()
            entry[i] = entry[i].lstrip()
            
        movieid = entry[1]
        if movieid not in movie_dataset.movies.keys():
            movieinfo = MovieInfo() 
            movie_dataset.movies.setdefault(movieinfo.id, movieinfo)
        else:
            movieinfo = movie_dataset.movies[movieid]
        
        movieinfo.path = entry[0]
        movieinfo.id = movieid
        movieinfo.posttype = entry[2]
        movieinfo.behavior = entry[3]
        subbehavior = entry[4]
        subbehavior_parsed = subbehavior.split('&')
        movieinfo.subbehavior = subbehavior_parsed 
        movieinfo.fake = False
        
        movieinfo.firstframe_ofinterest = int(entry[5])
        movieinfo.lastframe_ofinterest = int(entry[7])
        try:    
            movieinfo.landingframe = int(entry[6])
            print movieinfo.landingframe
        except:
            movieinfo.landingframe = None
        legextensionentry = entry[8]
        if legextensionentry == 'none':
            movieinfo.legextensionrange = None
        else:
            tmp = legextensionentry.split(':')
            movieinfo.legextensionrange = [int(tmp[0]), int(tmp[1])]
        movieinfo.infocus = int(entry[9])
        
    movie_list_fid.close()
    return movie_dataset
    
def load_movie_info_from_movie_files(movie_dataset):
    
    for key, movie in movie_dataset.movies.items():
        
        movie_info_filename = movie.path + movie.id + '.cih'
        movie_list_fid = open(movie_info_filename, 'r')
        
        line_number = 0
        for line in movie_list_fid.readlines():
            line_number += 1
            entry = line.split(':')
            
            # clean entry
            for i, enum in enumerate(entry):
                entry[i] = entry[i].rstrip()
                entry[i] = entry[i].lstrip()
                try:
                    entry[i] = entry[i].rstrip('\n')
                except:
                    pass
                entry[i] = entry[i].rstrip()
                entry[i] = entry[i].lstrip()
                
            if line_number == 2:
                movie.date_created = entry[1]
            if line_number == 3:
                movie.time_created = [int(entry[1]), int(entry[2])]
            if line_number == 13:
                movie.framerate = float(int(entry[1]))
            if line_number == 18:
                movie.startframe = int(entry[1])
            if line_number == 17:
                movie.totalframes = int(entry[1])
            if line_number == 19:
                movie.triggerframe = int(entry[1])
            if line_number == 21:
                movie.width = int(entry[1])
            if line_number == 22:
                movie.height = int(entry[1])
            
            if line_number >= 22:
                break
            
            
        # get the epoch time of when the movie was created
        # this is needed to fetch the object id from the object id list saved on the flydra computer
        movie_date = map(int, movie.date_created.split('/'))
        movie_time = movie.time_created
    
        date = datetime.date(movie_date[0], movie_date[1], movie_date[2])
        t = date.timetuple()
        timetuple = (t[0], t[1], t[2], movie_time[0], movie_time[1], t[5], t[6], t[7], t[8])
        movie_epochtime = time.mktime(timetuple)
        movie.epochtime_created = movie_epochtime
            
def load_obj_ids_from_directory(directory=None):
    
    if directory is None:
        directory = '/home/floris/data/sa1_movie_data/obj_id_lists/'
    cmd = 'ls ' + directory
    ls = os.popen(cmd).read()
    sa1_obj_id_files = ls.split('\n')
    for i, filename in enumerate(sa1_obj_id_files):
        if len(filename) == 0:
            del(sa1_obj_id_files[i])
        else:
            sa1_obj_id_files[i] = directory + sa1_obj_id_files[i]
    
    obj_id_list = None
    for filename in sa1_obj_id_files:
        ## load SA1 object ID data ##
        infile = open(filename, 'r')
        unpickler = pickle.Unpickler(infile)
        # this bullshit needed to deal with the pickle file, which has multiple dumps.. only the last load() call has the whole file
        new_obj_id_list = unpickler.load()
        while 1:
            try:
                new_obj_id_list = unpickler.load()
            except:
                break
        infile.close()
        
        if obj_id_list == None:
            obj_id_list = new_obj_id_list
        else:
            obj_id_list = np.vstack((obj_id_list, new_obj_id_list))
            
    return obj_id_list
            
def calc_movie_obj_ids(movie_dataset, obj_id_list):
    
    for key, movie in movie_dataset.movies.items():
        movie_time = movie.epochtime_created
        errors = np.zeros_like(obj_id_list[:,1])
        for i in range(len(obj_id_list)):
            errors[i] =  np.abs(obj_id_list[i,1] - movie_time)
        if min(errors) < 200:
            obj_id = int(obj_id_list[ np.argmin(errors), 0])
            trigger_stamp = float(obj_id_list[ np.argmin(errors), 1])
        else:
            obj_id = None
            trigger_stamp = 0
        
        movie.triggerstamp = trigger_stamp
        movie.objid = obj_id        

def calc_flydra_trajectory(movie_dataset, dataset):

    def get_trajectory(movie, dataset):
        t = time.localtime(movie.triggerstamp)
        try:
            obj_id = int(movie.objid)
        except: 
            print 'no obj id associated with movie'
            return None, None
            
        epochtime = movie.triggerstamp
        
        for k, trajectory in dataset.trajecs.items():
            flydra_obj_id = k.lstrip('1234567890')
            flydra_obj_id = flydra_obj_id.lstrip('_')
            flydra_obj_id = int(flydra_obj_id)
            #print 'dataset id: ', o, ' obj id: ', obj_id
            if obj_id == flydra_obj_id:
                # check timestamp:
                time_err = np.abs(trajectory.epoch_time[0] - epochtime)
                #print time_err
                if time_err < 200:
                    #print 'trajectory found'
                    return k, trajectory
            else:
                continue    
                
        #print 'failed to find trajectory'
        return None, None
        
    for key, movie in movie_dataset.movies.items():
        k, trajec = get_trajectory(movie, dataset)
        movie.flydra_trajec_id = k
        movie.trajec = trajec
            
def calc_sa1_timestamps(movie_dataset):
    for key, movie in movie_dataset.movies.items():
        movie.framenumbers = (np.arange(movie.firstframe_ofinterest, movie.lastframe_ofinterest+1, 1)+(-1*movie.startframe)).tolist()
        all_timestamps = np.arange(movie.startframe, movie.totalframes+movie.startframe+1, 1) / float(movie.framerate) + movie.triggerstamp
        movie.timestamps = all_timestamps[movie.framenumbers]
    
        if movie.landingframe is not None:
            movie.landingtime = movie.timestamps[movie.landingframe - movie.firstframe_ofinterest]
            movie.landingframe_relative = movie.landingframe - movie.firstframe_ofinterest
        else:
            movie.landingtime = None
        
def calc_syncronized_timerange(movie_dataset):
    for key, movie in movie_dataset.movies.items():
        if movie.trajec is not None:
            first_sync_time = np.max( [movie.trajec.epoch_time[0], movie.timestamps[0]] )
            last_sync_time = np.min( [movie.trajec.epoch_time[-1], movie.timestamps[-1]] )
            movie.syncrange = [first_sync_time, last_sync_time]
            
def load():

    movie_list = '/home/floris/data/sa1_movie_data/sa1_movie_processing_info/sa1_movie_list_info'
    movie_dataset = load_movie_info_from_movie_list(movie_list)
    load_movie_info_from_movie_files(movie_dataset)
    
    obj_id_list = load_obj_ids_from_directory()
    calc_movie_obj_ids(movie_dataset, obj_id_list)
    calc_sa1_timestamps(movie_dataset)
    
    flydra_dataset_filename = '/home/floris/data/sa1_movie_data/h5_files/reduced_dataset_with_post_types'
    infile = open(flydra_dataset_filename, 'r')
    unpickler = pickle.Unpickler(infile)
    flydra_dataset = unpickler.load()
    infile.close()
    calc_flydra_trajectory(movie_dataset, flydra_dataset)
    
    calc_syncronized_timerange(movie_dataset)
    
    return movie_dataset
            
if __name__ == '__main__':

    movie_dataset = load(movie_dataset)
    
    
    
