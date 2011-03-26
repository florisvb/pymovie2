import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize
from scipy.ndimage.measurements import center_of_mass
import scipy.ndimage as ndimage
import scipy.interpolate
from scipy.ndimage.morphology import binary_erosion

import copy

inf = np.inf

###############################################################################
# Basic Image Processing
###############################################################################

def in_range(val, rang):
    if val > rang[0] and val < rang[1]:
        return True
    else:
        return False
        
def threshold(img, threshold_lo, threshold_hi=255):
    threshed_lo = img>threshold_lo
    threshed_hi = img<threshold_hi
    threshed = threshed_lo*threshed_hi
    threshed *= 255
    return threshed
    
def absdiff(a,b):
    img = np.array(a, dtype=float)
    bkg = np.array(b, dtype=float)
    diff = img-bkg
    absdiff = np.abs(diff)
    return absdiff
    
def compare(a, b=None, method=None):

    if type(a) is list:
        result = a[0]
        for i in range([1,len(a)]):
            if method == 'lighten':
                result = np.maximum(result,a[i])
            elif method == 'darken':
                result = np.minimum(result,a[i])
        return result
    elif b is not None:
        if method is 'lighten':
            result = np.maximum(a,b)
        elif method is 'darken':
            result = np.minimum(a,b)
        return result
    else:
        ValueError('please enter a valid a,b pair')
        
def darken(a, b=None):
    result = compare(a,b,method='darken')
    return result
def lighten(a, b=None):
    result = compare(a,b,method='lighten')
    return result
    
def auto_adjust_levels(img):
    img2 = (img-img.min())
    if img2.max() > 0:
        img3 = img2*int(255/float(img2.max()))
    else:
        img3 = img2
    return img3
    
def get_ellipse_cov(img, erode=False, recenter=True):
    # Pattern. Recogn. 20, Sept. 1998, pp. 31-40
    # J. Prakash, and K. Rajesh
    # Human Face Detection and Segmentation using Eigenvalues of Covariance Matrix, Hough Transform and Raster Scan Algorithms

    #eroded_img = binary_erosion(img)
    #boundary = img-eroded_img
    
    if img is not None:
    
        if erode:
            try:
                img = binary_erosion(img)
            except:
                pass
                
        if recenter:
            center = center_of_blob(img)
        else:
            center = np.array([0,0])

        if 1:
            pts = np.transpose(np.nonzero(img))
            for pt in pts:
                pt -= center
            pts = (np.transpose(np.nonzero(img))).T
            cov = np.cov(pts)
            cov = np.nan_to_num(cov)
            e,v = np.linalg.eig(cov)
            ratio = max(e) / min(e)
            
            i = np.argmax(e)
            long_axis = v[:,i]
        if recenter is False:
            return long_axis, ratio
        else:
            return center, long_axis, ratio
            
    else:
        return [0,0],0
        
def rebin( a, newshape ):
        '''Rebin an array to a new shape.
        '''
        assert len(a.shape) == len(newshape)

        slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
        coordinates = scipy.mgrid[slices]
        indices = coordinates.astype('i')   #choose the biggest smaller integer index
        return a[tuple(indices)]
        
def extract_uimg(img, size, zero):  
    return img[ zero[0]:zero[0]+size[0], zero[1]:zero[1]+size[1] ]
    
def rotate_image(img, rot):
    
    imgrot = np.zeros_like(img)
    
    for r in range(img.shape[0]):
        for w in range(img.shape[1]):
            ptrot = np.dot(rot, np.array([r,w]))
            rrot = ptrot[0]
            wrot = ptrot[1]
            if rrot < 0:
                rrot += img.shape[0]
            if wrot < 0:
                wrot += img.shape[1]
            imgrot[rrot, wrot] = img[r,w]
    return imgrot
    
    
###############################################################################
# Misc Geometry Functions
###############################################################################

def plot_circle(xmesh, ymesh, center, r):
    center = list(center)
    
    def in_circle(x,y):
        R = np.sqrt((X-center[1])**2 + (Y-center[0])**2)
        Z = R<r
        return Z

    x = np.arange(0, xmesh, 1)
    y = np.arange(0, ymesh, 1)
    X,Y = np.meshgrid(x, y)

    Z = in_circle(X, Y)
    
    return Z
    
    
###############################################################################
# Blob Manipulations
###############################################################################


def find_blobs(img, sizerange=[0,inf], aslist=True):
    blobs, nblobs = ndimage.label(img)
    blob_list = []
    if nblobs < 1:
        if aslist is False:
            return np.zeros_like(img)
        else:
            return [np.zeros_like(img)]
    #print 'n blobs: ', nblobs
    # erode filter
    n_filtered = 0
    for n in range(1,nblobs+1):
        blob_size = (blobs==n).sum()
        if not in_range(blob_size, sizerange):
            blobs[blobs==n] = 0
            nblobs -= 1
        else:
            if aslist:
                b = np.array(blobs==n, dtype=np.uint8)
                blob_list.append(b)
            else:
                n_filtered += 1
                blobs[blobs==n] = n_filtered
    
    if aslist is False:
        if nblobs < 1:
            return np.zeros_like(img)
        else:
            blobs = np.array( blobs, dtype=np.uint8)
            return blobs
    else:
        if len(blob_list) < 1:
            blob_list = [np.zeros_like(img)]
        return blob_list
        
def find_blob_nearest_to_point(img, pt):
    centers = center_of_blob(img)
    errs = np.zeros(len(centers))
    for i, center in enumerate(centers):
        errs[i] = np.linalg.norm(center - pt)    
    nearest_index = np.argmin(errs)
    return img[nearest_index]
    
def find_biggest_blob(img):
    blobs, nblobs = ndimage.label(img)
    
    if nblobs < 1:
        return None
    
    if nblobs == 1:
        return blobs

    biggest_blob_size = 0
    biggest_blob = None
    for n in range(1,nblobs+1):
        blob_size = (blobs==n).sum()
        if blob_size > biggest_blob_size:
            biggest_blob = blobs==n
            biggest_blob_size = blob_size
    biggest_blob = np.array(biggest_blob, dtype=np.uint8)
    return biggest_blob 
    
def center_of_blob(img):
    if type(img) is list:
        centers = []
        for blob in img:
            center = np.array([center_of_mass(blob)[i] for i in range(2)])
            centers.append(center)
        return centers
    else:
        center = np.array([center_of_mass(img)[i] for i in range(2)])
        return center
        
###############################################################################
# Background Subtraction
###############################################################################

'''
mask = binary array, 1 and 0
'''

def get_uimg( img_roi, relative_center, uimg_roi_radius ):
    row_lo = np.max( [int(round(relative_center[0]))-uimg_roi_radius, 0] )
    row_hi = np.min( [int(round(relative_center[0]))+uimg_roi_radius, img_roi.shape[0]] )
    col_lo = np.max( [int(round(relative_center[1]))-uimg_roi_radius, 0] )
    col_hi = np.min( [int(round(relative_center[1]))+uimg_roi_radius, img_roi.shape[1]] )
    
    uimg = img_roi[row_lo:row_hi, col_lo:col_hi]
    relative_zero = np.array([row_lo, col_lo])
    
    return uimg, relative_zero

def find_object_with_background_subtraction(img, background, mask=None, guess=None, guess_radius=None, sizerange=[0,inf], thresh=10, uimg_roi_radius=30, return_uimg=True):

    if guess is not None:
        if True in np.isnan(np.array(guess)):
            guess = None
            guess_radius = None
        else:
            guess = np.array(np.round(guess), dtype=int)
            original_guess = copy.copy(guess)
    if guess_radius is not None:
        guess_radius = int(guess_radius)
        
    # explore the full image for objects
    if guess_radius is None:
        img_roi = img
        diff = absdiff(img, background)
        if mask is not None:
            diff *= mask
        zero = np.array([0,0])
    
    # explore just the guessed ROI: MUCH MORE EFFICIENT!!
    if guess_radius is not None:
        row_lo = np.max( [guess[0]-guess_radius, 0] )
        row_hi = np.min( [guess[0]+guess_radius, img.shape[0]] )
        col_lo = np.max( [guess[1]-guess_radius, 0] )
        col_hi = np.min( [guess[1]+guess_radius, img.shape[1]] )
        img_roi = img[row_lo:row_hi, col_lo:col_hi]
        background_roi = background[row_lo:row_hi, col_lo:col_hi]
        diff = absdiff(img_roi, background_roi)
        zero = np.array([row_lo, col_lo])
        guess = np.array([ (row_hi-row_lo)/2. , (col_hi-col_lo)/2. ])
        
    thresh_adj = 0
    blob = []
    while np.sum(blob) <= 0: # use while loop to force finding an object
        diffthresh = threshold(diff, thresh+thresh_adj, threshold_hi=255)
        
        # find blobs
        if guess is not None:
            blobs = find_blobs(diffthresh, sizerange=sizerange, aslist=True)
            if len(blobs) > 1:
                blob = find_blob_nearest_to_point(blobs, guess)
            else:
                blob = blobs[0]
        else:
            blob = find_biggest_blob(diffthresh)
            
        thresh_adj -= 1
        if thresh_adj+thresh <= 0: # failed to find anything at all!!
            center = original_guess
            print 'failed to find object!!!'
            print center, zero
            
            if return_uimg:
                uimg, relative_zero = get_uimg( img_roi, center-zero, uimg_roi_radius )
                zero += relative_zero
                return center, uimg, zero
            else:
                return center    
    
    
    relative_center = center_of_blob(blob)
    center = relative_center + zero
    
    # find a uimg
    if return_uimg:
        uimg, relative_zero = get_uimg( img_roi, relative_center, uimg_roi_radius )
        zero += relative_zero
        return center, uimg, zero
    else:
        return center
        
        
def find_object(img, background=None, threshrange=[1,254], sizerange=[10,400], dist_thresh=10, erode=False, check_centers=False):
    if background is not None:
        diff = absdiff(img, background)
    else:
        diff = img
    imgadj = auto_adjust_levels(diff)
    body = threshold(imgadj, threshrange[0], threshrange[1])
    
    if erode is not False:
        for i in range(erode):
            body = binary_erosion(body)
                    
    if check_centers is False:
        blobs = find_blobs(body, sizerange=sizerange, aslist=False)
    else:
        blobs = find_blobs(body, sizerange=sizerange, aslist=True)
    body = blobs
    
    if check_centers:
        centers = center_of_blob(blobs)
        dist = []
        for center in centers:
            diff = np.linalg.norm( center - np.array(img.shape)/2. )
            dist.append(diff)
        body = np.zeros_like(img)
        for j, d in enumerate(dist):
            if d < dist_thresh:
                body += blobs[j]
                                        
    if body.max() > 1:
        body /= body.max()
    
    if body is None:
        body = np.zeros_like(img)
    
    body = np.array(body*255, dtype=np.uint8)
    
    return body


def find_ellipse(img, background=None, threshrange=[1,254], sizerange=[10,400], dist_thresh=10, erode=False, check_centers=False):
    
    body = find_object(img, background=background, threshrange=threshrange, sizerange=sizerange, dist_thresh=dist_thresh, erode=erode, check_centers=check_centers)
    center, long_axis, ratio = get_ellipse_cov(body, erode=False, recenter=True)
    
    return center, long_axis, body
    
    
