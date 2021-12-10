import copy
import math
import os
import errno
import numpy as np
import scipy.spatial.distance as scipy_dist
from PIL import Image
from PIL import ImageDraw
# from PIL import ImageFont
# import ipdb


def gen_image(angle_cat, numel, radius, size_imx, size_imy, shear, rot):
    ''' Returns an image from given category at given shear & rotation
    '''
    # max_height = (size_imy/2) - (4*radius) # maximum distance of element from centre
    # min_height = max_height / 2 # maximum distance of element from centre
    height = (size_imy/4) + 15 # distance of element from centre
    # heights = [max_height, min_height, max_height, min_height] # list of el dists (left-to-right)
    theta = [((180 - angle_cat) / 2) + rot] # angle of each element before shear
    xloc = [0] # this list will contain x locations of all elements
    yloc = [0] # this list will contain y locations of all elements
    for el in range(numel): # compute location parameters of each elements
        ### Get angle of each element
        theta.append(theta[-1] + (angle_cat / (numel - 1)))
        theta_orig = theta[el] - rot # angle before rotation

        ### Get xloc, yloc based on angle and height of element
        x_orig = height * np.cos(theta_orig * np.pi/180)
        y_orig = height * np.sin(theta_orig * np.pi/180) + 25

        ### Calculate new xloc, yloc, height based on shear
        # x_dash = x_orig + shear * pow(y_orig,3) # change in xloc after shear: pow(3)
        x_dash = x_orig + shear * pow(y_orig,2) # change in xloc after shear: quadratic
        # x_dash = x_orig + (shear*30) * y_orig # change in xloc after shear: linear
        # x_dash = x_orig + (shear*10) * y_orig + shear * pow(y_orig,2) # linear + non-linear
#         y_dash = y_orig + shear * pow(x_orig,2) # equal shear in y direction
#         x_dash = x_orig + (shear*30) * y_orig # change in xloc after shear
#         y_dash = y_orig - (shear*1000) * x_orig # equal shear in y direction
        y_dash = y_orig # replace above if no shear in y direction
        height_el = np.sqrt(pow(x_dash, 2) + pow(y_dash, 2)) # new hypotenuse
        theta_shear = math.degrees(math.atan(abs(y_dash / x_dash))) # angle after shear

        ### Convert from [-90,90] to [0,360]
        if x_dash < 0 and y_dash > 0:
            theta_shear = 180 - theta_shear
        elif x_dash < 0 and y_dash < 0:
            theta_shear = 180 + theta_shear
        elif x_dash > 0 and y_dash < 0:
            theta_shear = 360 - theta_shear
        theta_shear_rot = theta_shear + rot # angle after shear as well as rotation

        ### Calculate final xloc, yloc based on height after shear & rotation
        xloc.append(height_el * np.cos(theta_shear_rot * np.pi/180))
        yloc.append(-height_el * np.sin(theta_shear_rot * np.pi/180))

    ### Add another row at half-way distance: to increase dots, so that rotation become monotonic
    two_rows = False
    if two_rows == True:
        height = height * (1/2)
        theta = [((180 - angle_cat) / 2) + rot] # angle of each element before shear
        for el in range(numel): # compute location parameters of each elements
            ### Get angle of each element
            theta.append(theta[-1] + (angle_cat / (numel - 1)))
            theta_orig = theta[el] - rot # angle before rotation

            ### Get xloc, yloc based on angle and height of element
            x_orig = height * np.cos(theta_orig * np.pi/180)
            y_orig = height * np.sin(theta_orig * np.pi/180) + 25

            ### Calculate new xloc, yloc, height based on shear
            x_dash = x_orig + shear * pow(y_orig,2) # change in xloc after shear: quadratic
            y_dash = y_orig # replace above if no shear in y direction
            height_el = np.sqrt(pow(x_dash, 2) + pow(y_dash, 2)) # new hypotenuse
            theta_shear = math.degrees(math.atan(abs(y_dash / x_dash))) # angle after shear

            ### Convert from [-90,90] to [0,360]
            if x_dash < 0 and y_dash > 0:
                theta_shear = 180 - theta_shear
            elif x_dash < 0 and y_dash < 0:
                theta_shear = 180 + theta_shear
            elif x_dash > 0 and y_dash < 0:
                theta_shear = 360 - theta_shear
            theta_shear_rot = theta_shear + rot # angle after shear as well as rotation

            ### Calculate final xloc, yloc based on height after shear & rotation
            xloc.append(height_el * np.cos(theta_shear_rot * np.pi/180))
            yloc.append(-height_el * np.sin(theta_shear_rot * np.pi/180))

    if rot <= 0: # during testing
        ### Shift all elements to centre-left of image
        xloc = [xx + size_imx/3 for xx in xloc]
        yloc = [yy + 2*size_imy/3 for yy in yloc]
    else: # during training
        ### Shift all elements to centre-right of image
        xloc = [xx + size_imx/2 for xx in xloc]
        yloc = [yy + 2*size_imy/3 for yy in yloc]
        if any(xx > size_imx for xx in xloc):
            xloc = [xx - 50 for xx in xloc]
        if any(yy > size_imy for yy in yloc):
            yloc = [yy - 50 for yy in yloc]


    ### Draw the image
    im = Image.new('RGB', (size_imx, size_imy), color='white')
    drawing = ImageDraw.Draw(im)
    shape = 'polygon' # 'polygon' / 'dots' / 'both'
    if shape == 'dots':
        for el in range(len(xloc)):
            drawing.ellipse([(xloc[el]-radius, yloc[el]-radius),
                            (xloc[el]+radius, yloc[el]+radius)], fill='black')
    elif shape == 'polygon':
        xy = []
        for el in zip(xloc, yloc):
            xy += el
        drawing.polygon(xy, fill='black', outline='black')
    elif shape == 'both':
        xy = []
        for el in zip(xloc, yloc):
            xy += el
        drawing.polygon(xy, fill=(200, 200, 200), outline=(200, 200, 200))
        for el in range(len(xloc)):
            drawing.ellipse([(xloc[el]-radius, yloc[el]-radius),
                             (xloc[el]+radius, yloc[el]+radius)], fill='black')

    return(im)


def compute_distance(im1, im2, dist_type='cosine'):
    ''' Returns the Euclidean or Cosine distance between two images
    '''
    arr1 = np.array(im1)
    arr2 = np.array(im2)
    if dist_type == 'euclidean':
        eudist = np.sum((arr1-arr2)**2) # Euclidean distance
        return eudist
    elif dist_type == 'cosine':
        flatarr1 = arr1.ravel() # flatten the 3D to 1D
        flatarr2 = arr2.ravel()
        cosdist = scipy_dist.cosine(flatarr1, flatarr2)
        abs_cosdist = abs(cosdist)  # Added as polygon+dots leads to negative distance
        return abs_cosdist
        # return cosdist


def find_nearest(dist, distlist, rotlist, base_rot):
    '''
    Return rotation with minimum distance to dist. If this gives multiple rotations,
    return the rotation with and minimum distance to base_rot
    Arguments:
        dist: Distance required (i.e. find rotation at this distance)
        distlist: List of distances at all rotations, given shear
        rotlist: List of rotations that correspond to distlist
        base_rot: List of rotations in first column of testgrid
    '''
    distlist = np.asarray(distlist, dtype=np.float64) # Convert list to numpy array
    if dist > distlist.max():
        raise Exception('Required dist = {0} is larger than dist of max rotation = {1}'.format(dist, distlist.max()))
    delta_dist = (distlist.max() - distlist.min()) / 50 # small distance
    closest_ixs =  np.where(np.abs(distlist - dist) < delta_dist) # Indices within delta d
    while closest_ixs[0].size == 0:
        delta_dist = delta_dist * 2
        closest_ixs =  np.where(np.abs(distlist - dist) < delta_dist) # Indices within delta d

    ### Since distance may be a non-monotonic function of rotation/shear, one rotation/shear
    ### may correspond to multiple distances. Therefore, choose one of these
    if closest_ixs[0].size > 1: # choose the one closest to base_rot
        rots_at_closest_ixs = [rotlist[ii] for ii in closest_ixs[0]] # list of rotations at closest_ixs
        rots_at_closest_ixs = np.asarray(rots_at_closest_ixs, dtype=np.float64) # to use np.where next
        min_rotix_at_mindist =  (np.abs(rots_at_closest_ixs - base_rot)).argmin()
        minrot = rots_at_closest_ixs[min_rotix_at_mindist]
    elif closest_ixs[0].size == 1: # only one distance that falls within delta_dist
        minrot = rotlist[closest_ixs[0][0]]

    return minrot


def get_isodists(cat, numel, radius, size_imx, size_imy, shear, distvec, dist_type, base_rots):
    ''' Returns list of images, all at the given shear, but different
        rotations s:t rotations correspond to distances in distvec

    Input arguments
    base_rots:  a list containing rotations at shear=0, used for finding closest
                rotations, when multiple rotations at particular shear
    '''
    imrots = [] # list of rotations at given shear and dist in distvec
    min_rot = 0
    max_rot = -90
    step_rot = -0.1

    ### Generate distances of all rotations
    im_orig = gen_image(cat, numel, radius, size_imx, size_imy, 0, 0)
    distlist = []
    rotlist = []
    for rot in np.arange(min_rot, max_rot, step_rot):
        im_rot = gen_image(cat, numel, radius, size_imx, size_imy, shear, rot)
        rotlist.append(rot)
        dist = compute_distance(im_orig, im_rot, dist_type)
        distlist.append(dist)

    ### Get images with closest distances to distvec
    best_rots = []
    for dist_ix, dist in enumerate(distvec):
        if base_rots is not None:
            ### Compute distance of image in first column (zero shear) and same row as current object
            base_rot_dd = base_rots[dist_ix]
        else:
            base_rot_dd = 0
        best_rot = find_nearest(dist, distlist, rotlist, base_rot_dd)
        best_im = gen_image(cat, numel, radius, size_imx, size_imy, shear, best_rot)
        imrots.append(copy.deepcopy(best_im))
        best_rots.append(best_rot)

    '''
    ### DEBUG: Add distance to each image (for debug)
    for im in imrots:
        dist = compute_distance(im_orig, im)
        drawing = ImageDraw.Draw(im)
        font = ImageFont.truetype("/usr/share/fonts/TTF/arial.ttf", 20)
        drawing.text((0, 0), str("{0:0.5f}".format(shear)) + ', ' + str("{0:0.3f}".format(dist)), (0, 0, 0), font=font)
    '''

    return (imrots, best_rots)


def get_rotations(cat, numel, radius, size_imx, size_imy, shear, numrot=10):
    ''' Returns list of images, all at the given shear, but different
        rotations 
    '''
    min_rot = 0
    max_rot = -90

    ### Generate distances of all rotations
    imlist = []
    for rot in np.linspace(min_rot, max_rot, numrot):
        im_rot = gen_image(cat, numel, radius, size_imx, size_imy, shear, rot)
        imlist.append(im_rot)

    return(imlist)


def create_dir(dirname):
    ''' Create the given directory name if it doesn't already exist
    '''
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def gen_transform(im):
    ''' Generate a random translation and scale transform of image
    '''
    min_scale = 2 #1 # minimum scale transform
    max_scale = 4 # maximum scale transform
    max_trans = im.size[0] * 2 # maximum translation transform
    new_sizex = (im.size[0] * max_scale) + max_trans
    new_sizey = (im.size[1] * max_scale) + max_trans

    scale = min_scale + np.random.rand() * (max_scale - min_scale)

    imcopy = im.copy()
    imcopy = imcopy.convert('RGBA')

    ### Scale
    imcopy = imcopy.resize((int(math.ceil(im.size[0] * scale)),
                            int(math.ceil(im.size[1] * scale))),
                            resample=Image.LANCZOS)

    ### Make sure background is white
    fff = Image.new('RGBA', imcopy.size, (255,)*4)
    imcopy = Image.composite(imcopy, fff, imcopy)
    imcopy = imcopy.convert(im.mode)

    ### Translate (paste into new image)
    newim = Image.new('RGB', (new_sizex, new_sizey), color='white')
    tx_ii = int(math.floor(np.random.rand() * max_trans))
    ty_ii = int(math.floor(np.random.rand() * max_trans))
    newim.paste(imcopy, (tx_ii, ty_ii))

    ### Resize it to original size
    newim = newim.resize((im.size[0], im.size[1]), resample=Image.LANCZOS)

    return(newim)
