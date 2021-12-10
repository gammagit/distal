from PIL import Image
import numpy as np
import isostim as iso
# import ipdb
import matplotlib.pyplot as plt


size_imx = 224 #196 #224
size_imy = 224 #196 #224

numel = 4 # number of elements, besides cetral element
radius = 20 #17 # radius of each element
cats = [90, 100, 110, 120, 130, 140, 150]  # [90, 110, 130, 150] # each category is determined by total angle subtended

### Parameters for random training rotations
min_trot = 0 # minimum rotation for training images
max_trot = 360 # 45 # 180 # maximum rotation for training images
min_tshear = 0 #-0.007 # minimum rotation for training images
max_tshear = 0 # 180 # maximum rotation for training images

### Init variables that depend on type of distance
dist_type = 'cosine' # 'rotation', 'cosine' or 'euclidean'
if dist_type == 'rotation':
    shears = np.linspace(0, 0.5, num=10)
    numrot = 10
elif dist_type == 'euclidean':
    shears = np.linspace(0, 1.0, num=200)
    min_dist = 1950 # minimum distance between shears
elif dist_type == 'cosine':
    # shears = np.linspace(0, 0.02, num=200) # when shear is linear
    # shears = np.linspace(0, 0.010, num=200) # when shear is quadratic
    shears = np.linspace(0, 0.007, num=200) # quadratic for polygons
    # shears = np.linspace(0, 0.01, num=200) # for JEREMY
    # shears = np.linspace(0, 0.008, num=200) # when shear is quadratic
    # shears = np.linspace(0, 0.00008, num=200) # when shear is pow(3)
    min_dist = 0.005 # minimum distance between shears

teach_rotation = True # teach rotation on some categories
test_category = [2, 3] # category to leave out (for testing) when teach_rotation == True
BASE_PATH = './data/isostim_teach_rot/'
numtrain = 5000 #200
numtest = 100 #20 #10 # number of test images at each location of grid
numtest_cv = 10 #10 # number of test images at each location of grid

### Create a set of baseline shapes
alltest = [] # a (super)list of all test images
for angle_cat in iter(cats):
    ### Construct a base image for category
    print("Creating base images for category {0}    ".format(str(angle_cat)), end="\r")
    im = iso.gen_image(angle_cat, numel, radius, size_imx, size_imy, 0, 0)

    if dist_type == 'euclidean' or dist_type == 'cosine':
        ### Construct a new list of shears s:t distances are monotonically increasing
        ### NOTE: differet list for each cateogry
        # used_shears = []
        # dist_shears = []
        dist_allshears = np.zeros(shears.shape)
        prev_max = -min_dist - 1 # running max dist, initialised neg s:t 0 shear is in the list 
        for shear_ix, shear in enumerate(shears):
            imsh = iso.gen_image(angle_cat, numel, radius, size_imx, size_imy, shear, 0)
            dsh = iso.compute_distance(im, imsh, dist_type)
            dist_allshears[shear_ix] = dsh
            # if dsh > prev_max + min_dist:
            #     used_shears.append(shear)
            #     dist_shears.append(dsh)
            #     prev_max = dsh

        ### Get shears at fixed set of distances
        nshears = 8
        used_shears = []
        dist_shears = []
        max_dist = dist_allshears.max()
        min_dist = dist_allshears.min()
        dist_vec = np.linspace(min_dist, max_dist, nshears) # vector of distances at which to get shears
        for dd in iter(dist_vec):
            ### There are two problems to computing min dist: (i) the distance vs shear function is
            ### non-monotonic, and (ii) distances are only available at discrete set of points.
            ### Therefore, the following code (i) uses max shear that's closest (could also use min/median
            ### but max gives large shears, which is psychologically more interesting), and (ii) computes
            ### closest distances within a small band (delta d)
            delta_dist = (dist_allshears.max() - dist_allshears.min()) / 50 # small distance
            closest_ixs =  np.where(np.abs(dist_allshears - dd) < delta_dist) # Indices within delta d
            while closest_ixs[0].size == 0:
                delta_dist = delta_dist * 2
                closest_ixs =  np.where(np.abs(dist_allshears - dd) < delta_dist) # Indices within delta d
            largest_close_ix = closest_ixs[0].max() # largest because non-monotonic function
            largest_shear_dd = shears[largest_close_ix]
            dist_dd = dist_allshears[largest_close_ix]
            used_shears.append(largest_shear_dd)
            dist_shears.append(dist_dd)
    elif dist_type == 'rotation':
        used_shears = shears


#     plt.plot(dist_allshears)

    catlist = [] # list of lists, each sublist at different shear, each element at diff rotation
    base_rots = None # rotation values at zero shear: other rotations in row will be made close to this
    for ixsh,shear in enumerate(used_shears):
        if dist_type == 'euclidean' or dist_type == 'cosine':
            imsh = iso.gen_image(angle_cat, numel, radius, size_imx, size_imy, shear, 0)
            dsh = iso.compute_distance(im, imsh)
            distvec = dist_shears[ixsh:]
            shlist,shrots = iso.get_isodists(angle_cat, numel, radius, size_imx,
                                      size_imy, shear, distvec, dist_type, base_rots)
            if shear == 0:
                base_rots = shrots
        elif dist_type == 'rotation':
            shlist = iso.get_rotations(angle_cat, numel, radius, size_imx,
                                       size_imy, shear, numrot)
        catlist.append(shlist)

    alltest.append(catlist)

    allim = Image.new('RGB', (size_imx*10 + 40, size_imy*10+40), color='gray')
    x_offset = 0
    y_offset = 0
    for shlist in catlist:
        for im in shlist:
            allim.paste(im, (x_offset, y_offset))
            y_offset += im.size[1] + 2
        x_offset += im.size[0] + 2
        y_offset = 0
    allim.save(BASE_PATH + 'all_b' + str(angle_cat) + '.png')

### Create training data from baseline shapes (translation, scale variation)
trainloc = BASE_PATH + 'train/'
for catix, angle_cat in enumerate(cats):
    print("Creating training data for category {0}    ".format(str(angle_cat)), end="\r")
    ### Create directory for each category
    catdir = trainloc + str(catix+1) + '/'
    iso.create_dir(catdir)

    ### Add transformations of base image to directory
#     baseim = alltest[catix][0][0] # 0 shear, 0 distance
    for imix in range(numtrain):
        # If teach_rotation is true, then train the first category on only upright images,
        # other categories on all rotations
        if (teach_rotation == True) and (catix in test_category):
            roti = 0
        else:
            ### Train random rotations of base image between min_trot & max_trot
            roti = min_trot + np.random.rand() * (max_trot - min_trot)
        sheari = min_tshear + np.random.rand() * (max_tshear - min_tshear)
        baseim = iso.gen_image(angle_cat, numel, radius, size_imx, size_imy, sheari, roti)
        im = iso.gen_transform(baseim) # gen random scale & translation
        # im = baseim ### NOTE: FOR JEREMY -- TO BE REPLACED BY ABOVE LINE !!!!!
        filename = catdir + str(imix+1) + '.png'
        im.save(filename)

testcv = BASE_PATH + 'test_cv/'
for catix, angle_cat in enumerate(cats):
    ### Create directory for each category
    catdir = testcv + str(catix+1) + '/'
    iso.create_dir(catdir)

    ### Add transformations of base image to directory
    baseim = alltest[catix][0][0] # 0 shear, 0 distance
    for imix in range(numtest_cv):
        im = iso.gen_transform(baseim) # gen random scale & translation
        # im = baseim ### NOTE: FOR JEREMY -- TO BE REPLACED BY ABOVE LINE !!!!!
        filename = catdir + str(imix+1) + '.png'
        im.save(filename)


### Create test data from transforms (set from each point on grid)
testloc = BASE_PATH + 'testgrid/'
for catix, angle_cat in enumerate(cats):
    print("Creating test data at category {0}       ".format(str(angle_cat)), end="\r")
    catlist = alltest[catix]

    ### Create directory for each category
    catdir = testloc + str(catix+1) + '/'
    iso.create_dir(catdir)

    ### Create images for all shears
    for shix, shlist in enumerate(catlist):
        ### Create directory for each shear
        shdir = catdir + 'sh_' + str(shix+1) + '/'
        iso.create_dir(shdir)

        ### Create images for all distances at this shear
        for distix, distim in enumerate(shlist):
            ### Create directory for each distance
            ### Also create empty directories for other categories
            for tempcat in range(len(cats)):
                emptydir = shdir + 'dist_' + str(distix+1) + '/' + str(tempcat+1) + '/'
                iso.create_dir(emptydir)
            distdir = shdir + 'dist_' + str(distix+1) + '/' + str(catix+1) + '/'

            for imix in range(numtest):
                im = iso.gen_transform(distim) # gen random scale & translation
                # im = distim  ### NOTE: FOR JEREMY -- TO BE REPLACED BY ABOVE LINE !!!!!
                filename = distdir + str(imix+1) + '.png'
                im.save(filename)

