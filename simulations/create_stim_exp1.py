from PIL import Image
from PIL import ImageDraw
import numpy
import math

size_imx = 100 # size(x) of image
size_imy = 100 # size(y) of image

fsize_imx = 196 # final size(x) of image - so that conv+pool leads to 11x11
fsize_imy = 196 # final size(y) of image

imlist = []
v1list = []
v2list = []
v3list = []
foillist = []

teach_v1 = True
teach_v1_exclude_exact = True
train_trans = True
train_rots = True
train_scales = True

if teach_v1 is True:
    # BASE_PATH = './data/teach_v1/'
    # BASE_PATH = './data/teach_v1_not_v3/'
    BASE_PATH = './data/teach_v1_exclude_left/'
else:
#     BASE_PATH = './data/orig/'
#     BASE_PATH = './data/no_rot_scale/'
    BASE_PATH = './data/no_rot_scale_trans/'

### Eight object types based on lines are left/right or above/below
ot = [(p,q,r,s) for p in range(2) for q in range(2)
                for r in range(2) for s in range(2)]

ul = 10 # unit length
len1 = ul * (3/2) # length of the first segment
len2 = ul * 6
len3 = ul * 4
len4 = ul * 5
len5 = ul * (3/2)
midx = size_imx / 2
midy = size_imx / 2

allim = Image.new('RGB', (size_imx + 4, size_imy*16+32), color='gray')
allv1 = Image.new('RGB', (size_imx + 4, size_imy*16+32), color='gray')
allv2 = Image.new('RGB', (size_imx + 4, size_imy*16+32), color='gray')
allv3 = Image.new('RGB', (size_imx + 4, size_imy*16+32), color='gray')
allfoil = Image.new('RGB', (size_imx + 4, size_imy*16+32), color='gray')

y_offset = 0
count = 1
for oo in ot:
    x1s = midx - (ul * 3/4)
    x1e = x1s + len1
    y1s = midy + 3*ul
    y1e = midy + 3*ul
    x2s = midx
    x2e = midx
    y2s = midy + 3*ul
    y2e = y2s - len2
    if oo[0] == 0:
        x3s = midx - (3+1/2)*ul
        x3e = x3s + len3
    else:
        x3s = midx - ul# ul/2
        x3e = x3s + len3
#     y3s = midy + (2+1/2)*ul
#     y3e = midy + (2+1/2)*ul
    y3s = midy + (2)*ul
    y3e = midy + (2)*ul
    if oo[1] == 0:
        x4s = midx - ul #ul/2
        x4e = x4s + len4
    else:
        x4s = midx - (3+1/2)*ul
        x4e = x4s + len4
    y4s = midy - (1+1/2)*ul ### original
#     y4s = midy - (1+3/4)*ul
#     y4s = midy - 2*ul + (numpy.random.rand() * 1.5 * ul) ### variable
    y4e = y4s
    if oo[2] == 0:
        y5s = y4e + ul/2
#         y5s = midy - (3/4)*ul
    else:
        y5s = y4e + (1+1/4) * ul
#         y5s = midy
    y5e = y5s - len5
    if oo[3] == 0:
        x5s = x4e
        x5e = x4e
    else:
        x5s = x4s
        x5e = x4s

    ### Create empty image
    im = Image.new('RGB', (size_imx, size_imy), color='white')
    v1 = Image.new('RGB', (size_imx, size_imy), color='white')
    v2 = Image.new('RGB', (size_imx, size_imy), color='white')
    v3 = Image.new('RGB', (size_imx, size_imy), color='white')
    foil = Image.new('RGB', (size_imx, size_imy), color='white')

    ### Draw shape on it
    drawing_b = ImageDraw.Draw(im)
    drawing_v1 = ImageDraw.Draw(v1)
    drawing_v2 = ImageDraw.Draw(v2)
    drawing_v3 = ImageDraw.Draw(v3)
    drawing_foil = ImageDraw.Draw(foil)

    ### Draw each segment for basis image
    drawing_b.line([(x1s, y1s), (x1e, y1e)], fill=0, width=2) # base
    drawing_b.line([(x2s, y2s), (x2e, y2e)], fill=0, width=2) # vertical
    drawing_b.line([(x3s, y3s), (x3e, y3e)], fill=0, width=2) # base
    drawing_b.line([(x4s, y4s), (x4e, y4e)], fill=0, width=2) # base
    drawing_b.line([(x5s, y5s), (x5e, y5e)], fill=0, width=2) # base
    imlist.append(im)

    shift = (3/4) * ul

    ### Variant 1
    if oo[2] == 0:
        shift_v1 = shift
    else:
        shift_v1 = -shift
    y5s_v1 = y5s + shift_v1
    y5e_v1 = y5e + shift_v1
#     if oo[2] == 0: # reverse the y-position of segment 5
#         y5s = midy ### Close to original in HS96
        ###y5s = midy + (1/2) * ul ### Extreme
#         y5e = y5s - len5
#     else:
#         y5s = midy - (3/4)*ul ### Close to original in HS96
        ###y5s = midy - (1+1/4) * ul ### Extreme
#         y5e = y5s - len5
    drawing_v1.line([(x1s, y1s), (x1e, y1e)], fill=0, width=2) # base
    drawing_v1.line([(x2s, y2s), (x2e, y2e)], fill=0, width=2) # vertical
    drawing_v1.line([(x3s, y3s), (x3e, y3e)], fill=0, width=2) # base
    drawing_v1.line([(x4s, y4s), (x4e, y4e)], fill=0, width=2) # base
    drawing_v1.line([(x5s, y5s_v1), (x5e, y5e_v1)], fill=0, width=2) # base
    v1list.append(v1)

    ### Variant 2: shift y-pos of segments 4 & 5
    shift_v2 = shift ### original
#     shift_v2 = 3*shift ### extreme
    y4s_v2 = y4s + shift_v2
    y4e_v2 = y4e + shift_v2
    y5s_v2 = y5s + shift_v2
    y5e_v2 = y5e + shift_v2
#     shift = 2*ul # Extreme
#     y4s = midy - (1+1/2)*ul + shift
#     y4e = midy - (1+1/2)*ul + shift
#     if oo[2] == 0:
#         y5s = midy - (3/4)*ul + shift
        ###y5s = midy - (1+1/10)*ul + shift
#         y5e = y5s - len5
#     else:
#         y5s = midy + shift
        ###y5s = midy - (1/6)*ul + shift
        ###y5s = midy - (3/4)*ul + shift
#         y5e = y5s - len5
    drawing_v2.line([(x1s, y1s), (x1e, y1e)], fill=0, width=2) # base
    drawing_v2.line([(x2s, y2s), (x2e, y2e)], fill=0, width=2) # vertical
    drawing_v2.line([(x3s, y3s), (x3e, y3e)], fill=0, width=2) # base
    drawing_v2.line([(x4s, y4s_v2), (x4e, y4e_v2)], fill=0, width=2) # base
    drawing_v2.line([(x5s, y5s_v2), (x5e, y5e_v2)], fill=0, width=2) # base
    v2list.append(v2)

    ### Variant 3: shift y-pos of segments 1 - i.e. segment 1 at top
    y1s_v3 = y2e
    y1e_v3 = y1s_v3
    drawing_v3.line([(x1s, y1s_v3), (x1e, y1e_v3)], fill=0, width=2) # base
    drawing_v3.line([(x2s, y2s), (x2e, y2e)], fill=0, width=2) # vertical
    drawing_v3.line([(x3s, y3s), (x3e, y3e)], fill=0, width=2) # base
    drawing_v3.line([(x4s, y4s), (x4e, y4e)], fill=0, width=2) # base
    drawing_v3.line([(x5s, y5s), (x5e, y5e)], fill=0, width=2) # base
    v3list.append(v3)

    ### Foil
    shift_foil = shift
    y3s_foil = y3s + ul/2
    y3e_foil = y3s_foil
    y4s_foil = y4s + shift_foil
    y4e_foil = y4e + shift_foil
    y5s_foil = y5s_v1 + shift_foil
    y5e_foil = y5e_v1 + shift_foil
    drawing_foil.line([(x1s, y1s), (x1e, y1e)], fill=0, width=2) # base
    drawing_foil.line([(x2s, y2s), (x2e, y2e)], fill=0, width=2) # vertical
    drawing_foil.line([(x3s, y3s_foil), (x3e, y3e_foil)], fill=0, width=2) # base
    drawing_foil.line([(x4s, y4s_foil), (x4e, y4e_foil)], fill=0, width=2) # base
    drawing_foil.line([(x5s, y5s_foil), (x5e, y5e_foil)], fill=0, width=2) # base
    foillist.append(foil)

    allim.paste(im, (2, y_offset))
    allv1.paste(v1, (2, y_offset))
    allv2.paste(v2, (2, y_offset))
    allv3.paste(v3, (2, y_offset))
    allfoil.paste(foil, (2, y_offset))
    y_offset += im.size[1] + 2

allim.save(BASE_PATH + 'all_b.png')
allv1.save(BASE_PATH + 'all_v1.png')
allv2.save(BASE_PATH + 'all_v2.png')
allv3.save(BASE_PATH + 'all_v3.png')
allfoil.save(BASE_PATH + 'all_foil.png')

if teach_v1 is not True:
    objs_b = [imlist[0], imlist[8], imlist[14], imlist[13], imlist[5], imlist[3]]
    objs_v1 = [v1list[0], v1list[8], v1list[14], v1list[13], v1list[5], v1list[3]]
    objs_v2 = [v2list[0], v2list[8], v2list[14], v2list[13], v2list[5], v2list[3]]
else:
    if teach_v1_exclude_exact is False:
        ### Teach V1 variants are different for five (out of six) categories
        # objs_b = [imlist[0], imlist[8], imlist[14], imlist[13], imlist[5], v1list[0], v1list[8], v1list[14], v1list[13], v1list[5], imlist[3], foillist[3]]
        # objs_v1 = [v1list[0], v1list[8], v1list[14], v1list[13], v1list[5], imlist[0], imlist[8], imlist[14], imlist[13], imlist[5], v1list[3], foillist[3]]
        # objs_v2 = [v2list[0], v2list[8], v2list[14], v2list[13], v2list[5], v2list[2], v2list[10], v2list[12], v2list[15], v2list[7], v2list[3], foillist[3]]
        objs_b = [imlist[0], imlist[8], imlist[14], imlist[13], imlist[3], v1list[0], v1list[8], v1list[14], v1list[13], v1list[3], imlist[5]]
        objs_v1 = [v1list[0], v1list[8], v1list[14], v1list[13], v1list[3], imlist[0], imlist[8], imlist[14], imlist[13], imlist[3], v1list[5]]
        objs_v2 = [v2list[0], v2list[8], v2list[14], v2list[13], v2list[3], v2list[2], v2list[10], v2list[12], v2list[15], v2list[7], v2list[5]]
        objs_v3 = [v3list[0], v3list[8], v3list[14], v3list[13], v3list[3], v3list[0], v3list[8], v3list[14], v3list[13], v3list[3], v3list[5]]
    else: # Exclude imlist[13] which varies the exact relational change to be tested (between imlist[5] and v1list[5])
        objs_b = [imlist[0], imlist[8], imlist[14], imlist[4], v3list[0], v1list[0], v1list[8], v1list[14], v1list[4], v3list[2], imlist[5]]
        objs_v1 = [v1list[0], v1list[8], v1list[14], v1list[4], v1list[3], imlist[0], imlist[8], imlist[14], imlist[4], imlist[3], v1list[5]]
        objs_v2 = [v2list[0], v2list[8], v2list[14], v2list[4], v2list[3], v2list[2], v2list[10], v2list[12], v2list[6], v2list[7], v2list[5]]
        objs_v3 = [v3list[0], v3list[8], v3list[14], v3list[4], v3list[3], v3list[0], v3list[8], v3list[14], v3list[4], v3list[3], v3list[5]]

### Varition along y4 - chosen to increase confusion
# objs_b = [imlist[0], imlist[1], imlist[6], imlist[7], imlist[8], imlist[9]]
# objs_v1 = [v1list[0], v1list[1], v1list[6], v1list[7], v1list[8], v1list[9]]
# objs_v2 = [v2list[0], v2list[1], v2list[6], v2list[7], v2list[8], v2list[9]]


### Create some images with stimset (for debugging)
bigim = Image.new('RGB', (size_imx + 4, size_imy*12+32), color='gray')
bigv1 = Image.new('RGB', (size_imx + 4, size_imy*12+32), color='gray')
bigv2 = Image.new('RGB', (size_imx + 4, size_imy*12+32), color='gray')
bigv3 = Image.new('RGB', (size_imx + 4, size_imy*12+32), color='gray')
y_offset = 0
for oix,oo in enumerate(objs_b):
    bigim.paste(oo, (2, y_offset))
    y_offset += oo.size[1] + 2
bigim.save(BASE_PATH + 'stim_b.png')

y_offset = 0
for oix,oo in enumerate(objs_v1):
    bigv1.paste(oo, (2, y_offset))
    y_offset += oo.size[1] + 2
bigv1.save(BASE_PATH + 'stim_v1.png')

y_offset = 0
for oix,oo in enumerate(objs_v2):
    bigv2.paste(oo, (2, y_offset))
    y_offset += oo.size[1] + 2
bigv2.save(BASE_PATH + 'stim_v2.png')

y_offset = 0
for oix,oo in enumerate(objs_v3):
    bigv3.paste(oo, (2, y_offset))
    y_offset += oo.size[1] + 2
bigv3.save(BASE_PATH + 'stim_v3.png')

### Select objects used in Hummel & Stankiewicz (1996)
ntrain = 5000 # number of exemplars of each object to train
ntest = 1000 # number of exemplars of each object to train
ntest_cv = 500 # cross-validation set
if train_trans is True:
    max_trans = 50 # maximum translation in pixels
else:
    max_trans = 0
if train_rots is True:
    max_rot = 40 # maximum rotation (in degrees)
else:
    max_rot = 0
if train_scales is True:
    min_scale = 1/2 # minimum scale
else:
    min_scale = 1

size_obx = int(math.ceil(numpy.sqrt(2) * size_imx + max_trans)) # so it fit rotated & trans img
size_oby = int(math.ceil(numpy.sqrt(2) * size_imy + 50))

### Tranform image (translate, scale, rotate)
count_obj = 1
for oix,oo in enumerate(objs_b):
    count_im = 1
    for ii in range(ntrain):
        tx_ii = int(math.floor(numpy.random.rand() * max_trans))
        ty_ii = int(math.floor(numpy.random.rand() * max_trans))
        rot_ii = numpy.random.rand() * max_rot - (max_rot/2)
        scale_ii = min_scale + numpy.random.rand() * (1 - min_scale)

        obj_ii = oo.copy()
        obj_ii = obj_ii.convert('RGBA')
        ### Scale
        obj_ii = obj_ii.resize((int(math.ceil(size_imx * scale_ii)),
                                int(math.ceil(size_imy * scale_ii))),
                                resample=Image.LANCZOS)
        ### Rotate
        obj_ii = obj_ii.rotate(rot_ii, resample=Image.BICUBIC, expand=1)
        ### Make sure background is white
        fff = Image.new('RGBA', obj_ii.size, (255,)*4)
        obj_ii = Image.composite(obj_ii, fff, obj_ii)
        obj_ii = obj_ii.convert(oo.mode)
        newim = Image.new('RGB', (size_obx, size_oby), color='white')
        ### Paste into new image
        newim.paste(obj_ii, (tx_ii, ty_ii))
        ### Resize it to original size
        newim = newim.resize((fsize_imx, fsize_imy), resample=Image.LANCZOS)
        
        newim.save(BASE_PATH + 'train/' + str(count_obj) + '/image_' + str(count_im) + '.png')
        count_im += 1

    count_obj += 1

### Create test set from original basis images
count_obj = 1
for oix,oo in enumerate(objs_b):
    count_im = 1
    for ii in range(ntest_cv):
        tx_ii = int(math.floor(numpy.random.rand() * max_trans))
        ty_ii = int(math.floor(numpy.random.rand() * max_trans))
        rot_ii = numpy.random.rand() * max_rot - (max_rot/2)
        scale_ii = min_scale + numpy.random.rand() * (1 - min_scale)

        obj_ii = oo.copy()
        obj_ii = obj_ii.convert('RGBA')
        ### Scale
        obj_ii = obj_ii.resize((int(math.ceil(size_imx * scale_ii)),
                                int(math.ceil(size_imy * scale_ii))),
                                resample=Image.LANCZOS)
        ### Rotate
        obj_ii = obj_ii.rotate(rot_ii, resample=Image.BICUBIC, expand=1)
        ### Make sure background is white
        fff = Image.new('RGBA', obj_ii.size, (255,)*4)
        obj_ii = Image.composite(obj_ii, fff, obj_ii)
        obj_ii = obj_ii.convert(oo.mode)
        newim = Image.new('RGB', (size_obx, size_oby), color='white')
        ### Paste into new image
        newim.paste(obj_ii, (tx_ii, ty_ii))
        ### Resize it to original size
        newim = newim.resize((fsize_imx, fsize_imy), resample=Image.LANCZOS)
        
        newim.save(BASE_PATH + 'test_cv/' + str(count_obj) + '/image_' + str(count_im) + '.png')
        count_im += 1

    count_obj += 1

### Create test set from original basis images
count_obj = 1
for oix,oo in enumerate(objs_b):
    count_im = 1
    for ii in range(ntest):
        tx_ii = int(math.floor(numpy.random.rand() * max_trans))
        ty_ii = int(math.floor(numpy.random.rand() * max_trans))
        rot_ii = numpy.random.rand() * max_rot - (max_rot/2)
        scale_ii = min_scale + numpy.random.rand() * (1 - min_scale)

        obj_ii = oo.copy()
        obj_ii = obj_ii.convert('RGBA')
        ### Scale
        obj_ii = obj_ii.resize((int(math.ceil(size_imx * scale_ii)),
                                int(math.ceil(size_imy * scale_ii))),
                                resample=Image.LANCZOS)
        ### Rotate
        obj_ii = obj_ii.rotate(rot_ii, resample=Image.BICUBIC, expand=1)
        ### Make sure background is white
        fff = Image.new('RGBA', obj_ii.size, (255,)*4)
        obj_ii = Image.composite(obj_ii, fff, obj_ii)
        obj_ii = obj_ii.convert(oo.mode)
        newim = Image.new('RGB', (size_obx, size_oby), color='white')
        ### Paste into new image
        newim.paste(obj_ii, (tx_ii, ty_ii))
        ### Resize it to original size
        newim = newim.resize((fsize_imx, fsize_imy), resample=Image.LANCZOS)
        
        newim.save(BASE_PATH + 'test_b/' + str(count_obj) + '/image_' + str(count_im) + '.png')
        count_im += 1

    count_obj += 1

### Create test set from V1 variants
count_obj = 1
for oix,oo in enumerate(objs_v1):
    count_im = 1
    for ii in range(ntest):
        tx_ii = int(math.floor(numpy.random.rand() * max_trans))
        ty_ii = int(math.floor(numpy.random.rand() * max_trans))
        rot_ii = numpy.random.rand() * max_rot - (max_rot/2)
        scale_ii = min_scale + numpy.random.rand() * (1 - min_scale)

        obj_ii = oo.copy()
        obj_ii = obj_ii.convert('RGBA')
        obj_ii = obj_ii.resize((int(math.ceil(size_imx * scale_ii)),
                                int(math.ceil(size_imy * scale_ii))),
                                resample=Image.LANCZOS)
        obj_ii = obj_ii.rotate(rot_ii, resample=Image.BICUBIC, expand=1)
        fff = Image.new('RGBA', obj_ii.size, (255,)*4)
        obj_ii = Image.composite(obj_ii, fff, obj_ii)
        obj_ii = obj_ii.convert(oo.mode)
        newim = Image.new('RGB', (size_obx, size_oby), color='white')
        newim.paste(obj_ii, (tx_ii, ty_ii))
        ### Resize it to original size
        newim = newim.resize((fsize_imx, fsize_imy), resample=Image.LANCZOS)
        
        newim.save(BASE_PATH + 'test_v1/' + str(count_obj) + '/image_' + str(count_im) + '.png')
        count_im += 1

    count_obj += 1

### Create test set from V2 variants
count_obj = 1
for oix,oo in enumerate(objs_v2):
    count_im = 1
    for ii in range(ntest):
        tx_ii = int(math.floor(numpy.random.rand() * max_trans))
        ty_ii = int(math.floor(numpy.random.rand() * max_trans))
        rot_ii = numpy.random.rand() * max_rot - (max_rot/2)
        scale_ii = min_scale + numpy.random.rand() * (1 - min_scale)

        obj_ii = oo.copy()
        obj_ii = obj_ii.convert('RGBA')
        obj_ii = obj_ii.resize((int(math.ceil(size_imx * scale_ii)),
                                int(math.ceil(size_imy * scale_ii))),
                                resample=Image.LANCZOS)
        obj_ii = obj_ii.rotate(rot_ii, resample=Image.BICUBIC, expand=1)
        fff = Image.new('RGBA', obj_ii.size, (255,)*4)
        obj_ii = Image.composite(obj_ii, fff, obj_ii)
        obj_ii = obj_ii.convert(oo.mode)
        newim = Image.new('RGB', (size_obx, size_oby), color='white')
        newim.paste(obj_ii, (tx_ii, ty_ii))
        ### Resize it to original size
        newim = newim.resize((fsize_imx, fsize_imy), resample=Image.LANCZOS)
        
        newim.save(BASE_PATH + 'test_v2/' + str(count_obj) + '/image_' + str(count_im) + '.png')
        count_im += 1

    count_obj += 1

### Create test set from V3 variants
count_obj = 1
for oix,oo in enumerate(objs_v3):
    count_im = 1
    for ii in range(ntest):
        tx_ii = int(math.floor(numpy.random.rand() * max_trans))
        ty_ii = int(math.floor(numpy.random.rand() * max_trans))
        rot_ii = numpy.random.rand() * max_rot - (max_rot/2)
        scale_ii = min_scale + numpy.random.rand() * (1 - min_scale)

        obj_ii = oo.copy()
        obj_ii = obj_ii.convert('RGBA')
        obj_ii = obj_ii.resize((int(math.ceil(size_imx * scale_ii)),
                                int(math.ceil(size_imy * scale_ii))),
                                resample=Image.LANCZOS)
        obj_ii = obj_ii.rotate(rot_ii, resample=Image.BICUBIC, expand=1)
        fff = Image.new('RGBA', obj_ii.size, (255,)*4)
        obj_ii = Image.composite(obj_ii, fff, obj_ii)
        obj_ii = obj_ii.convert(oo.mode)
        newim = Image.new('RGB', (size_obx, size_oby), color='white')
        newim.paste(obj_ii, (tx_ii, ty_ii))
        ### Resize it to original size
        newim = newim.resize((fsize_imx, fsize_imy), resample=Image.LANCZOS)
        
        newim.save(BASE_PATH + 'test_v3/' + str(count_obj) + '/image_' + str(count_im) + '.png')
        count_im += 1

    count_obj += 1
