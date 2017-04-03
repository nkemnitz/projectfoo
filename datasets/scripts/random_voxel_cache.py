import numpy as np
import os
import re
import sys

DATASET_SIZE = [3072,3072,512]
MIP_COUNT = 4
DATASET = "golden_v0"

VOXEL_CACHE_OUTPUT = './Volume-cubey.512x512x512.uint16.voxelcache.32x32blk.raw'
PAGE_TABLE_OUTPUT = './Volume-cubey.512x512x512.uint16.pagetable.16x32blk.raw'
PAGE_DIRECTORY_OUTPUT = './Volume-cubey.512x512x512.uint16.pagedirectory.4x1.raw'

VOXEL_BLOCK_SIZE = 32
VOXEL_BLOCKS = 16
VOXEL_BLOCK_CACHE_SIZE = VOXEL_BLOCK_SIZE * VOXEL_BLOCKS # 512

PAGE_TABLE_BLOCK_SIZE = 32
PAGE_TABLES = 16
PAGE_TABLE_CACHE_SIZE = PAGE_TABLE_BLOCK_SIZE * PAGE_TABLES # 512

PAGE_DIRECTORY_SIZE = 2

EMPTY = 0
MAPPED = 1
NOT_MAPPED = 2

DATATYPE = np.uint16

def get_random_voxelblocks(mip, n):
    paths = []
    for root, dirnames, filenames in os.walk("{}/segmentation/{}".format(DATASET, mip)):
        for filename in filenames:
            paths.append(os.path.join(root, filename))

    selection = np.random.choice(paths, n, replace=False)
    return selection

def get_values_from_path(path):
    try:
        found = re.search("/(\d)/z(\d+)-(\d+)/y(\d+)-(\d+)/(\d+)-(\d+)_", path)
        return (int(found.group(1)), int(found.group(6)), int(found.group(7)), int(found.group(4)), int(found.group(5)), int(found.group(2)), int(found.group(3)))
    except AttributeError:
        return None

###############################################################################

page_directory = np.full([
        2*PAGE_DIRECTORY_SIZE, # for MIP (x1.5 would be enough, but power of two might be better)
        2*PAGE_DIRECTORY_SIZE,
        2*PAGE_DIRECTORY_SIZE,
        4
    ], NOT_MAPPED, dtype=np.uint8)

page_table = np.full([
        PAGE_TABLE_CACHE_SIZE, # depth / VOXEL_BLOCK_SIZE,
        PAGE_TABLE_CACHE_SIZE, # height / VOXEL_BLOCK_SIZE,
        PAGE_TABLE_CACHE_SIZE, # width / VOXEL_BLOCK_SIZE,
        4                      # last value is RGBA for x,y,z, (flag) coordinates
    ], NOT_MAPPED, dtype=np.uint8)


voxel_cache = np.full([
        VOXEL_BLOCK_CACHE_SIZE,
        VOXEL_BLOCK_CACHE_SIZE,
        VOXEL_BLOCK_CACHE_SIZE
    ], NOT_MAPPED, dtype=DATATYPE)


paths = [
    get_random_voxelblocks(0, 1792),
    get_random_voxelblocks(1, 1024),
    get_random_voxelblocks(2, 768),
    get_random_voxelblocks(3, 512)#512
]


active_page_table = [0,1,2,3]
active_page_table_entry_pos = [
    [0,0,0],
    [2*PAGE_TABLE_BLOCK_SIZE, 0, 0],
    [2*PAGE_TABLE_BLOCK_SIZE, 1*PAGE_TABLE_BLOCK_SIZE, 0],
    [2*PAGE_TABLE_BLOCK_SIZE, 2*PAGE_TABLE_BLOCK_SIZE, 0]

]
last_page_table = 3

page_directory_mip_entry = [
    [0,0,0], # 4,4,4
    [2,0,0], # 2,2,2
    [2,1,0], # 1,1,1
    [2,2,0], # 1,1,1
]

vc_x = 0
vc_y = 0
vc_z = 0


for level in xrange(len(paths)): #
    print "Level {}".format(level)
    sys.stdout.flush()
    for path in paths[level]:
        mip, x0, x1, y0, y1, z0, z1 = get_values_from_path(path)

        # write VOXEL_BLOCK_SIZE x VOXEL_BLOCK_SIZE x VOXEL_BLOCK_SIZE block to voxel cache
        data = np.fromfile(path, dtype=DATATYPE)
        voxel_cache[vc_z:vc_z+VOXEL_BLOCK_SIZE, vc_y:vc_y+VOXEL_BLOCK_SIZE, vc_x:vc_x+VOXEL_BLOCK_SIZE] = data.reshape(32,32,32)
        voxel_block_status = MAPPED
        if any(x in data for x in [4366, 9247]) is False:
            voxel_block_status = EMPTY

        # write voxel cache start point to page table block
        pt_pos = [active_page_table_entry_pos[mip][0] + x0 / (PAGE_TABLE_BLOCK_SIZE),
                  active_page_table_entry_pos[mip][1] + y0 / (PAGE_TABLE_BLOCK_SIZE),
                  active_page_table_entry_pos[mip][2] + z0 / (PAGE_TABLE_BLOCK_SIZE)]
        page_table[pt_pos[2], pt_pos[1], pt_pos[0]] = [vc_x / VOXEL_BLOCK_SIZE, vc_y / VOXEL_BLOCK_SIZE, vc_z / VOXEL_BLOCK_SIZE, voxel_block_status]
        page_table_status = MAPPED
        if np.all(page_table[pt_pos[2]:pt_pos[2]+32, pt_pos[1]:pt_pos[1]+32, pt_pos[0]:pt_pos[0]+32] == EMPTY):
            page_table_status = EMPTY

        # write page table block start point to page directory
        pd_pos = [page_directory_mip_entry[mip][0] + (x0 / (PAGE_TABLE_BLOCK_SIZE * VOXEL_BLOCK_SIZE)),
                  page_directory_mip_entry[mip][1] + (y0 / (PAGE_TABLE_BLOCK_SIZE * VOXEL_BLOCK_SIZE)),
                  page_directory_mip_entry[mip][2] + (z0 / (PAGE_TABLE_BLOCK_SIZE * VOXEL_BLOCK_SIZE))]
        page_directory[pd_pos[2], pd_pos[1], pd_pos[0]] = [pt_pos[0] / PAGE_TABLE_BLOCK_SIZE, pt_pos[1] / PAGE_TABLE_BLOCK_SIZE, pt_pos[2] / PAGE_TABLE_BLOCK_SIZE, page_table_status]

        # jump to next voxel cache pos
        vc_z += VOXEL_BLOCK_SIZE

        if vc_z >= VOXEL_BLOCKS * VOXEL_BLOCK_SIZE:
            vc_z = 0
            vc_y += VOXEL_BLOCK_SIZE

        if vc_y >= VOXEL_BLOCKS * VOXEL_BLOCK_SIZE:
            vc_y = 0
            vc_x += VOXEL_BLOCK_SIZE

print page_directory.shape
print page_table.shape
print voxel_cache.shape
page_directory.tofile("/usr/people/nkemnitz/src/projectfoo/datasets/pagedirectory.raw")
page_table.tofile("/usr/people/nkemnitz/src/projectfoo/datasets/pagetable.raw")
voxel_cache.tofile("/usr/people/nkemnitz/src/projectfoo/datasets/voxelcache.raw")
