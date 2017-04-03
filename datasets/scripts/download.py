import sys
import os
sys.path.insert(0, '../neuroglancer/python')
sys.path.insert(0, '../neuroglancer/python/neuroglancer')
sys.path.insert(0, '../neuroglancer/python/neuroglancer/ingest')
sys.path.insert(0, '../neuroglancer/python/neuroglancer/ingest/volumes')

import numpy as np
from gcloudvolume import GCloudVolume

dataset_name = "golden_v0"
layer = "segmentation"
mip = 3

vol = GCloudVolume(dataset_name, layer, mip, cache_files=True, use_ls=False)

for z in xrange(0, 256, 64):
    for y in xrange(0, 256, 64):
        for x in xrange(0, 256, 64):
            segmentation = vol[x:x+64, y:y+64, z:z+64].astype(np.uint16)
            segmentation.resize((64, 64, 64))
            segmentationZYX = segmentation.swapaxes(0, 2)

            for z_sub in xrange(2):
                for y_sub in xrange(2):
                    for x_sub in xrange(2):
                        seg = segmentationZYX[z_sub * 32 : z_sub * 32 + 32, y_sub * 32 : y_sub * 32 + 32, x_sub * 32 : x_sub * 32 + 32,]
                        path = "./{}/{}/{}/z{}-{}/y{}-{}".format(dataset_name, layer, mip, z + z_sub * 32, z + z_sub * 32 + 31, y + y_sub * 32, y + y_sub * 32 + 31)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        seg.tofile("{}/{}-{}_{}-{}_{}-{}".format(path, x + x_sub * 32, x + x_sub * 32 + 31, y + y_sub * 32, y + y_sub * 32 + 31, z + z_sub * 32, z + z_sub * 32 + 31))


