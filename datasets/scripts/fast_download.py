import sys
sys.path.insert(0, '../neuroglancer/python')
sys.path.insert(0, '../neuroglancer/python/neuroglancer')
sys.path.insert(0, '../neuroglancer/python/neuroglancer/ingest')
sys.path.insert(0, '../neuroglancer/python/neuroglancer/ingest/volumes')

from gcloudvolume import GCloudVolume


vol = GCloudVolume("golden_v0", "segmentation", 3, cache_files=True, use_ls=True)
segmentation = vol[0:256, 0:256, 0:256]

print(vol.shape)
