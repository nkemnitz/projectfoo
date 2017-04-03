precision highp float;
precision highp int;
precision highp usampler2D;
precision highp usampler3D;

uniform usampler2D selectionTex;
uniform vec2 seeds; // TODO: should be uvec2
uniform vec3 sizes; // TODO: should be uvec3

uniform usampler3D cubeTex;
uniform float fovy;

uniform lowp usampler3D voxelCache;
uniform lowp usampler3D pageTable;
uniform lowp usampler3D pageDirectory;

in vec3 frontPos;
in vec3 rayDir;

//layout(location = 0) out vec4 glFragColor;
layout(location = 1) out uint glFragSegID;
layout(location = 2) out vec4 glFragDepth;

const vec3 ZERO3 = vec3(0.0);
const vec4 ZERO4 = vec4(0.0);
const vec3 ONE3 = vec3(1.0);
const vec4 ONE4 = vec4(1.0);
const vec4 WHITE = vec4(1.0);
const vec4 BLACK = vec4(0.0, 0.0, 0.0, 1.0);
const vec4 RED = vec4(1.0, 0.0, 0.0, 1.0);
const vec4 GREEN = vec4(0.0, 1.0, 0.0, 1.0);
const vec4 BLUE = vec4(0.0, 0.0, 1.0, 1.0);
const uint PAGE_DIRECTORY = uint(4);
const uint PAGE_TABLE = uint(8);
const uint VOXEL_BLOCK = uint(12);

const float SQRT2INV = 0.70710678118;
const float SQRT3INV = 0.57735026919;

const float STEPSIZE = 0.001;

// TODO: Move to uniforms
const bool DEBUG = true;
const float DATASET_SIZE = 2048.0;
const float BLOCK_SIZE_VOXEL = 32.0;
const float BLOCK_SIZE_PAGE = 32.0;
const float BLOCK_COUNT_VOXEL = 16.0;
const float BLOCK_COUNT_PAGE = 16.0;
const float TOTAL_SIZE_PAGE_DIRECTORY = 4.0;
const float TOTAL_SIZE_PAGE_CACHE = BLOCK_SIZE_PAGE * BLOCK_COUNT_PAGE;
const float TOTAL_SIZE_VOXEL_CACHE = BLOCK_SIZE_VOXEL * BLOCK_COUNT_VOXEL;

const vec3 MIP_OFFSET[4] = vec3[4](
  vec3(0.0, 0.0, 0.0),
  vec3(2.0, 0.0, 0.0),
  vec3(2.0, 1.0, 0.0),
  vec3(2.0, 2.0, 0.0)
);

const vec3 MIP_FACTOR[4] = vec3[4](
  vec3(1.0, 1.0, 1.0),
  vec3(0.5, 0.5, 1.0),
  vec3(0.25, 0.25, 1.0),
  vec3(0.125, 0.125, 1.0)
);

const uint EMPTY = uint(0);
const uint MAPPED = uint(1);
const uint NOT_MAPPED = uint(2);




const uint _murmur3_32_c1 = uint(0xcc9e2d51);
const uint _murmur3_32_c2 = uint(0x1b873593);
const uint _murmur3_32_n = uint(0xe6546b64);
const uint _murmur3_32_mix1 = uint(0x85ebca6b);
const uint _murmur3_32_mix2 = uint(0xc2b2ae35);

const uint _fnv_offset_32 = uint(0x811c9dc5);
const uint _fnv_prime_32 = uint(16777619);

ivec3 lastPositionPageDirectory[4] = ivec3[4](
  ivec3(-1, -1, -1),
  ivec3(-1, -1, -1),
  ivec3(-1, -1, -1),
  ivec3(-1, -1, -1)
);

uvec4 lastPositionPageDirectoryResult[4] = uvec4[4](
  uvec4(0,0,0,0),
  uvec4(0,0,0,0),
  uvec4(0,0,0,0),
  uvec4(0,0,0,0)
);

ivec3 lastPositionPageTable[4] = ivec3[4](
  ivec3(-1, -1, -1),
  ivec3(-1, -1, -1),
  ivec3(-1, -1, -1),
  ivec3(-1, -1, -1)
);

uvec4 lastPositionPageTableResult[4] = uvec4[4](
  uvec4(0,0,0,0),
  uvec4(0,0,0,0),
  uvec4(0,0,0,0),
  uvec4(0,0,0,0)
);

ivec3 lastPositionVoxelTable[4] = ivec3[4](
  ivec3(-1, -1, -1),
  ivec3(-1, -1, -1),
  ivec3(-1, -1, -1),
  ivec3(-1, -1, -1)
);

uvec4 lastPositionVoxelTableResult[4] = uvec4[4](
  uvec4(0,0,0,0),
  uvec4(0,0,0,0),
  uvec4(0,0,0,0),
  uvec4(0,0,0,0)
);


// --------------------- BEGIN Voxel Lookup ----------------------------

/*
 * Denormalize Texture Coordinates
 *
 * @param position Coordinates in TC: [0,1]
 *
 * @return coordinates in voxel space
 */
vec3 getVoxelCoordinates(vec3 position) {
  return position * DATASET_SIZE;
}

/*
 * Look up an entry from the Page Directory
 *
 * @param positionVoxel The actual raw position in the dataset we are looking at in voxel space
 * 
 * @return PageDirectoryEntry (RGBA8 uint) :
 *                        R - x coordinate in Page Table
 *                        G - y coordinate in Page Table
 *                        B - z coordinate in Page Table
 *                        A - Mapping flag indicates mapping status
 */
uvec4 getDirectoryEntry(vec3 positionVoxel, uint mip) {
  // Get bits representing page directory ( lop off page table and voxel table bits )
  vec3 positionPageDirectory = (positionVoxel * MIP_FACTOR[mip]) / (BLOCK_SIZE_PAGE * BLOCK_SIZE_VOXEL);

  ivec3 iPositionPageDirectory = ivec3(MIP_OFFSET[mip] + positionPageDirectory);
  if (iPositionPageDirectory == lastPositionPageDirectory[mip]) {
    return lastPositionPageDirectoryResult[mip];
  }

  lastPositionPageDirectory[mip] = iPositionPageDirectory;
  lastPositionPageDirectoryResult[mip] = texelFetch(pageDirectory, iPositionPageDirectory, 0);

  // Use bits to index into page directory
  return lastPositionPageDirectoryResult[mip];
}

/*
 * Look up an  from the Page Table
 *
 * @param offset The offset page table block we are looking at ( still needs conversion to texture space )
 * @param positionVoxel The actual raw position in the dataset we are looking at in voxel space
 *
 * @return PageTableEntry (RGBA8 uint):
 *                        R - x coordinate in Voxel Cache
 *                        G - y coordinate in Voxel Cache
 *                        B - z coordinate in Voxel Cache
 *                        A - Mapping flag indicates mapping status
 */
uvec4 getPageEntry(vec3 offset, vec3 positionVoxel, uint mip) {
  // Get bits representing page table ( between page directory and voxel table bits )
  vec3 positionPageTablePosition = mod((positionVoxel * MIP_FACTOR[mip]) / BLOCK_SIZE_VOXEL, BLOCK_SIZE_PAGE);
  vec3 positionPageTableOffset = offset * BLOCK_SIZE_PAGE;
  ivec3 iPositionPageTable = ivec3(positionPageTableOffset + positionPageTablePosition);

  if (iPositionPageTable == lastPositionPageTable[mip]) {
    return lastPositionPageTableResult[mip];
  }

  lastPositionPageTable[mip] = iPositionPageTable;
  lastPositionPageTableResult[mip] = texelFetch(pageTable, iPositionPageTable, 0);

  // use bits to index into page table
  return lastPositionPageTableResult[mip];
}

/*
 * Look up a voxel's id from the Voxel Cache
 *
 * @param offset The offset Voxel block we are looking at ( still needs conversion to texture space )
 * @param positionVoxel The actual raw position in the dataset we are looking at in voxel space
 *
 * @return VoxelBlockEntry (R16 uint): TODO allow 64 bit segmentids
 *                        R - SegmentId
 *                        G - 0
 *                        B - 0
 *                        A - 0
 */
uvec4 getVoxelEntry(vec3 offset, vec3 positionVoxel, uint mip) {
  // Get bits representing voxel table ( lop off everything before voxel table bits )
  vec3 positionVoxelTablePosition = mod(positionVoxel * MIP_FACTOR[mip], BLOCK_SIZE_VOXEL);
  vec3 positionVoxelTableOffset = offset * BLOCK_SIZE_VOXEL;
  ivec3 iPositionVoxelTable = ivec3(positionVoxelTableOffset + positionVoxelTablePosition);

  if (iPositionVoxelTable == lastPositionVoxelTable[mip]) {
    return lastPositionVoxelTableResult[mip];
  }

  lastPositionVoxelTable[mip] = iPositionVoxelTable;
  lastPositionVoxelTableResult[mip] = texelFetch(voxelCache, iPositionVoxelTable, 0);

  // use bits to index into voxel table
  return lastPositionVoxelTableResult[mip];
}

/*
 * Lookup the segment id for a position
 *
 * @param positionVoxel The position to look up in dataset voxel space
 *
 * @return SegmentID and Mapping State(uvec4): TODO allow 64 bit segmentids
 *                        R - SegmentID (32 upper bits)
 *                        G - SegmentID (32 lower bits)
 *                        B - Flag (EMPTY/MAPPED/NOT_MAPPED)
 *                        A - Level (PAGE_DIRECTORY/PAGE_TABLE/VOXEL_BLOCK)
 */
uvec4 getSegIDMapping(vec3 positionVoxel, uint mip) {
  uvec4 entryDirectory = getDirectoryEntry(positionVoxel, mip);
  if (entryDirectory.a != MAPPED) {
    return uvec4(0, 0, uint(entryDirectory.a), PAGE_DIRECTORY);
  }

  uvec4 entryTable = getPageEntry(vec3(entryDirectory.rgb), positionVoxel, mip);
  if (entryTable.a != MAPPED) {
    return uvec4(0, 0, uint(entryTable.a), PAGE_TABLE);
  }

  uvec4 entryVoxel = getVoxelEntry(vec3(entryTable.rgb), positionVoxel, mip);

  return uvec4(0, int(entryVoxel.r), MAPPED, VOXEL_BLOCK);//return uvec4(0, entryVoxel.r, MAPPED, VOXEL_BLOCK);*/
}

/*
 * Same as getSegIDMapping(), this returns only the segID portion
 *
 * @param position The actual raw positon in the dataset we are looking at in object space: (0,1)
 *
 * @return SegmentID (
 */
uint getSegIDMIP(vec3 position, uint mip) {
  vec3 positionVoxel = getVoxelCoordinates(position);
  return getSegIDMapping(positionVoxel, mip).g;
}

uint getSegID(vec3 position) {
  vec3 positionVoxel = getVoxelCoordinates(position);
  /*uvec4 seg = getSegIDMapping(positionVoxel, uint(0));
  if (seg.b == MAPPED)
    return seg.g;*/
  
  /*seg = getSegIDMapping(positionVoxel, uint(1));
  if (seg.b == MAPPED)
    return seg.g;

  seg = getSegIDMapping(positionVoxel, uint(2));
  if (seg.b == MAPPED)
    return seg.g;*/

  return getSegIDMapping(positionVoxel, uint(3)).g;
}

// --------------------- END Voxel Lookup ----------------------------



// --------------------- BEGIN Cuckoo Hashing ----------------------------

uint hashCombine(uint m, uint n) {
  return m ^ (n + uint(0x517cc1b7) + (n << 6) + (n >> 2));
}

uint rol(uint m, uint n) {
  return (m << n) | (m >> (uint(32) - n));
}

uint murmur3_32(uint key, uint seed) {
  uint k = key;
  k *= _murmur3_32_c1;
  k = rol(k, uint(15));
  k *= _murmur3_32_c2;

  uint h = seed;
  h ^= k;
  h = rol(h, uint(13));
  h = h * uint(5) + _murmur3_32_n;

  h ^= uint(4);
  h ^= h >> uint(16);
  h *= _murmur3_32_mix1;
  h ^= h >> uint(13);
  h *= _murmur3_32_mix2;
  h ^= h >> uint(16);

  return h;
}

uint fnv1a_32(uint key, uint seed) {
  uint h = seed;
  h ^= (key & uint(0xFF000000)) >> 24;
  h *= _fnv_prime_32;
  h ^= (key & uint(0x00FF0000)) >> 16;
  h *= _fnv_prime_32;
  h ^= (key & uint(0x0000FF00)) >> 8;
  h *= _fnv_prime_32;
  h ^= key & uint(0x000000FF);
  h *= _fnv_prime_32;

  return h;
}

bool isVisible(uint segID) {
  if (segID == uint(0)) {
    return false;
  }
  uint hi = uint(0);

  uint hashPos = hashCombine(segID, hi) % uint(sizes.z);
  uint x = hashPos % uint(sizes.x);
  uint y = hashPos / uint(sizes.x);

  uvec2 texel = texelFetch(selectionTex, ivec2(x, y), 0).rg;
  if (texel.r == segID && texel.g == hi) {
    return true;
  }

  hashPos = hashCombine(fnv1a_32(segID, uint(seeds.y)), fnv1a_32(hi, uint(seeds.y))) % uint(sizes.z);
  x = hashPos % uint(sizes.x);
  y = hashPos / uint(sizes.x);

  texel = texelFetch(selectionTex, ivec2(x, y), 0).rg;
  if (texel.r == segID && texel.g == hi) {
    return true;
  }
}

// --------------------- END Cuckoo Hashing ----------------------------

// good enough for now, don't use for serious stuff (beware lowp)
float rand(vec2 co){
  return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// true if v is inside the box, otherwise false
bool isInsideBox(vec3 v, vec3 blf, vec3 trb) {
  return dot(step(blf, v), step(v, trb)) > 2.99;
}

vec2 iRayBox(vec3 pos, vec3 dir, vec3 blf, vec3 trb) {
  if (any(equal(dir, ZERO3))) {
    dir += 0.000001;
    dir = normalize(dir);
  }

  vec3 tmin3 = (blf - pos) / dir;
  vec3 tmax3 = (trb - pos) / dir;

  vec3 tenter3 = min(tmin3, tmax3);
  vec3 texit3 = max(tmin3, tmax3);

  float tenter = max(tenter3.x, max(tenter3.y, tenter3.z));
  float texit = min(texit3.x, min(texit3.y, texit3.z));

  return vec2(tenter, texit);
}

vec2 iRayCurrentMIPVoxel(vec3 pos, vec3 dir, uint mip) {
  vec3 mipVoxelSize = 1.0 / (MIP_FACTOR[mip] * DATASET_SIZE);
  vec3 voxelStart = mipVoxelSize * floor(pos / mipVoxelSize);
  vec3 voxelEnd = voxelStart + mipVoxelSize;

  return iRayBox(pos, dir, voxelStart, voxelEnd);
}

vec2 iRayCurrentMIPBlock(vec3 pos, vec3 dir, uint mip, uint blocktype) {
  float blocksize = 1.0; // single voxel

  if (blocktype == PAGE_DIRECTORY) {
    blocksize = BLOCK_SIZE_PAGE * BLOCK_SIZE_VOXEL;
  }
  else if (blocktype == PAGE_TABLE) {
    blocksize = BLOCK_SIZE_VOXEL;
  }

  vec3 mipBlockSize = blocksize / (MIP_FACTOR[mip] * DATASET_SIZE);
  vec3 blockStart = mipBlockSize * floor(pos / mipBlockSize);
  vec3 blockEnd = blockStart + mipBlockSize;

  return iRayBox(pos, dir, blockStart, blockEnd);
}

vec4 segColor(uint segID) {
  float fsegID = float(segID);
  return vec4(rand(vec2(fsegID / 255.0, 0.123)),
        rand(vec2(fsegID / 255.0, 0.456)),
        rand(vec2(fsegID / 255.0, 0.789)),
        1.0);
}

vec4 segColorIfVisible(uint segID) {
  if (!isVisible(segID)) {
    return vec4(0.0);
  }
  return segColor(segID);
}

vec4 mixColorsIfVisible(vec4 col1, vec4 col2, float ratio) {
  if (col1.a == 0.0) {
    return col2;
  } else if (col2.a == 0.0) {
    return col1;
  } else {
    return mix(col1, col2, ratio);
  }
}

vec4 calcSmoothColor(vec3 texCoord, float stepsize) {
  vec3 ratio = fract(texCoord * 256.0 - 0.5);
  texCoord -= 0.5 * stepsize;

  vec4 colorCorners[8];
  vec4 colorMixed[6];

  colorCorners[0] = segColorIfVisible(getSegID(vec3(texCoord.x, texCoord.y, texCoord.z)));
  colorCorners[1] = segColorIfVisible(getSegID(vec3(texCoord.x, texCoord.y, texCoord.z + stepsize)));
  colorCorners[2] = segColorIfVisible(getSegID(vec3(texCoord.x + stepsize, texCoord.y, texCoord.z)));
  colorCorners[3] = segColorIfVisible(getSegID(vec3(texCoord.x + stepsize, texCoord.y, texCoord.z + stepsize)));
  colorCorners[4] = segColorIfVisible(getSegID(vec3(texCoord.x, texCoord.y + stepsize, texCoord.z)));
  colorCorners[5] = segColorIfVisible(getSegID(vec3(texCoord.x, texCoord.y + stepsize, texCoord.z + stepsize)));
  colorCorners[6] = segColorIfVisible(getSegID(vec3(texCoord.x + stepsize, texCoord.y + stepsize, texCoord.z)));
  colorCorners[7] = segColorIfVisible(getSegID(vec3(texCoord.x + stepsize, texCoord.y + stepsize,
    texCoord.z + stepsize)));

  colorMixed[0] = mixColorsIfVisible(colorCorners[0], colorCorners[1], ratio.z);
  colorMixed[1] = mixColorsIfVisible(colorCorners[2], colorCorners[3], ratio.z);
  colorMixed[2] = mixColorsIfVisible(colorCorners[4], colorCorners[5], ratio.z);
  colorMixed[3] = mixColorsIfVisible(colorCorners[6], colorCorners[7], ratio.z);

  colorMixed[4] = mixColorsIfVisible(colorMixed[0], colorMixed[1], ratio.x);
  colorMixed[5] = mixColorsIfVisible(colorMixed[2], colorMixed[3], ratio.x);

  return mixColorsIfVisible(colorMixed[4], colorMixed[5], ratio.y);
}

vec3 calcGradient(vec3 texCoord, float stepsize) {
  vec3 gradient = ZERO3;
  uint centerSegID = getSegID(texCoord);

  // 6 face adjacent voxels
  if (getSegID(texCoord + stepsize * vec3(-1.0,  0.0,  0.0)) != centerSegID)  { gradient += vec3(-1.0,  0.0,  0.0); }
  if (getSegID(texCoord + stepsize * vec3( 1.0,  0.0,  0.0)) != centerSegID)  { gradient += vec3( 1.0,  0.0,  0.0); }
  if (getSegID(texCoord + stepsize * vec3( 0.0, -1.0,  0.0)) != centerSegID)  { gradient += vec3( 0.0, -1.0,  0.0); }
  if (getSegID(texCoord + stepsize * vec3( 0.0,  1.0,  0.0)) != centerSegID)  { gradient += vec3( 0.0,  1.0,  0.0); }
  if (getSegID(texCoord + stepsize * vec3( 0.0,  0.0, -1.0)) != centerSegID)  { gradient += vec3( 0.0,  0.0, -1.0); }
  if (getSegID(texCoord + stepsize * vec3( 0.0,  0.0,  1.0)) != centerSegID)  { gradient += vec3( 0.0,  0.0,  1.0); }

  // 12 edge adjacent voxels
  if (getSegID(texCoord + stepsize * vec3(-1.0, -1.0,  0.0)) != centerSegID)
    { gradient += vec3(-SQRT2INV, -SQRT2INV,  0.0); }
  if (getSegID(texCoord + stepsize * vec3(-1.0,  1.0,  0.0)) != centerSegID)
    { gradient += vec3(-SQRT2INV,  SQRT2INV,  0.0); }
  if (getSegID(texCoord + stepsize * vec3( 1.0, -1.0,  0.0)) != centerSegID)
    { gradient += vec3( SQRT2INV, -SQRT2INV,  0.0); }
  if (getSegID(texCoord + stepsize * vec3( 1.0,  1.0,  0.0)) != centerSegID)
    { gradient += vec3( SQRT2INV,  SQRT2INV,  0.0); }

  if (getSegID(texCoord + stepsize * vec3(-1.0,  0.0, -1.0)) != centerSegID)
    { gradient += vec3(-SQRT2INV,  0.0, -SQRT2INV); }
  if (getSegID(texCoord + stepsize * vec3(-1.0,  0.0,  1.0)) != centerSegID)
    { gradient += vec3(-SQRT2INV,  0.0,  SQRT2INV); }
  if (getSegID(texCoord + stepsize * vec3( 1.0,  0.0, -1.0)) != centerSegID)
    { gradient += vec3( SQRT2INV,  0.0, -SQRT2INV); }
  if (getSegID(texCoord + stepsize * vec3( 1.0,  0.0,  1.0)) != centerSegID)
    { gradient += vec3( SQRT2INV,  0.0,  SQRT2INV); }

  if (getSegID(texCoord + stepsize * vec3( 0.0, -1.0, -1.0)) != centerSegID)
    { gradient += vec3( 0.0, -SQRT2INV, -SQRT2INV); }
  if (getSegID(texCoord + stepsize * vec3( 0.0, -1.0,  1.0)) != centerSegID)
    { gradient += vec3( 0.0, -SQRT2INV,  SQRT2INV); }
  if (getSegID(texCoord + stepsize * vec3( 0.0,  1.0, -1.0)) != centerSegID)
    { gradient += vec3( 0.0,  SQRT2INV, -SQRT2INV); }
  if (getSegID(texCoord + stepsize * vec3( 0.0,  1.0,  1.0)) != centerSegID)
    { gradient += vec3( 0.0,  SQRT2INV,  SQRT2INV); }

  // 8 corner adjacent voxels
  if (getSegID(texCoord + stepsize * vec3(-1.0, -1.0, -1.0)) != centerSegID)
    { gradient += vec3(-SQRT3INV, -SQRT3INV, -SQRT3INV); }
  if (getSegID(texCoord + stepsize * vec3(-1.0, -1.0,  1.0)) != centerSegID)
    { gradient += vec3(-SQRT3INV, -SQRT3INV,  SQRT3INV); }
  if (getSegID(texCoord + stepsize * vec3(-1.0,  1.0, -1.0)) != centerSegID)
    { gradient += vec3(-SQRT3INV,  SQRT3INV, -SQRT3INV); }
  if (getSegID(texCoord + stepsize * vec3(-1.0,  1.0,  1.0)) != centerSegID)
    { gradient += vec3(-SQRT3INV,  SQRT3INV,  SQRT3INV); }
  if (getSegID(texCoord + stepsize * vec3( 1.0, -1.0, -1.0)) != centerSegID)
    { gradient += vec3( SQRT3INV, -SQRT3INV, -SQRT3INV); }
  if (getSegID(texCoord + stepsize * vec3( 1.0, -1.0,  1.0)) != centerSegID)
    { gradient += vec3( SQRT3INV, -SQRT3INV,  SQRT3INV); }
  if (getSegID(texCoord + stepsize * vec3( 1.0,  1.0, -1.0)) != centerSegID)
    { gradient += vec3( SQRT3INV,  SQRT3INV, -SQRT3INV); }
  if (getSegID(texCoord + stepsize * vec3( 1.0,  1.0,  1.0)) != centerSegID)
    { gradient += vec3( SQRT3INV,  SQRT3INV,  SQRT3INV); }

  return gradient;
}

vec3 phongBlinn(vec3 normal, vec3 eyeDir, vec3 lightDir, vec3 ambientCol, vec3 diffuseCol, vec3 specularCol,
  float shininess) {
  // Ambient
  vec3 color = ambientCol;

  // Diffuse
  float lambert = max(0.0, dot(lightDir, normal));
  if (lambert > 0.0) {
    color += lambert * diffuseCol;

    // Specular (Blinn)
    vec3 halfDir = normalize(eyeDir + lightDir);
    float spec = pow(max(0.0, dot(halfDir, normal)), shininess);
    color += spec * specularCol;
  }

  return color;
}

vec3 gammaCorrect(vec3 linearColor, float gamma) {
  return pow(linearColor, vec3(1.0 / gamma));
}

float calcStepsize(float screenDist) {
  //return max(0.5 / 256.0, 0.5 / 256.0 * screenDist * tan(0.5*fovy));
  return max(0.5 / DATASET_SIZE, 0.5 / DATASET_SIZE * screenDist * tan(0.5*fovy));
}

bool inShadow(vec3 pos, vec3 lightDir) {
  float stepsize = max(5.0 * STEPSIZE, calcStepsize(distance(pos, frontPos)));

  vec3 shadowPos = pos + stepsize * lightDir;
  while (isInsideBox(shadowPos, ZERO3, ONE3) == true) {
    if (isVisible(getSegID(shadowPos)) == true) {
      return true;
    }
    shadowPos += stepsize * lightDir;
  }
  return false;
}

float occlusion(vec3 pos, float stepsize) {
  float occl = 0.0;

  if (isVisible(getSegID(pos + stepsize * vec3(-1.0,  0.0,  0.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 1.0,  0.0,  0.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 0.0, -1.0,  0.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 0.0,  1.0,  0.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 0.0,  0.0, -1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 0.0,  0.0,  1.0))))  { occl += 1.0; }

  // 12 edge adjacent voxels
  if (isVisible(getSegID(pos + stepsize * vec3(-1.0, -1.0,  0.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3(-1.0,  1.0,  0.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 1.0, -1.0,  0.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 1.0,  1.0,  0.0))))  { occl += 1.0; }

  if (isVisible(getSegID(pos + stepsize * vec3(-1.0,  0.0, -1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3(-1.0,  0.0,  1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 1.0,  0.0, -1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 1.0,  0.0,  1.0))))  { occl += 1.0; }

  if (isVisible(getSegID(pos + stepsize * vec3( 0.0, -1.0, -1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 0.0, -1.0,  1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 0.0,  1.0, -1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 0.0,  1.0,  1.0))))  { occl += 1.0; }

  // 8 corner adjacent voxels
  /*if (isVisible(getSegID(pos + stepsize * vec3(-1.0, -1.0, -1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3(-1.0, -1.0,  1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3(-1.0,  1.0, -1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3(-1.0,  1.0,  1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 1.0, -1.0, -1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 1.0, -1.0,  1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 1.0,  1.0, -1.0))))  { occl += 1.0; }
  if (isVisible(getSegID(pos + stepsize * vec3( 1.0,  1.0,  1.0))))  { occl += 1.0; }*/

  return 1.0 - 2.5 * max(0.0, occl / 18.0 - 0.60);
}

void main() {
  glFragColor = ZERO4;

  vec3 pos = frontPos;
  vec3 dir = normalize(rayDir);

  // Calculate multipliers 't' (pos + t * dir) for entry and exit points, stored in rayStartStop.x and rayStartStop.y
  vec2 rayStartStop = iRayBox(pos, dir, ZERO3, vec3(1.0, 1.0, 0.125));

  // Make sure we don't start behind the camera (negative t)
  rayStartStop.x = max(0.0, rayStartStop.x) + 0.0001;
  pos += rayStartStop.x * dir;

  // Terminate if ray never entered the box
  if (isInsideBox(pos, vec3(0.0), vec3(1.0)) == false) {
    return;
  }

  uint segID = uint(0);
  uint visibleSegID = uint(0);
  vec3 visiblePos = frontPos;

  // TODO: Set better light(s)
  vec3 lightDir = normalize(-dir + cross(dir, vec3(0.0, 1.0, 0.0)));
  vec4 color = ZERO4;

  float steps = 0.0;
  vec3 same = ZERO3;

  while (true) {
    steps += 1.0;

    // Break Condition: Left the dataset bounding box
    if (isInsideBox(pos, vec3(0.0), vec3(1.0)) == false) {
      break;
    }

    // Hack for golden v0
    if (pos.z > 0.125) { // only 256 pixel deep
      break;
    }


    float curStepsize = calcStepsize(distance(pos, frontPos));

    vec3 positionVoxel = getVoxelCoordinates(pos);

    ivec3 oldPDPos = lastPositionVoxelTable[3];
    uvec4 segIDMapping3 = getSegIDMapping(positionVoxel.xyz, uint(3));
    if (oldPDPos == lastPositionVoxelTable[3]) {
      same.g += 1.0;
    } else {
      same.r += 1.0;
    }

    if (segIDMapping3.b == EMPTY) {
      vec2 blockStartStop = iRayCurrentMIPBlock(pos, dir, uint(3), segIDMapping3.a);
      pos += max(curStepsize, blockStartStop.y) * dir;
      continue;
    }

    uvec4 segIDMapping2 = getSegIDMapping(positionVoxel.xyz, uint(2));
    if (segIDMapping2.b == EMPTY) {
      vec2 blockStartStop = iRayCurrentMIPBlock(pos, dir, uint(2), segIDMapping2.a);
      pos += max(curStepsize, blockStartStop.y) * dir;
      continue;
    }

    uvec4 segIDMapping1 = getSegIDMapping(positionVoxel.xyz, uint(1));
    if (segIDMapping1.b == EMPTY) {
      vec2 blockStartStop = iRayCurrentMIPBlock(pos, dir, uint(1), segIDMapping1.a);
      pos += max(curStepsize, blockStartStop.y) * dir;
      continue;
    }

    uvec4 segIDMapping0 = getSegIDMapping(positionVoxel.xyz, uint(0));
    if (segIDMapping0.b == EMPTY && segIDMapping0.a < VOXEL_BLOCK) {
      vec2 blockStartStop = iRayCurrentMIPBlock(pos, dir, uint(0), segIDMapping0.a);
      pos += max(curStepsize, blockStartStop.y) * dir;
      continue;
    }

    uint mip = uint(0);
    uvec4 segIDMapping = segIDMapping0;
    if (segIDMapping.b != MAPPED) {
      mip = uint(1);
      segIDMapping = segIDMapping1;
    }
    if (segIDMapping.b != MAPPED) {
      mip = uint(2);
      segIDMapping = segIDMapping2;
    }
    if (segIDMapping.b != MAPPED) {
      mip = uint(3);
      segIDMapping = segIDMapping3;
    }

    // Voxel block is not mapped - skip whole block
    // TODO: Dangerous, If the whole MIP3 PageTable is not mapped, we might skip over a higher-res, mapped PageTable!
    if (segIDMapping.b != MAPPED) {
      vec2 blockStartStop = iRayCurrentMIPBlock(pos, dir, uint(3), segIDMapping.a);
      pos += max(curStepsize, blockStartStop.y) * dir;
      continue;
    } else {
      segID = segIDMapping.g;

      if (isVisible(segID)) {
        vec3 normal = normalize(calcGradient(pos, 1.0 / 2048.0));
        vec3 camDir = -dir;
        //vec3 basecol = calcSmoothColor(pos, 1.0 / 256.0).rgb;
        vec3 basecol = segColor(segID).rgb;
        color.rgb = 0.07 * basecol * 1.0; //occlusion(pos, 2.0 * curStepsize);
        //if (!inShadow(pos, lightDir)) {
          color.rgb = phongBlinn(normal, camDir, lightDir, 0.07 * basecol, 1.0*basecol, vec3(0.5), 16.0);
        //}
        color.rgb = gammaCorrect(color.rgb, 2.2);
        color.a = 1.0;

        visibleSegID = segID;
        visiblePos = pos;

        // Break Condition: Hit a visible segment
        break;
      }
    }

    // Voxel Block was mapped, but voxel itself is empty. Skip large voxels (mip 2 or larger)
    float t = 0.0;
    if (mip >= uint(1)) {
      t = iRayCurrentMIPBlock(pos, dir, mip, VOXEL_BLOCK).y;
    }
    pos += max(curStepsize, t) * dir;

    
  }
  //color = vec4(same.r/1000.0, same.g/1000.0, 0.0, 1.0);
  //color.a = 1.0;
  glFragColor = color;
  glFragSegID = visibleSegID;
  glFragDepth = vec4(distance(visiblePos, frontPos));

}
