#version 460 core

#define MAX_AO_ITERATIONS 8

uniform vec3 startGridSize;
layout(std430, binding = 2) buffer octree
{
    uint octreeData[];
};
layout(std430, binding = 3) buffer octreeTricubic
{
    uint octreeTricubicData[];
};

uint roundFloat(float a)
{
    return (a >= 0.5) ? 1 : 0;
}

uniform float epsilon;

//Options 
uniform int maxIterations;
uniform int maxShadowIterations;

uniform float overRelaxation;
uniform bool useAO;

uniform bool useShadows;
uniform bool useSoftShadows;

//Lighting
uniform int lightNumber;
uniform vec3 lightPos[4];
uniform float lightIntensity[4];
uniform vec3 lightColor[4];
uniform float lightRadius[4];

//Material
uniform float matMetallic;
uniform float matRoughness;
uniform vec3 matAlbedo;
uniform vec3 matF0;

uniform float minBorderValue;
uniform float distanceScale;
uniform float time;

in vec3 gridPosition;
in vec3 gridNormal;
in vec3 cameraPos;

out vec4 fragColor;

const uint isLeafMask = 1 << 31;
const uint childrenIndexMask = ~(1 << 31);
const uint isMarkedMask = 1 << 30;

const float pos_infinity = uintBitsToFloat(0x7F800000);
const float neg_infinity = uintBitsToFloat(0xFF800000);

//Ray slab intersection
bool raySlabIntersection(vec3 bbmin, vec3 bbmax, vec3 o, vec3 d_inv, out float tmin, out float tmax) {
    
    tmin = neg_infinity;
    tmax = pos_infinity;

    //x
    float tx1 = (bbmin.x - o.x) * d_inv.x;
    float tx2 = (bbmax.x - o.x) * d_inv.x;

    tmin = max(tmin, min(tx1, tx2));
    tmax = min(tmax, max(tx1, tx2));

    //y
    float ty1 = (bbmin.y - o.y) * d_inv.y;
    float ty2 = (bbmax.y - o.y) * d_inv.y;

    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    //z
    float tz1 = (bbmin.z - o.z) * d_inv.z;
    float tz2 = (bbmax.z - o.z) * d_inv.z;

    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    return tmax >= tmin;
}

float rayLeafIntersection(vec3 bbmax, vec3 bbmin, vec3 o, vec3 d_inv) {
    float tx = (((d_inv.x > 0) ? bbmax.x : bbmin.x) - o.x) * d_inv.x;
    float ty = (((d_inv.y > 0) ? bbmax.y : bbmin.y) - o.y) * d_inv.y;
    float tz = (((d_inv.z > 0) ? bbmax.z : bbmin.z) - o.z) * d_inv.z;

    if (tx < 0) tx = 1e8;
    if (ty < 0) ty = 1e8;
    if (tz < 0) tz = 1e8;
    return min(tx, min(ty, tz));
}

// Light functions
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
	
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
	
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
	
    return ggx1 * ggx2;
}

//Access to the sdfOctree
bool isOutsideGrid(vec3 point) 
{
    vec3 fracPart = point * startGridSize;
    ivec3 arrayPos = ivec3(floor(fracPart));

    return arrayPos.x < 0 || arrayPos.y < 0 || arrayPos.z < 0 ||
       arrayPos.x >= startGridSize.x || arrayPos.y >= startGridSize.y || arrayPos.z >= startGridSize.z;
}

uint getLeaf(vec3 point, out vec3 leafPos, out float leafSize) 
{
    vec3 fracPart = point * startGridSize;
    ivec3 arrayPos = ivec3(floor(fracPart));
    fracPart = fract(fracPart);

    int index = arrayPos.z * int(startGridSize.y * startGridSize.x) +
                arrayPos.y * int(startGridSize.x) +
                arrayPos.x;
    uint currentNode = octreeData[index];

    leafSize = 1.0/startGridSize.x;
    leafPos = vec3(arrayPos)/startGridSize;
    while(!bool(currentNode & isLeafMask))
    {
        uvec3 childPos = uvec3(roundFloat(fracPart.x), roundFloat(fracPart.y), roundFloat(fracPart.z));
        uint childIdx = (childPos.z << 2) +
                        (childPos.y << 1) +
                         childPos.x;

        currentNode = octreeData[(currentNode & childrenIndexMask) + childIdx];
        fracPart = fract(2.0 * fracPart);
        leafSize *= 0.5;
        leafPos = leafPos + vec3(childPos) * leafSize;
    }
    return currentNode;
}

void getPolynomial(uint currentNode, out float array[8]) 
{
    uint vIndex = currentNode & childrenIndexMask;
    array[0] = uintBitsToFloat(octreeData[vIndex]);
    array[1] = uintBitsToFloat(octreeData[vIndex + 1]);
    array[2] = uintBitsToFloat(octreeData[vIndex + 2]);
    array[3] = uintBitsToFloat(octreeData[vIndex + 3]);
    array[4] = uintBitsToFloat(octreeData[vIndex + 4]);
    array[5] = uintBitsToFloat(octreeData[vIndex + 5]);
    array[6] = uintBitsToFloat(octreeData[vIndex + 6]);
    array[7] = uintBitsToFloat(octreeData[vIndex + 7]);
    return;
}

float getDistance(vec3 point)
{
    vec3 fracPart = point * startGridSize;
    ivec3 arrayPos = ivec3(floor(fracPart));

    if(arrayPos.x < 0 || arrayPos.y < 0 || arrayPos.z < 0 ||
       arrayPos.x >= startGridSize.x || arrayPos.y >= startGridSize.y || arrayPos.z >= startGridSize.z)
    {
            vec3 q = abs(point - vec3(0.5)) - 0.5;
            point = clamp(point, vec3(1e4, 1e4, 1e4), vec3(1.0 - 1e4, 1.0 - 1e4, 1.0 - 1e4));
            return length(max(q, vec3(0.0)))/distanceScale + minBorderValue;
    }

    fracPart = fract(fracPart);

    int index = arrayPos.z * int(startGridSize.y * startGridSize.x) +
                arrayPos.y * int(startGridSize.x) +
                arrayPos.x;
    uint currentNode = octreeData[index];

    while(!bool(currentNode & isLeafMask))
    {
        uint childIdx = (roundFloat(fracPart.z) << 2) + 
                        (roundFloat(fracPart.y) << 1) + 
                         roundFloat(fracPart.x);

        currentNode = octreeData[(currentNode & childrenIndexMask) + childIdx];
        fracPart = fract(2.0 * fracPart);
    }

    uint vIndex = currentNode & childrenIndexMask;

    float d00 = uintBitsToFloat(octreeData[vIndex]) * (1.0f - fracPart.x) +
                uintBitsToFloat(octreeData[vIndex + 1]) * fracPart.x;
    float d01 = uintBitsToFloat(octreeData[vIndex + 2]) * (1.0f - fracPart.x) +
                uintBitsToFloat(octreeData[vIndex + 3]) * fracPart.x;
    float d10 = uintBitsToFloat(octreeData[vIndex + 4]) * (1.0f - fracPart.x) +
                uintBitsToFloat(octreeData[vIndex + 5]) * fracPart.x;
    float d11 = uintBitsToFloat(octreeData[vIndex + 6]) * (1.0f - fracPart.x) +
                uintBitsToFloat(octreeData[vIndex + 7]) * fracPart.x;

    float d0 = d00 * (1.0f - fracPart.y) + d01 * fracPart.y;
    float d1 = d10 * (1.0f - fracPart.y) + d11 * fracPart.y;

    return d0 * (1.0f - fracPart.z) + d1 * fracPart.z;
}

float getDistanceTricubic(vec3 point) //tricubic interpolation
{
    vec3 fracPart = point * startGridSize;
    ivec3 arrayPos = ivec3(floor(fracPart));

    if(arrayPos.x < 0 || arrayPos.y < 0 || arrayPos.z < 0 ||
       arrayPos.x >= startGridSize.x || arrayPos.y >= startGridSize.y || arrayPos.z >= startGridSize.z)
    {
            vec3 q = abs(point - vec3(0.5)) - 0.5;
            //point = clamp(point, vec3(1e4, 1e4, 1e4), vec3(1.0 - 1e4, 1.0 - 1e4, 1.0 - 1e4));
            return length(max(q, vec3(0.0)))/distanceScale + minBorderValue;
            //return length(max(q, vec3(0.0)))/distanceScale + getDistance(clampCoord);
    }

    fracPart = fract(fracPart);

    int index = arrayPos.z * int(startGridSize.y * startGridSize.x) +
                arrayPos.y * int(startGridSize.x) +
                arrayPos.x;
    uint currentNode = octreeTricubicData[index];

    while(!bool(currentNode & isLeafMask))
    {
        uint childIdx = (roundFloat(fracPart.z) << 2) + 
                        (roundFloat(fracPart.y) << 1) + 
                         roundFloat(fracPart.x);

        currentNode = octreeTricubicData[(currentNode & childrenIndexMask) + childIdx];
        fracPart = fract(2.0 * fracPart);
    }

    uint vIndex = currentNode & childrenIndexMask;

    return 0.0
         + uintBitsToFloat(octreeTricubicData[vIndex + 0]) + uintBitsToFloat(octreeTricubicData[vIndex + 1]) * fracPart[0] + uintBitsToFloat(octreeTricubicData[vIndex + 2]) * fracPart[0] * fracPart[0] + uintBitsToFloat(octreeTricubicData[vIndex + 3]) * fracPart[0] * fracPart[0] * fracPart[0] + uintBitsToFloat(octreeTricubicData[vIndex + 4]) * fracPart[1] + uintBitsToFloat(octreeTricubicData[vIndex + 5]) * fracPart[0] * fracPart[1] + uintBitsToFloat(octreeTricubicData[vIndex + 6]) * fracPart[0] * fracPart[0] * fracPart[1] + uintBitsToFloat(octreeTricubicData[vIndex + 7]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[1] + uintBitsToFloat(octreeTricubicData[vIndex + 8]) * fracPart[1] * fracPart[1] + uintBitsToFloat(octreeTricubicData[vIndex + 9]) * fracPart[0] * fracPart[1] * fracPart[1] + uintBitsToFloat(octreeTricubicData[vIndex + 10]) * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] + uintBitsToFloat(octreeTricubicData[vIndex + 11]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] + uintBitsToFloat(octreeTricubicData[vIndex + 12]) * fracPart[1] * fracPart[1] * fracPart[1] + uintBitsToFloat(octreeTricubicData[vIndex + 13]) * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[1] + uintBitsToFloat(octreeTricubicData[vIndex + 14]) * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[1] + uintBitsToFloat(octreeTricubicData[vIndex + 15]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[1]
         + uintBitsToFloat(octreeTricubicData[vIndex + 16]) * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 17]) * fracPart[0] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 18]) * fracPart[0] * fracPart[0] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 19]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 20]) * fracPart[1] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 21]) * fracPart[0] * fracPart[1] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 22]) * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 23]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 24]) * fracPart[1] * fracPart[1] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 25]) * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 26]) * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 27]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 28]) * fracPart[1] * fracPart[1] * fracPart[1] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 29]) * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[1] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 30]) * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[1] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 31]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[1] * fracPart[2]
         + uintBitsToFloat(octreeTricubicData[vIndex + 32]) * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 33]) * fracPart[0] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 34]) * fracPart[0] * fracPart[0] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 35]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 36]) * fracPart[1] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 37]) * fracPart[0] * fracPart[1] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 38]) * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 39]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 40]) * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 41]) * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 42]) * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 43]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 44]) * fracPart[1] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 45]) * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 46]) * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 47]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2]
         + uintBitsToFloat(octreeTricubicData[vIndex + 48]) * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 49]) * fracPart[0] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 50]) * fracPart[0] * fracPart[0] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 51]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 52]) * fracPart[1] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 53]) * fracPart[0] * fracPart[1] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 54]) * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 55]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 56]) * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 57]) * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 58]) * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 59]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 60]) * fracPart[1] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 61]) * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 62]) * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] * fracPart[2] + uintBitsToFloat(octreeTricubicData[vIndex + 63]) * fracPart[0] * fracPart[0] * fracPart[0] * fracPart[1] * fracPart[1] * fracPart[1] * fracPart[2] * fracPart[2] * fracPart[2];
}

//SCENE
float map(vec3 pos)
{
    return distanceScale * getDistanceTricubic(pos);
}


//LIGHTING
float getAO(vec3 pos, vec3 n)
{
    float occ = 0.0;
    float decay = 1.0;
    for(int i=0; i < MAX_AO_ITERATIONS; i++)
    {
        float h = 0.002 + 0.1 * float(i)/8.0;
        float d = map(pos + n * h);
        occ += max(h-d, 0.0);
        decay *= 0.8;
    }

    return min(1.0 - 1.5 * occ, 1.0);
}

bool isInsideNode(vec3 p, vec3 cubeMin, vec3 cubeMax)
{
    bool insideX, insideY, insideZ;
    insideX = p.x >= cubeMin.x && p.x <= cubeMax.x;
    insideY = p.y >= cubeMin.y && p.y <= cubeMax.y;
    insideZ = p.z >= cubeMin.z && p.z <= cubeMax.z;
    return insideX && insideY && insideZ;
}

// The degree of the polynomials for which we compute roots
#define MAX_DEGREE 3
// When there are fewer intersections/roots than theoretically possible, some
// array entries are set to this value
#define NO_INTERSECTION 3.4e38


// Searches a single root of a polynomial within a given interval.
// \param out_root The location of the found root.
// \param out_end_value The value of the given polynomial at end.
// \param poly Coefficients of the polynomial for which a root should be found.
//        Coefficient poly[i] is multiplied by x^i.
// \param begin The beginning of an interval where the polynomial is monotonic.
// \param end The end of said interval.
// \param begin_value The value of the given polynomial at begin.
// \param error_tolerance The error tolerance for the returned root location.
//        Typically the error will be much lower but in theory it can be
//        bigger.
// \return true if a root was found, false if no root exists.
bool newton_bisection(out float out_root, out float out_end_value,
    float poly[MAX_DEGREE + 1], float begin, float end,
    float begin_value, float error_tolerance)
{
    if (begin == end) {
        out_end_value = begin_value;
        return false;
    }
    // Evaluate the polynomial at the end of the interval
    out_end_value = poly[MAX_DEGREE];

    for (int i = MAX_DEGREE - 1; i != -1; --i)
        out_end_value = out_end_value * end + poly[i];
    // If the values at both ends have the same non-zero sign, there is no root
    if (begin_value * out_end_value > 0.0)
        return false;
    // Otherwise, we find the root iteratively using Newton bisection (with
    // bounded iteration count)
    float current = 0.5 * (begin + end);

    for (int i = 0; i != 90; ++i) {
        // Evaluate the polynomial and its derivative
        float value = poly[MAX_DEGREE] * current + poly[MAX_DEGREE - 1];
        float derivative = poly[MAX_DEGREE];

        for (int j = MAX_DEGREE - 2; j != -1; --j) {
            derivative = derivative * current + value;
            value = value * current + poly[j];
        }
        // Shorten the interval
        bool right = begin_value * value > 0.0;
        begin = right ? current : begin;
        end = right ? end : current;
        // Apply Newton's method
        float guess = current - value / derivative;
        // Pick a guess
        float middle = 0.5 * (begin + end);
        float next = (guess >= begin && guess <= end) ? guess : middle;
        // Move along or terminate
        bool done = abs(next - current) < error_tolerance;
        current = next;
        if (done)
            break;
    }
    out_root = current;
    return true;
}


// Finds all roots of the given polynomial in the interval [begin, end] and
// writes them to out_roots. Some entries will be NO_INTERSECTION but other
// than that the array is sorted. The last entry is always NO_INTERSECTION.
void find_roots(out float out_roots[MAX_DEGREE + 1], float poly[MAX_DEGREE + 1], float begin, float end) {
    float tolerance = (end - begin) * epsilon;
    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.
    float derivative[MAX_DEGREE + 1];
    derivative[0] = poly[MAX_DEGREE - 2];
    derivative[1] = float(MAX_DEGREE - 1) * poly[MAX_DEGREE - 1];
    derivative[2] = (0.5 * float((MAX_DEGREE - 1) * MAX_DEGREE)) * poly[MAX_DEGREE - 0];

    for (int i = 3; i != MAX_DEGREE + 1; ++i)
        derivative[i] = 0.0;
    // Compute its two roots using the quadratic formula
    float discriminant = derivative[1] * derivative[1] - 4.0 * derivative[0] * derivative[2];
    if (discriminant >= 0.0) {
        float sqrt_discriminant = sqrt(discriminant);
        float scaled_root = derivative[1] + ((derivative[1] > 0.0) ? sqrt_discriminant : (-sqrt_discriminant));
        float root_0 = clamp(-2.0 * derivative[0] / scaled_root, begin, end);
        float root_1 = clamp(-0.5 * scaled_root / derivative[2], begin, end);
        out_roots[MAX_DEGREE - 2] = min(root_0, root_1);
        out_roots[MAX_DEGREE - 1] = max(root_0, root_1);
    }
    else {
        // Indicate that the cubic derivative has a single root
        out_roots[MAX_DEGREE - 2] = begin;
        out_roots[MAX_DEGREE - 1] = begin;
    }
    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched roots are set to begin
    out_roots[MAX_DEGREE] = end;

    for (int i = 0; i != MAX_DEGREE - 2; ++i)
        out_roots[i] = begin;
    // Work your way up to derivatives of higher degree until you reach the
    // polynomial itself. This implementation may seem peculiar: It always
    // treats the derivative as though it had degree MAX_DEGREE and it
    // constructs the derivatives in a contrived way. Changing that would
    // reduce the number of arithmetic instructions roughly by a factor of two.
    // However, it would also cause register spilling, which has a far more
    // negative impact on the overall run time. Profiling indicates that the
    // current implementation has no spilling whatsoever.

    for (int degree = 3; degree != MAX_DEGREE + 1; ++degree) {
        // Take the integral of the previous derivative (scaled such that the
        // constant coefficient can still be copied directly from poly)
        float prev_derivative_order = float(MAX_DEGREE + 1 - degree);

        for (int i = MAX_DEGREE; i != 0; --i)
            derivative[i] = derivative[i - 1] * (prev_derivative_order * (1.0 / float(i)));
        // Copy the constant coefficient without causing spilling. This part
        // would be harder if the derivative were not scaled the way it is.

        for (int i = 0; i != MAX_DEGREE - 2; ++i)
            derivative[0] = (degree == MAX_DEGREE - i) ? poly[i] : derivative[0];
        // Determine the value of this derivative at begin
        float begin_value = derivative[MAX_DEGREE];

        for (int i = MAX_DEGREE - 1; i != -1; --i)
            begin_value = begin_value * begin + derivative[i];
        // Iterate over the intervals where roots may be found

        for (int i = 0; i != MAX_DEGREE; ++i) {
            if (i < MAX_DEGREE - degree)
                continue;
            float current_begin = out_roots[i];
            float current_end = out_roots[i + 1];
            // Try to find a root
            float root;
            if (newton_bisection(root, begin_value, derivative, current_begin, current_end, begin_value, tolerance))
                out_roots[i] = root;
            else if (degree < MAX_DEGREE)
                // Create an empty interval for the next iteration
                out_roots[i] = out_roots[i - 1];
            else
                out_roots[i] = NO_INTERSECTION;
        }
    }
    // We no longer need this array entry
    out_roots[MAX_DEGREE] = NO_INTERSECTION;
}

bool solvePolynomialEquation(vec3 p, vec3 v, float[8] values, out float[4] results, out int resultNum, float t_in, float t_out)
{
    float a, b, c, d, e, f, g, h;
    a = -1.0 * values[0] + 1.0 * values[1] + 1.0 * values[2] - 1.0 * values[3] + 1.0 * values[4] - 1.0 * values[5] - 1.0 * values[6] + 1.0 * values[7];
    b = 1.0 * values[0] - 1.0 * values[1] - 1.0 * values[2] + 1.0 * values[3];
    c = 1.0 * values[0] - 1.0 * values[2] - 1.0 * values[4] + 1.0 * values[6];
    d = 1.0 * values[0] - 1.0 * values[1] - 1.0 * values[4] + 1.0 * values[5];
    e = -1.0 * values[0] + 1.0 * values[1];
    f = -1.0 * values[0] + 1.0 * values[2];
    g = -1.0 * values[0] + 1.0 * values[4];
    h = 1.0 * values[0];


    float c0, c1, c2, c3;

    c0 = p.x * p.y * p.z * a + p.x * p.y * b + p.x * p.z * d + p.x * e + p.y * p.z * c + p.y * f + p.z * g + h;
    c1 = p.x * p.y * a * v.z + p.x * p.z * a * v.y + p.x * b * v.y + p.x * d * v.z + p.y * p.z * a * v.x + p.y * b * v.x + p.y * c * v.z + p.z * c * v.y + p.z * d * v.x + e * v.x + f * v.y + g * v.z;
    c2 = p.x * a * v.y * v.z + p.y * a * v.x * v.z + p.z * a * v.x * v.y + b * v.x * v.y + c * v.y * v.z + d * v.x * v.z;
    c3 = a * v.x * v.y * v.z;

    float[4] eq;
    // eq[0] = c3;
    // eq[1] = c2;
    // eq[2] = c1;
    // eq[3] = c0;

    eq[3] = c3;
    eq[2] = c2;
    eq[1] = c1;
    eq[0] = c0;
    
    

    //solveCubic(eq, results, resultNum);
    find_roots(results, eq, t_in, t_out);
    for (int i = 0; i < 3; ++i)
    {
        if (results[i] != NO_INTERSECTION)
        {
            resultNum = 3;
            return true;
        }
    }
    return false;
}

bool raymarchV3(vec3 o, vec3 dir, out vec3 result)
{
    vec3 d_inv = vec3(1 / dir.x, 1 / dir.y, 1 / dir.z);

    float t_in;
    float t_out;
    bool intersect = raySlabIntersection(vec3(0), vec3(1), o, d_inv, t_in, t_out);
    if (t_out < 0) return false;
    //intersect bounding box first, if no intersection return false automatically
    if (!intersect) return false;

    float t_end = t_out;

    float t = max(0, t_in) + epsilon;

    while (t >= 0 && t <= t_end) {
        vec3 leafPos; 
        float leafSize;

        if (isOutsideGrid(o + dir * t)) return false;

        uint leafNode = getLeaf(o + dir * t, leafPos, leafSize);
        vec3 leafBBmax;
        vec3 leafBBmin;

        leafBBmax = leafPos + leafSize;
        leafBBmin = leafPos;

        float t_int = rayLeafIntersection(leafBBmax, leafBBmin, o + dir * t, d_inv);

        if (bool(leafNode & isMarkedMask))
        {
            float t_res;
            float[8] values;
            getPolynomial(leafNode, values);

            vec3 p = o + dir * t;
            vec3 v = dir;

            p = (p - leafPos) / leafSize;
            v = v / leafSize;

            float leaf_t_in, leaf_t_out;
            raySlabIntersection(vec3(0), vec3(1), p, 1/v, leaf_t_in, leaf_t_out); 
            
            if (t == epsilon) leaf_t_in = t;

            float[4] results;
            int resultNum;
            if (solvePolynomialEquation(p, v, values, results, resultNum, leaf_t_in, leaf_t_out))
            {
                for (int i = 0; i < resultNum; ++i) {
                    float t_res = float(results[i]);
                    float t_aux = t + t_res;
                    vec3 intersectionPoint = o + dir * t_aux;
                    if (isInsideNode(intersectionPoint, leafBBmin, leafBBmax)) {
                        t += t_res;
                        result = o + dir * t;
                        return true;
                    }
                }
            }
        }
        t += t_int;
        t += epsilon;
    }
    return false;
}

float hardshadow(in vec3 ro, in vec3 rd)
{
    vec3 result;
    if (raymarchV3(ro, rd, result)) return 0.0;
    return 1.0;
}

//Inigo Quilez improved soft shadow
float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float w )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<maxShadowIterations && t<maxt; i++ )
    {
        float h = distanceScale * getDistanceTricubic(ro + t*rd);
        res = min( res, h/(w*t) );
        t += clamp(h, 0.005, 0.50);
        if( res<-1.0 || t>maxt ) break;
    }
    res = max(res,-1.0);
    return 0.25*(1.0+res)*(1.0+res)*(2.0-res);
}

//hard shadows using v3
float shadow(in vec3 ro, in vec3 rd, float mint, float maxt, float w )
{
    return useSoftShadows ? softshadow(ro, rd, mint, maxt, w) : hardshadow(ro+mint*rd, rd);
}

vec3 mapColor(vec3 pos, vec3 cameraPos)
{
    //Normal
    vec3 N = normalize(gridNormal);
    //View vector
    vec3 V = normalize(cameraPos - pos);

    //Plane vs model
    vec3 aPos = pos + vec3(-0.5, -0.1, -0.5);
    float fd = max(length(aPos.xz) - 1.3, abs(aPos.y) - 0.07);


    // Fresnel parameter
    vec3 F0 = mix(matF0, matAlbedo, matMetallic);

    vec3 Lo = vec3(0.0);

    // Directional lights
    for (int i = 0; i < lightNumber; i++) 
    {
        float distToLight = length(lightPos[i] - pos);
        vec3 L = normalize(lightPos[i] - pos);
        vec3 H = normalize(V + L);

        vec3 sunColor = lightIntensity[i] * lightColor[i];

        float coneAngle = atan(lightRadius[i]/distToLight);
        float solidAngle = PI * sin(coneAngle) * pow((lightRadius[i]/distToLight), 2.0);
        float intensity = useShadows ? shadow(pos + epsilon * L, L, 0.005, distToLight, solidAngle) : 1.0f;
        vec3 radiance = sunColor * intensity;
        
        // Cook-torrance brdf
        float NDF = DistributionGGX(N, H, matRoughness);        
        float G = GeometrySmith(N, V, L, matRoughness);      
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);       
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - matMetallic;	  
        
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + epsilon;
        vec3 specular = numerator / denominator;  
            
        // Add to outgoing radiance Lo
        float NdotL = max(dot(N, L), 0.0);                
        Lo += (kD * matAlbedo / PI + specular) * radiance * NdotL;
    }

    vec3 ambient = useAO ? vec3(0.5) * matAlbedo * getAO(pos, N) : vec3(0.5) * matAlbedo; // Ambient light estimation
    vec3 color = ambient + Lo;

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));
   
    return color;
}

void main()
{
    vec3 outColor = vec3(0.9);
    
    vec3 hitPoint = gridPosition;
    outColor = mapColor(hitPoint, cameraPos);

    fragColor = vec4(outColor, 1.0);
}