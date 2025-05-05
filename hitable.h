#ifndef HITABLEH
#define HITABLEH

#include "ray.h"
#include "aabb.h"

class material;

struct hit_record // This structure holds the info about a ray-object intersection
{
	float t; // The distance from the ray origin to the intersection point
	vec3 p; // The intersection point
	vec3 normal; // The normal at the intersection point
	material* mat_ptr; // Pointer to the material of the object(intersected object)
};

class hitable { // Abstract base class for any object that can be "hit" by a ray
public:
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0; // This function checks if the ray intersects the object
	__device__ virtual bool bounding_box(aabb& box) const = 0;
	__device__ virtual ~hitable() {}
};

#endif