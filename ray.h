#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray  // This class represents a ray in 3D space, it has an origin and a direction
{
public:
    __device__ ray() {}
	__device__ ray(const vec3& a, const vec3& b, float time = 0.0f) { // Constructor initializes the ray with an origin and a direction
        A = a; 
        B = b;
		tm = time;
    }
	__device__ vec3 origin() const { // This function returns the origin of the ray
        return A; 
    }
	__device__ vec3 direction() const { // This function returns the direction of the ray
        return B; 
    }
	__device__ vec3 point_at_parameter(float t) const { // This function returns a point along the ray at distance t from the origin
        return A + t * B; 
    } 
	__device__ float time() const { // This function returns the time of the ray
		return tm;
	}

    vec3 A; // Origin point of the ray
    vec3 B; // Direction vector of the ray
	float tm; // Time of the ray (for motion blur)
};

#endif