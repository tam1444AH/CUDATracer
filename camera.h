#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// We genrate a random point inside a unit disk for depth of field/thin lens effects
__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class camera { // This class represents a camera in 3D space, it also produces rays for the scene
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist,
        float _t0, float _t1) : time0(_t0), time1(_t1) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float theta = vfov * ((float)M_PI) / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        // Horizontal and vertical spans of the view plane
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }
    // Generate a ray passing through normalized screen coords (s,t)
    // s = [0,1] horizontally, t = [0,1] vertically
    // local_rand_state: RNG for lens sampling
    __device__ ray get_ray(float s, float t, curandState* local_rand_state) {
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
		float time = time0 + curand_uniform(local_rand_state) * (time1 - time0);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset, time);
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w; // camera coordinate frame
    float lens_radius; // controls aperture size
	float time0, time1; // for motion blur
};

#endif