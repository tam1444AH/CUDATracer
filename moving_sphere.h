#ifndef MOVING_SPHEREH
#define MOVING_SPHEREH

#include "hitable.h"
#include "ray.h"
#include "vec3.h"
#include <math.h>
#include "material.h" // for the material pointer

class moving_sphere : public hitable {
public:
    vec3 center0, center1; // start/end positions
    float time0, time1;     // shutter interval
    float radius;
    material* mat_ptr;

    __device__ moving_sphere(const vec3& c0, const vec3& c1,
        float t0, float t1,
        float r, material* m)
        : center0(c0), center1(c1),
        time0(t0), time1(t1),
        radius(r), mat_ptr(m) {
    }
    __device__ virtual ~moving_sphere() { delete mat_ptr; }

    __device__ vec3 center(float tm) const {
        // linear interpolation
        return center0 + ((tm - time0) / (time1 - time0)) * (center1 - center0);
    }

    __device__ virtual bool hit(const ray& r,
        float t_min, float t_max,
        hit_record& rec) const override {

        // compute center at this ray’s time
        vec3 cen = center(r.time());
        vec3 oc = r.origin() - cen;

        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius * radius;
        float disc = b * b - a * c;

        if (disc > 0) {
            float sq = sqrtf(disc);
            float t1 = (-b - sq) / a;

            if (t1 < t_max && t1 > t_min) {
                rec.t = t1;
                rec.p = r.point_at_parameter(t1);
                rec.normal = (rec.p - cen) / radius;
                rec.mat_ptr = mat_ptr;
                return true;
            }

            float t2 = (-b + sq) / a;

            if (t2 < t_max && t2 > t_min) {
                rec.t = t2;
                rec.p = r.point_at_parameter(t2);
                rec.normal = (rec.p - cen) / radius;
                rec.mat_ptr = mat_ptr;
                return true;
            }
        }
        return false;
    }

	__device__ virtual bool bounding_box(aabb& box) const override {
		aabb box0(center0 - vec3(radius, radius, radius),
			center0 + vec3(radius, radius, radius));
		aabb box1(center1 - vec3(radius, radius, radius),
			center1 + vec3(radius, radius, radius));
		box = surrounding_box(box0, box1);
		return true;
	}
};

#endif // MOVING_SPHEREH
