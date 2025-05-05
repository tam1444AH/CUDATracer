#ifndef PLANEH
#define PLANEH

#include "hitable.h"
#include "material.h"   // for the material pointer
#include "vec3.h"
#include "ray.h"

class plane : public hitable {
public:

	__device__ plane(const vec3& normal_, float d_, material* m_) // Constructor for the plane class
        : normal(unit_vector(normal_)), d(d_), mat_ptr(m_) {
    }

    __device__ virtual ~plane() { delete mat_ptr; }

    // Ray‐plane intersection
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
        float denom = dot(r.direction(), normal);

        // if denom = 0, ray is parallel to plane
        if (fabsf(denom) < 1e-6f) {
            return false;
        }

        float t = (d - dot(normal, r.origin())) / denom;
		if (t < t_min || t > t_max) { // check if t is within the range
            return false;
        }

        rec.t = t; 
        rec.p = r.point_at_parameter(t);
        rec.normal = normal;
        rec.mat_ptr = mat_ptr;
        return true;
    }

	__device__ virtual bool bounding_box(aabb& box) const override {
        box = aabb(
            vec3(-100.0f, -0.01f, -100.0f),
            vec3(100.0f, +0.01f, 100.0f)
        );
		return true;
	}

	vec3      normal; // Normal vector of the plane
	float     d; // Distance from the origin to the plane along the normal
	material* mat_ptr; // Pointer to the material of the plane
};

#endif 