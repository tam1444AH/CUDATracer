#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include "material.h" // for the material pointer

class sphere : public hitable { // This class represents a sphere in 3D space, its a hitable object
public:
    __device__ sphere() {
    }
    __device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m) {};

    __device__ virtual ~sphere() { delete mat_ptr; }

    __device__ virtual bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const override { // This function checks if the ray intersects the sphere, between t_min and t_max
        vec3 oc = r.origin() - center; // Vector from ray origin to sphere center

        float a = dot(r.direction(), r.direction()); // Quadratic coefficients for intersection test:
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;

        if (discriminant > 0) { // If discriminant > 0, ray intersects sphere
            float temp = (-b - sqrt(discriminant)) / a;

            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.point_at_parameter(rec.t);
                rec.normal = (rec.p - center) / radius;
                rec.mat_ptr = mat_ptr;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.point_at_parameter(rec.t);
                rec.normal = (rec.p - center) / radius;
                rec.mat_ptr = mat_ptr;
                return true;
            }
        }
        return false; // No valid intersection found
    }

    __device__ virtual bool sphere::bounding_box(aabb& box) const override {
        box = aabb(center - vec3(radius, radius, radius),
            center + vec3(radius, radius, radius));
        return true;
    }

    vec3 center;
    float radius;
	material* mat_ptr; // Pointer to the material of the sphere
};


#endif