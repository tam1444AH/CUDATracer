#ifndef TRIANGLEH
#define TRIANGLEH
#include "hitable.h"
#include "aabb.h"

struct triangle : public hitable {
	vec3 v0, v1, v2;
	material* mat_ptr;
	__device__ triangle(const vec3& a, const vec3& b, const vec3& c, material* m)
		: v0(a), v1(b), v2(c), mat_ptr(m) { }
	__device__ virtual ~triangle() { delete mat_ptr; }

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
		const float EPS = 1e-6f;
		vec3 edge1 = v1 - v0;
		vec3 edge2 = v2 - v0;
		vec3 pvec = cross(r.direction(), edge2);
		float det = dot(edge1, pvec);

		if (fabs(det) < EPS) return false; // This ray is parallel to this triangle

		float inv_det = 1.0f / det;
		vec3 tvec = r.origin() - v0;
		float u = dot(tvec, pvec) * inv_det;

		if (u < 0.0f || u > 1.0f) return false;

		vec3 qvec = cross(tvec, edge1);
		float v = dot(r.direction(), qvec) * inv_det;
		if (v < 0.0f || u + v > 1.0f) return false;
		float t = dot(edge2, qvec) * inv_det;
		if (t < t_min || t > t_max) return false;
		rec.t = t;
		rec.p = r.point_at_parameter(t);
		rec.normal = unit_vector(cross(edge1, edge2));
		rec.mat_ptr = mat_ptr;
		return true;

	}

	__device__ virtual bool bounding_box(aabb& box) const override {
		//vec3 mn = min(v0, min(v1, v2));
		//vec3 mx = max(v0, max(v1, v2));
		vec3 mn = vec3(fminf(v0.x(), fminf(v1.x(), v2.x())),
			fminf(v0.y(), fminf(v1.y(), v2.y())),
			fminf(v0.z(), fminf(v1.z(), v2.z())));		
		vec3 mx = vec3(fmaxf(v0.x(), fmaxf(v1.x(), v2.x())),
			fmaxf(v0.y(), fmaxf(v1.y(), v2.y())),
			fmaxf(v0.z(), fmaxf(v1.z(), v2.z())));
		box = aabb(mn, mx);
		return true;
	}
};
#endif