#ifndef BOXH
#define BOXH	

#include "hitable.h"
#include "aabb.h"

class box : public hitable {
public:
	vec3 p0, p1;
	material* mat_ptr;

	__device__ box(const vec3& _p0, const vec3& _p1, material* m)
		: p0(_p0), p1(_p1), mat_ptr(m) {}

	__device__ virtual ~box() { delete mat_ptr; }

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
		float t0 = t_min;
		float t1 = t_max;

		vec3 invD = vec3(1.0f / r.direction().x(),
			1.0f / r.direction().y(),
			1.0f / r.direction().z());

		vec3 o = r.origin();

		// check x axis
		float tx0 = (p0.x() - o.x()) * invD.x();
		float tx1 = (p1.x() - o.x()) * invD.x();
		if (invD.x() < 0) {
			float temp = tx0;
			tx0 = tx1;
			tx1 = temp;
		}
		t0 = tx0 > t0 ? tx0 : t0;
		t1 = tx1 < t1 ? tx1 : t1;
		if (t1 <= t0) return false;

		// check y axis
		float ty0 = (p0.y() - o.y()) * invD.y();
		float ty1 = (p1.y() - o.y()) * invD.y();
		if (invD.y() < 0) {
			float temp = ty0;
			ty0 = ty1;
			ty1 = temp;
		}
		t0 = ty0 > t0 ? ty0 : t0;
		t1 = ty1 < t1 ? ty1 : t1;
		if (t1 <= t0) return false;

		// check z axis
		float tz0 = (p0.z() - o.z()) * invD.z();
		float tz1 = (p1.z() - o.z()) * invD.z();
		if (invD.z() < 0) {
			float temp = tz0;
			tz0 = tz1;
			tz1 = temp;
		}
		t0 = tz0 > t0 ? tz0 : t0;
		t1 = tz1 < t1 ? tz1 : t1;
		if (t1 <= t0) return false;

		rec.t = t0;
		rec.p = r.point_at_parameter(t0);

		vec3 n;
		if (rec.t == tx0) {
			n = vec3(-1, 0, 0);
		}
		else if (rec.t == tx1) {
			n = vec3(1, 0, 0);
		}
		else if (rec.t == ty0) {
			n = vec3(0, -1, 0);
		}
		else if (rec.t == ty1) {
			n = vec3(0, 1, 0);
		}
		else if (rec.t == tz0) {
			n = vec3(0, 0, -1);
		}
		else {
			n = vec3(0, 0, 1);
		}

		rec.normal = n;
		rec.mat_ptr = mat_ptr;
		return true;
	}

	__device__ virtual bool bounding_box(aabb& output_box) const override {
		output_box = aabb(p0, p1);
		return true;
	}


};


#endif 