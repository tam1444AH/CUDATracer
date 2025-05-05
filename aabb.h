#ifndef AABBH
#define AABBH

#include "ray.h"
#include "vec3.h"
#include <algorithm>  

struct aabb {
	vec3 _min;
	vec3 _max;

	__device__
		aabb() {}
	__device__
		aabb(const vec3& a, const vec3& b) {
		_min = a;
		_max = b;
	}

	__device__
	bool hit(const ray& r, float tmin, float tmax) const {

		for (int a = 0; a < 3; a++) {
			float invD = 1.0f / r.direction()[a];
			float t0 = (_min[a] - r.origin()[a]) * invD;
			float t1 = (_max[a] - r.origin()[a]) * invD;
			if (invD < 0.0f) {
				float temp = t0;
				t0 = t1;
				t1 = temp;
			}
			tmin = t0 > tmin ? t0 : tmin;
			tmax = t1 < tmax ? t1 : tmax;
			if (tmax <= tmin) {
				return false;
			}
		}
		return true;
	}
};

__device__
inline aabb surrounding_box(const aabb& box0, const aabb& box1) {
	vec3 small(fminf(box0._min.x(), box1._min.x()),
			   fminf(box0._min.y(), box1._min.y()),
			   fminf(box0._min.z(), box1._min.z()));
	vec3 big(fmaxf(box0._max.x(), box1._max.x()), 
		     fmaxf(box0._max.y(), box1._max.y()),
		     fmaxf(box0._max.z(), box1._max.z()));
	return aabb(small, big);
}

#endif