#ifndef BVHH
#define BVHH

#include "hitable.h"
#include "aabb.h"
#include <algorithm>
// This BVH feature has not been implemented yet.
/*
class bvh_node : public hitable {
public:
	hitable* left;
	hitable* right;
	aabb box;

	__device__ bvh_node(hitable** src_list, int n) {

		int axis = int(3 * rand() / (RAND_MAX + 1.0f));
		auto comparator = (axis == 0) ? box_x_compare :
			(axis == 1) ? box_y_compare
			: box_z_compare;

		if (n == 1) {
			left = right = src_list[0];
		}
		else if (n == 2) {
			if (comparator(src_list[0], src_list[1])) {
				left = src_list[0];
				right = src_list[1];
			}
			else {
				left = src_list[1];
				right = src_list[0];
			}
		}
		else {
			std::sort(src_list, src_list + n, comparator);
			int mid = n / 2;
			left = new bvh_node(src_list, mid);
			right = new bvh_node(src_list + mid, n - mid);
		}

		aabb box_left, box_right;
		bool ok1 = left->bounding_box(box_left);
		bool ok2 = right->bounding_box(box_right);
		if (!ok1 || !ok2) {
			std::cerr << "No bounding box in bvh_node constructor\n";
		}
		box = surrounding_box(box_left, box_right);
	}

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
		if (!box.hit(r, t_min, t_max)) {
			return false;
		}

		hit_record left_rec, right_rec;
		bool hit_left = left->hit(r, t_min, t_max, left_rec);
		bool hit_right = right->hit(r, t_min, t_max, right_rec);

		if (hit_right) {
			rec = right_rec;
			return true;
		}
		if (hit_left) {
			rec = left_rec;
			return true;
		}
		return false;

	}

	__host__ __device__ virtual bool bounding_box(aabb& b) const override {
		b = box;
		return true;
	}

private:
	static bool box_compare(hitable* a, hitable* b, int axis) {
		aabb box_a, box_b;
		a->bounding_box(box_a);
		b->bounding_box(box_b);
		return box_a._min[axis] < box_b._min[axis];
	}

	static bool box_x_compare(hitable* a, hitable* b) {
		return box_compare(a, b, 0);
	}
	static bool box_y_compare(hitable* a, hitable* b) {
		return box_compare(a, b, 1);
	}
	static bool box_z_compare(hitable* a, hitable* b) {
		return box_compare(a, b, 2);
	}
};
*/

#endif