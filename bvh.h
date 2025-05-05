#ifndef BVHH
#define BVHH

#include "hitable.h"
#include "aabb.h"

// A very basic device-side BVH: always split along X, O(n²) build via selection-sort.
class bvh_node : public hitable {
public:
    hitable* left;
    hitable* right;
    aabb box;

    // Build a BVH over src_list[0..n)
    __device__ bvh_node(hitable** src_list, int n) {
        auto comparator = box_x_compare;

        if (n == 1) {
            left = right = src_list[0];
        }
        else if (n == 2) {
            if (comparator(src_list[0], src_list[1])) {
                left = src_list[0];  right = src_list[1];
            }
            else {
                left = src_list[1];  right = src_list[0];
            }
        }
        else {
            // selection-sort by X
            for (int i = 0; i < n - 1; i++) {
                int minIdx = i;
                for (int j = i + 1; j < n; j++) {
                    if (comparator(src_list[j], src_list[minIdx]))
                        minIdx = j;
                }
                if (minIdx != i) {
                    hitable* tmp = src_list[i];
                    src_list[i] = src_list[minIdx];
                    src_list[minIdx] = tmp;
                }
            }
            int mid = n / 2;
            left = new bvh_node(src_list, mid);
            right = new bvh_node(src_list + mid, n - mid);
        }

        // compute this node’s box
        aabb box_left, box_right;
        if (!left->bounding_box(box_left) || !right->bounding_box(box_right)) {
            // you might printf here, but device-print is a pain
        }
        box = surrounding_box(box_left, box_right);
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
        if (!box.hit(r, t_min, t_max)) return false;
        hit_record left_rec, right_rec;
        bool hit_left = left->hit(r, t_min, t_max, left_rec);
        bool hit_right = right->hit(r, t_min, t_max, right_rec);
        if (hit_right) { rec = right_rec; return true; }
        if (hit_left) { rec = left_rec;  return true; }
        return false;
    }

    __device__ virtual bool bounding_box(aabb& b) const override {
        b = box;
        return true;
    }

private:
    // compare the minimum corner along axis
    static __device__ bool box_compare(hitable* a, hitable* b, int axis) {
        aabb A, B;
        a->bounding_box(A);
        b->bounding_box(B);
        return A._min[axis] < B._min[axis];
    }
    static __device__ bool box_x_compare(hitable* a, hitable* b) { return box_compare(a, b, 0); }
    static __device__ bool box_y_compare(hitable* a, hitable* b) { return box_compare(a, b, 1); }
    static __device__ bool box_z_compare(hitable* a, hitable* b) { return box_compare(a, b, 2); }
};

#endif
