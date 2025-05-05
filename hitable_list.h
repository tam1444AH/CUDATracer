#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

class hitable_list : public hitable { // This class is a simple aggregate of hitable pointers
public:
    __device__ hitable_list() {

    }
    __device__ hitable_list(hitable** l, int n) { // Overridden hit() method tests all objects in the list
        list = l; 
        list_size = n; 
    }
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

    __device__ virtual bool bounding_box(aabb& box) const override {
        if (list_size == 0) {
            return false;
        }
		aabb temp_box;
        bool first = list[0]->bounding_box(box);
		for (int i = 1; i < list_size; i++) {
			if (list[i]->bounding_box(temp_box)) {
				box = surrounding_box(box, temp_box);
			}
		}
        return first;
    }

    hitable** list;// Array of pointers to hittable objects
    int list_size; // Number of objects in the array
};

__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const { //This iterates over all objects, tracking the closest hit
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max; // Track closest intersection distance

    for (int i = 0; i < list_size; i++) { // Loop over all objects in the list
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
			rec = temp_rec; // Update the hit record with the closest hit
        }
    }
    return hit_anything; // Return true if any hit occurred
}

#endif