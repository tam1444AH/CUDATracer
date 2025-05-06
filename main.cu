#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "plane.h"
#include "moving_sphere.h"
#include "bvh.h"
#include "box.h"
#include "triangle.h"

#define MEDIUM_DENSITY 0.1f
#define FOG_COLOR vec3(0.8f, 0.8f, 0.9f)  // a pale bluish fog
#define RND (curand_uniform(&local_rand_state))

//#define DEBUG_NORMAL
//#define DEBUG_DEPTH

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// In this function we compute the color seen along a ray by bouncing/scattering. We unroll recursion into a loop of max-depth 50 to avoid GPU stack overflow
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state) {

    ray currentRay = r;
    vec3 curAttenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) { // max depth of 50
        hit_record rec;
        bool hit_surf = (*world)->hit(currentRay, 0.001f, FLT_MAX, rec);

		if (!hit_surf) { // if the ray does not hit anything, we return the background color
			vec3 unitDirection = unit_vector(currentRay.direction());
			float t = 0.5f * (unitDirection.y() + 1.0f);
			vec3 background = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return curAttenuation * background;
		}



		float rnd = curand_uniform(local_rand_state); // Enable this code to simulate fog
        float dist_to_scatter = -logf(rnd) / MEDIUM_DENSITY;

        if (rec.t < dist_to_scatter) {
            ray scattered;
            vec3 attenuation;

            if (rec.mat_ptr->scatter(currentRay, rec, attenuation, scattered, local_rand_state)) {
                curAttenuation *= attenuation;
                currentRay = scattered;
                continue;
            }
            else {
                return vec3(0, 0, 0);
            }
        }

        vec3 p = currentRay.point_at_parameter(dist_to_scatter);
        curAttenuation *= FOG_COLOR;

        vec3 new_dir = random_in_unit_sphere(local_rand_state);
        currentRay = ray(p, new_dir, currentRay.time());



		//ray scattered; // without fog
  //      vec3 attenuation;
  //      if (rec.mat_ptr->scatter(currentRay, rec, attenuation, scattered, local_rand_state)) {
  //          curAttenuation *= attenuation;
  //          currentRay = scattered;
  //          continue;
  //      }
  //      else {
  //          return vec3(0, 0, 0);
  //      }

    }
    return vec3(0.0, 0.0, 0.0); // we exceed the max depth, return black
}

__device__ vec3 debug_color(const ray& r, hitable** world) {
	hit_record record;
    if ((*world)->hit(r, 0.001f, FLT_MAX, record)) { 
#ifdef DEBUG_NORMAL
		vec3 normal = record.normal;
		return 0.5f * (normal + vec3(1.0f, 1.0f, 1.0f));
#elif defined(DEBUG_DEPTH)
		const float FAR = 100.0f;
		float g = 1.0f - fminf(record.t / FAR, 1.0f);
		return vec3(g, g, g);
#endif
    }
	return vec3(0.0, 0.0, 0.0); 
}

// This is a one-thread kernel to initialize the random state for world creation
__global__ void rand_init(curandState* randState) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, randState);
    }
}

// This function initializes the random state for each pixel
__global__ void render_init(int maxX, int maxY, curandState* randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= maxX) || (j >= maxY)) return; // out of bounds

    int pixelIndex = j * maxX + i;

    // We use unique seeds for each thread
    curand_init(1984 + pixelIndex, 0, 0, &randState[pixelIndex]);
}

// In this function, each thread computes the color of one pixel
__global__ void render(vec3* fb, int maxX, int maxY, int ns, camera** cam, hitable** world, curandState* rand_state) {

    int i = threadIdx.x + blockIdx.x * blockDim.x; // Pixel coordinates
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= maxX) || (j >= maxY)) return; // out of bounds

    int pixelIndex = j * maxX + i;

    curandState local_rand_state = rand_state[pixelIndex]; // We load the random state for this pixel
    vec3 col(0, 0, 0);

    for (int s = 0; s < ns; s++) { // We sample the pixel ns times
        float u = float(i + curand_uniform(&local_rand_state)) / float(maxX); // jittered sample within the pixel
        float v = float(j + curand_uniform(&local_rand_state)) / float(maxY);
        ray r = (*cam)->get_ray(u, v, &local_rand_state); // We get the ray from the camera
        col += color(r, world, &local_rand_state); // We compute the color along the ray
    }
    rand_state[pixelIndex] = local_rand_state; // We save the random state for this pixel

    // We average the color and apply gamma correction
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);

    fb[pixelIndex] = col; // We store the color in the framebuffer

	// We can also debug the normal or depth
 //   float u = (i + 0.5f) / float(maxX);
	//float v = (j + 0.5f) / float(maxY);
	//ray r = (*cam)->get_ray(u, v, &rand_state[pixelIndex]); // We get the ray from the camera
 //   fb[pixelIndex] = debug_color(r, world);
}

// This function (one-thread) creates the world and camera on the GPU
__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new plane(
			vec3(0, 1, 0), // normal
			0.0f, // d
			new checker(
				vec3(0.1f, 0.1f, 0.1f), // dark green
				vec3(0.9f, 0.9f, 0.9f), // almost white
				1.0f                   // one square per world‐unit
			)
		);
        int i = 1;
        for (int a = -11; a < 11; a++) { // We create random small spheres on the ground
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) { // Diffuse material
                    vec3 start = center;
                    vec3 end = center + vec3(sinf((float)i), 0, 0); // Small 'jitter' in x for motion blur effect;
                    d_list[i++] = new moving_sphere(
                        start, end,
                        0.0f, 1.0f,
                        0.2f,
                        new lambertian(vec3(RND * RND, RND * RND, RND * RND))
                    );
                }
                else if (choose_mat < 0.95f) { // Metal material
                    d_list[i++] = new sphere(center, 0.2,
                        new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else { // Glass material
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        // We then create three more larger spheres
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

        d_list[i++] = new box(
            vec3(-5, 0, -5),    // corner 1
            vec3(-3, 2, -3),    // corner 2
            new lambertian(vec3(0.8, 0.3, 0.3))
        );
        d_list[i++] = new box(
            vec3(-15, 0, -10),
            vec3(-10, 5, -5),
            new metal(vec3(0.8, 0.8, 0.9), 0.1)
        );

        for (int b = 0; b < 30; b++) {
            //float w = 1.0f + 2.0f * RND;
            //float h = 1.0f + 2.0f * RND;
            //float d = 1.0f + 2.0f * RND;
			float w = 0.3f + 0.7f * RND;
			float h = 0.3f + 0.7f * RND;
			float d = 0.3f + 0.7f * RND;

            float cx = -30.0f + 22.0f * RND;
            float cz = -10.0f + 15.0f * RND;

            //float cx = -11.0f + 22.0f * RND;
            //float cz = -25.0f + 10.0f * RND;

            vec3 p0 = vec3(cx - (w / 2), 0.0f, cz - (d / 2));
            vec3 p1 = vec3(cx + (w / 2), h, cz + (d / 2));

            if (RND < 0.5f) {
                d_list[i++] = new box(p0, p1, new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
            }
            else {
                d_list[i++] = new box(p0, p1,
                    new metal(vec3(
                        0.5f * (1 + RND),
                        0.5f * (1 + RND),
                        0.5f * (1 + RND)
                    ), 0.2f * RND)
                );
            }
        }

        // We save back the RNG state
        *rand_state = local_rand_state;

        vec3 baseCenter(-30.0f, 0.0f, 2.0f);
        float halfWidth = 3.0f;
        float height = 3.0f;

        vec3 b0 = baseCenter + vec3(-halfWidth, 0, -halfWidth);
        vec3 b1 = baseCenter + vec3(+halfWidth, 0, -halfWidth);
        vec3 b2 = baseCenter + vec3(+halfWidth, 0, +halfWidth);
        vec3 b3 = baseCenter + vec3(-halfWidth, 0, +halfWidth);
        vec3 apex = baseCenter + vec3(0, height, 0);

		// We create a pyramid with 6 triangles
		material* red1 = new lambertian(vec3(0.8f, 0.3f, 0.3f));
        material* red2 = new lambertian(vec3(0.8f, 0.3f, 0.3f));
        material* red3 = new lambertian(vec3(0.8f, 0.3f, 0.3f));
        material* red4 = new lambertian(vec3(0.8f, 0.3f, 0.3f));
        material* red5 = new lambertian(vec3(0.8f, 0.3f, 0.3f));
        material* red6 = new lambertian(vec3(0.8f, 0.3f, 0.3f));

        //vec3 apex(0, 6, -15);
        //vec3 b0(-4, 0, -10), b1(4, 0, -10), b2(4, 0, -3), b3(-4, 0, -3);

		d_list[i++] = new triangle(apex, b0, b1, red1);
		d_list[i++] = new triangle(apex, b1, b2, red2);
		d_list[i++] = new triangle(apex, b2, b3, red3);
		d_list[i++] = new triangle(apex, b3, b0, red4);

		d_list[i++] = new triangle(b0, b1, b2, red5);
		d_list[i++] = new triangle(b0, b2, b3, red6);

		// We set up the world 22*22 little spheres, 3 big ones, 2 boxes and 300 random boxes, 1 pyramid (6 triangles)
        /**d_world = new hitable_list(d_list, 22 * 22 + 1 + 3 + 2 + 30 + 6);*/
		// We create a BVH tree for the world
        *d_world = new bvh_node(d_list, 22 * 22 + 1 + 3 + 2 + 30 + 6);


        // We set up the camera
        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus, 0.0f, 1.0f);
    }
}

// This function frees the world and camera on the GPU
__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {

    int n = 22 * 22 + 1 + 3 + 2 + 30 + 6;
    for (int i = 0; i < n; i++) {
		delete d_list[i]; // delete the hitable objects
    }

    // Delete list and camera objects
    delete* d_world;
    delete* d_camera;
}

int main() {
    size_t cur_stack;
    cudaDeviceGetLimit(&cur_stack, cudaLimitStackSize);
    std::cerr << "Default CUDA thread stack size: " << cur_stack << " bytes\n";
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 256 * 1024));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64 * 1024 * 1024));
    int nx = 1920; // image width
    int ny = 1080; // image height
    int ns = 10; // number of samples per pixel
    int tx = 16; // block dimensions
    int ty = 16;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int numPixels = nx * ny;
    size_t fb_size = numPixels * sizeof(vec3); // size of framebuffer

    // allocate framebuffer
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate RNG state arrays
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, numPixels * sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

    // Init the RNG state for the world creation
    rand_init << <1, 1 >> > (d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    hitable** d_list;
    int num_hitables = 22 * 22 + 1 + 3 + 2 + 30 + 6;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(hitable*)));

    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));

    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

    create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    // Init RNG state for each pixel
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // We now launch the kernel
    render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();

    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output framebuffer as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up allocated objects
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));

    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}