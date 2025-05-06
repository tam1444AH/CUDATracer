# CUDA Path Tracer

This is a CUDA-based ray tracer that renders 3D scenes using GPU acceleration.  
It includes materials, spheres, boxes, BVH structures, and fog support.

---

##  Prerequisites

- **Windows** with:
  - **CUDA Toolkit 12.8**
  - **Visual Studio 2022**
  - **NVIDIA GPU (Compute Capability ≥ 8.6)** — e.g., RTX 30-series
- **x64 Native Tools Command Prompt for VS 2022**
-  **ImageMagick** for converting `.ppm` → `.jpg`
- **Make from GnuWin32**

---

##  Build Instructions

### 1. Open x64 Native Tools Command Prompt for VS 2022

### 2. Navigate to the Project

**cmd
cd "C:\Users\YourName\source\repos\CUDATracer"**

### 3. Build the Project

In the x64 Native Tools Command Prompt, while in the project directory run:
**make clean**

This should display:

**del /Q cudart.exe out.ppm out.jpg profile.log 2>nul || rm -f cudart.exe out.ppm out.jpg profile.log**

Then run **make** to build the project.

The prompt should display the following:

**"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8/bin/nvcc.exe" -ccbin cl  -m64 -rdc=true -gencode arch=compute_86,code=sm_86 -Xcompiler "/EHsc /std:c++14" \
       -o cudart.exe main.cu
main.cu
tmpxft_00004798_00000000-7_main.cudafe1.cpp
nvlink warning : Stack size for entry function '_Z12create_worldPP7hitableS1_PP6cameraiiP17curandStateXORWOW' cannot be statically determined
nvlink warning : Stack size for entry function '_Z6renderP4vec3iiiPP6cameraPP7hitableP17curandStateXORWOW' cannot be statically determined
   Creating library cudart.lib and object cudart.exp**

### 4. Run the Project

Then run in the command prompt: 

**cudart.exe > output.ppm**

Which should display:

*Default CUDA thread stack size: 1024 bytes
Rendering a 1920x1080 image with 10 samples per pixel in 16x16 blocks.
took 2.928 seconds.*

### 5. Convert the PPM to JPG
Finally, to see the image you can convert the PPM to JPG using ImageMagick.

**magick output.ppm output.jpg**

This should create a file called `output.jpg` in the project directory. Which will show the raytraced image of the code that is provided.
