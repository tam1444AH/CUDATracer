CUDA_PATH     := C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8

HOST_COMPILER := cl

NVCC           := "$(CUDA_PATH)/bin/nvcc.exe" -ccbin $(HOST_COMPILER)

GENCODE_FLAGS  := -gencode arch=compute_86,code=sm_86

NVCC_DBG       :=

NVCCFLAGS      := $(NVCC_DBG) -m64
NVCCFLAGS      += -rdc=true
HOST_FLAGS     := /EHsc /std:c++14

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h moving_sphere.h aabb.h bvh.h plane.h box.h

all: cudart.exe

cudart.exe: $(SRCS) $(INCS)
	@echo "Building $@ with CUDA Toolkit @ $(CUDA_PATH)"
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -Xcompiler "$(HOST_FLAGS)" \
       -o $@ $(SRCS) $(LDFLAGS)

out.ppm: cudart.exe
	@echo "Rendering image → $@"
	./cudart.exe > $@

out.jpg: out.ppm
	@echo "Converting $< → $@"
	magick convert out.ppm out.jpg

profile: cudart.exe
	nvprof ./cudart.exe > profile.log

clean:
	del /Q cudart.exe out.ppm out.jpg profile.log 2>nul || rm -f cudart.exe out.ppm out.jpg profile.log

.PHONY: all profile clean
