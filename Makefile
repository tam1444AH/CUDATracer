CUDA_PATH     := C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8
NVCC          := "$(CUDA_PATH)/bin/nvcc.exe"

GENCODE_FLAGS := -gencode arch=compute_86,code=sm_86
NVCCFLAGS     := -m64 -rdc=true

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h moving_sphere.h aabb.h bvh.h plane.h box.h

all: cudart.exe

cudart.exe: $(SRCS) $(INCS)
	@echo Building $@ with CUDA Toolkit @ $(CUDA_PATH)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -Xcompiler="/EHsc,/std:c++14" -o $@ $(SRCS)

out.ppm: cudart.exe
	@echo Rendering image → $@
	./cudart.exe > $@

out.jpg: out.ppm
	@echo Converting $< → $@
	magick convert out.ppm out.jpg

profile: cudart.exe
	nvprof ./cudart.exe > profile.log

clean:
	del /Q cudart.exe out.ppm out.jpg profile.log 2>nul || rm -f cudart.exe out.ppm out.jpg profile.log

.PHONY: all profile clean
