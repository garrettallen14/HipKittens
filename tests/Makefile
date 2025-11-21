HIPCC=/opt/rocm/bin/hipcc
ARCH=gfx942

all: hello_hip

hello_hip: hello_hip.cpp
	$(HIPCC) hello_hip.cpp -o hello_hip --offload-arch=$(ARCH)

clean:
	rm -f hello_hip
