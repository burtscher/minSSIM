NV_SM := 70

all: serial openmp gpu

serial:
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o float_analysis src/metrics.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o sblc_compress_ser src/compressor-standalone.cpp
	g++ -O3 -march=native -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o sblc_decompress_ser src/decompressor-standalone.cpp

openmp:
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o float_analysis src/metrics.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o sblc_compress_omp src/compressor-standalone.cpp
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o sblc_decompress_omp src/decompressor-standalone.cpp

gpu:
	g++ -O3 -march=native -fopenmp -mno-fma -ffp-contract=off -I./src/ -std=c++17 -o float_analysis src/metrics.cpp
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o sblc_compress_gpu src/compressor-standalone.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o sblc_decompress_gpu src/decompressor-standalone.cu
	nvcc -O3 -arch=sm_$(NV_SM) -fmad=false -Xcompiler "-O3 -march=native -fopenmp -mno-fma -ffp-contract=off" -I./src/ -o base_compress_gpu src/compressor-standalone_nominssim.cu

clean:
	rm -rf sblc_*compress_*
	rm -rf float_analysis