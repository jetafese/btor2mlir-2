cd ~/llvm-project/build
cmake -G Ninja ../llvm \
	-DCMAKE_C_COMPILER=clang-10 \ 
	-DCMAKE_CXX_COMPILER=clang++-10 \
	-DLLVM_ENABLE_PROJECTS=mlir \
	-DLLVM_BUILD_EXAMPLES=ON  \
	-DCMAKE_BUILD_TYPE=Debug -L \
	-DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
	-DLLVM_ENABLE_LLD=ON  \
	-DCMAKE_BUILD_PREFIX=$(pwd)/run
cmake --build .
cd ~/llvm-project/mlir/examples/btor2mlir

