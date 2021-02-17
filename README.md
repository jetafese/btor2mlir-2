# btor2mlir

Build the MLIR repo using the instructions in https://mlir.llvm.org/getting_started/. You'll get a folder `llvm-project`.

Clone this repo into `llvm-project/mlir/examples/`. Erase the contents of `llvm-project/mlir/examples/CMakeLists.txt` and replace it with one line: `add_subdirectory(btor2mlir)`. This will let cmake compile the toy folder in our repo instead of the default toy folder. 
