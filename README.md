# btor2mlir

Build the MLIR repo using the instructions in https://mlir.llvm.org/getting_started/. You'll get a folder `llvm-project`.

Clone this repo into `llvm-project/mlir/examples/`. Erase the contents of `llvm-project/mlir/examples/CMakeLists.txt` and replace it with one line: `add_subdirectory(btor2mlir)`. This will let cmake compile the toy folder in our repo instead of the default toy folder. 

As an interim way of building btor2mlir files within the MLIR repo, open `llvm-project/mlir/test/CMakeLists.txt` and find the following section:

```
if(LLVM_BUILD_EXAMPLES)
  list(APPEND MLIR_TEST_DEPENDS
    toyc-ch1
    ...
  )
endif()
```

Append the executable name `btor2mlir` to the `list()`. The changed section should look like:

```
if(LLVM_BUILD_EXAMPLES)
  list(APPEND MLIR_TEST_DEPENDS
    toyc-ch1
    ...
    btor2mlir
  )
endif()
```

You can now build the `llvm-project` repo again and run the `btor2mlir` executable file in `llvm-project/build/bin`.

Gotchas:
 - The #include guards for .td files require #include guard comments for the build to succeed. `clang-tidy-10` on lilla can o this (`clang-tidy-10 -fix-errors file.h`), but `clang-tidy` won't deal well with tablegen syntax.
 - The #define macros `GET_OP_CLASSES` and `GET_OP_LIST` for Tablegen definitions are not to be customized to the name of your dialect. Use those exact #define macro names.   
 - The "def" types in TableGen files can only be accessed by TableGen files. Type names that are in quotation marks (i.e. directly injected into the auto-generated code) must come from C++ files.  