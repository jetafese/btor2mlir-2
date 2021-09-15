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
    btor2mlir-ch2
  )
endif()
```

You can now build the `llvm-project` repo again and run the `btor2mlir` executable file in `llvm-project/build/bin`. Feel free to use the run script in this repository. 

# Running the current state

To see the mlir that gets generated for our btor2 example, run the following command: `~/llvm-project/build/bin/btor2mlir-ch2  -emit=mlir`

On the other hand, to see the generated AST for our btor2 example, run this: `~/llvm-project/build/bin/btor2mlir-ch2  -emit=ast`

Keep in mind that we are currently using a release build, as opposed to a debug build. This readme will be updated when that has been set up. 
