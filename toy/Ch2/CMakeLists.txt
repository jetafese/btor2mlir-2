# For a better template to copy, see examples/standalone
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_toy_chapter(btor2mlir-ch2
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp

  DEPENDS
  ToyCh2OpsIncGen

  )
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(btor2mlir-ch2
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)
