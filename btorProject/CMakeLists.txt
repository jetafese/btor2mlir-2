# For a better template to copy, see examples/standalone
add_subdirectory(include)

add_custom_target(Btor)
set_target_properties(Btor PROPERTIES FOLDER Examples)

macro(add_toy_chapter name)
  add_dependencies(Btor ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_toy_chapter name)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_toy_chapter(btor2mlir-ch2
  btorc.cpp
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
