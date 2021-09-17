#Having two include_directories is a hack; 
#will need to figure out how to do the 
#tablegens properly.
include_directories(${MLIR_INCLUDE_DIR})
include_directories(include/) 

add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_custom_target(BTOR)

macro(add_toy_chapter name)
  add_dependencies(BTOR ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_toy_chapter name)

add_toy_chapter(btor2mlir
  btor2exe.cpp  
  lib/btor/BtorDialect.cpp

  DEPENDS
  BtorOpsIncGen
)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(btor2mlir
  PRIVATE
    MLIRSupport
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)

