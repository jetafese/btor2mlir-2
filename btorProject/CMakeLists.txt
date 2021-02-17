set(LLVM_LINK_COMPONENTS
  Support
  )

add_executable(btor2mlir btor2exe.cpp)

include_directories(include/)
target_link_libraries(btor2mlir
  PRIVATE
    MLIRSupport)

