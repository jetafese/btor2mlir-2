set(LLVM_TARGET_DEFINITIONS BtorOps.td)
mlir_tablegen(BtorOps.h.inc -gen-op-decls -I /home/fcheng/llvm-project/mlir/include)
mlir_tablegen(BtorOps.cpp.inc -gen-op-defs -I /home/fcheng/llvm-project/mlir/include)
add_public_tablegen_target(BtorOpsIncGen)
