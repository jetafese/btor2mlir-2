set(LLVM_TARGET_DEFINITIONS BtorOps.td)
mlir_tablegen(BtorOps.h.inc -gen-op-decls)
mlir_tablegen(BtorOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(BtorOpsIncGen)