#ifndef BTOR_BTORDIALECT_H_
#define BTOR_BTORDIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
namespace mlir {
    namespace btor {
        class BtorDialect : public mlir::Dialect {
        public:
            explicit BtorDialect(mlir::MLIRContext *ctx);
            static llvm::StringRef getDialectNamespace() { return "btor";}
        };
    }
}
#define GET_OP_CLASSES
#include "btor/BtorOps.h.inc"    

#endif // BTOR_BTORDIALECT_H_