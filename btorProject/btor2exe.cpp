#include <iostream>
#include "mlir/IR/BuiltinTypes.h"
#include "btor/BtorDialect.h"
int main(int argc, char** argv) {
    mlir::VectorType a;
    std::cout << mlir::btor::BtorDialect::getDialectNamespace().str(); 
    return 0;
}