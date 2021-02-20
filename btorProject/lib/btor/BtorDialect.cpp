#include "btor/BtorDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::btor;

BtorDialect::BtorDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx, TypeID::get<BtorDialect>()) {
    
    addOperations<
    #define GET_OP_LIST
    #include "btor/BtorOps.cpp.inc"
    >();
}

void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       bool value) {
    auto dataType = I1();
    ConstantOp::build(builder, state, dataType, {});
}

static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
    mlir::BoolArrayAttr value;

    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseAttribute(value, "value", result.attributes)) {       
        return mlir::failure(); 
    }
    result.addTypes(value.getType());
    return mlir::success();                                            
}

static void print(mlir::OpAsmPrinter &printer, ConstantOp op) {
    printer << "btor.constant";
    printer.printOptionalAttrDict(op.getAttrs(), {"value"});
    printer << op.value();
}

static mlir::LogicalResult verify(ConstantOp op) {
    /*
    // If both the constant attribute and return type are boolean, op is valid. 
    if (op.getResult().getType().dyn_cast<mlir::btor::BoolType>() && 
        op.value().getType().case<mlir::btor::BoolType>()) {
        
        return mlir::success();
    }

    auto resultType = op.getResult().getType().dyn_cast<mlir::BoolArrayAttr>();
    if (!resultType) {
        return op.emitOpError(
            "Invalid type. btor2mlir language only supports bit vectors and "
            "booleans.");
    }

    auto attrType = op.value().getType().cast<mlir::BoolArrayAttr>();
    if (attrType.)*/
    return mlir::success;
}

#define GET_OP_CLASSES
#include "btor/BtorOps.cpp.inc"  