#include "btor/BtorDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::btor;

#include "btor/BtorDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// BtorDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void BtorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "btor/BtorOps.cpp.inc"
      >();
}

// BtorDialect::BtorDialect(mlir::MLIRContext *ctx)
//     : mlir::Dialect(getDialectNamespace(), ctx, TypeID::get<BtorDialect>()) {
    
//     addOperations<
//     #define GET_OP_LIST
//     #include "btor/BtorOps.cpp.inc"
//     >();
// }

//===----------------------------------------------------------------------===//
// ConstantOp

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       bool value) {
    auto dataType = RankedTensorType::get({}, builder.getI1Type());
    auto dataAttribute = DenseIntElementsAttr::get(dataType, value);
    ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
    mlir::DenseIntElementsAttr value;

    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseAttribute(value, "value", result.attributes)) {       
        return mlir::failure(); 
    }
    result.addTypes(value.getType());
    return success();                                            
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
static void print(mlir::OpAsmPrinter &printer, ConstantOp op) {
    printer << "btor.constant";
    printer.printOptionalAttrDict(op.getAttrs(), {"value"});
    printer << op.value();
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.
static mlir::LogicalResult verify(ConstantOp op) {
    
    auto resultType = op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
    // Unranked tensors cannot be shape checked. 
    if (!resultType) {
        return success();
    }

    auto attrType = op.value().getType().cast<mlir::TensorType>();
    if (attrType.getRank() != resultType.getRank()) {
        return op.emitOpError(
               "return type rank must match the one of the attached value "
               "attribute: Attribute rank ")
           << attrType.getRank() << " != Result rank " 
           << resultType.getRank();
    }

    for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
        if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
            return op.emitOpError(
                "return type shape mismatches its attribute at dimension ")
                << dim << ". Attribute dim " << dim << " length "
                << attrType.getShape()[dim]
                << " != Result dim " << dim << " length " 
                << resultType.getShape()[dim];
        }
    }
    return mlir::success();
}

//===----------------------------------------------------------------------===//
// BinaryOp

static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
    SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
    llvm::SMLoc operandsLoc = parser.getCurrentLocation();
    Type type;
    
    if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type)) {
            return mlir::failure();
    }

    if (FunctionType funcType = type.dyn_cast<FunctionType>()) {
        if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                                   result.operands)) {
            return mlir::failure();
        }
        result.addTypes(funcType.getResults());
        return mlir::success();
    }

    if (parser.resolveOperands(operands, type, result.operands)) {
        return mlir::failure();
    }
    result.addTypes(type);
    return mlir::success();
}

static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
    printer << op->getName() << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " : ";

    Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(), 
                     [=](Type type) { return type == resultType; })) {
        printer << resultType;
    }
    return;
    printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// AndOp

void BvAndOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getI1Type()));
    state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// GenericCallOp

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
    state.addTypes(UnrankedTensorType::get(builder.getI1Type()));
    state.addOperands(arguments);
    state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

static mlir::LogicalResult verify(ReturnOp op) {
    auto function = cast<FuncOp>(op->getParentOp());

    if (op.getNumOperands() > 1) {
        return op.emitOpError() << "expects at most 1 return operand";
    }

    const auto &results = function.getType().getResults();
    if (op.getNumOperands() != results.size()) {
        return op.emitOpError() 
               << "does not return the same number of values ("
               << op.getNumOperands() << ") as the enclosing function ("
               << results.size() << ")";
    }

    if (!op.hasOperand()) {
        return mlir::success();
    }
    //Check the reurn operands
    auto inputType = *op.operand_type_begin();
    auto resultType = results.front();

    //Unranked tensors cannot have their sies checked
    if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
        resultType.isa<mlir::UnrankedTensorType>()) {
        
        return mlir::success(); 
    }

    return op.emitError() << "type of return operand (" << inputType
                          << ") doesn't match function result type ("
                          << resultType << ")";
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "btor/BtorOps.cpp.inc"  