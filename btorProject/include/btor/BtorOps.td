#ifndef BTOR_OPS
#define BTOR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def I1ElementsAttr : SignlessIntElementsAttr<1>;

def Btor_Dialect : Dialect {
    let name = "btor";
    let summary = "A dialect for handling the btor format";
    let description = [{
        See the btor format for semantics. This dialect
        provides a representation of the language that is 
        amenable to analysis and optimization.
    }];
    let cppNamespace = "::mlir::btor";
}

class Btor_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Btor_Dialect, mnemonic, traits>;

def ConstantOp : Btor_Op<"constant", [ NoSideEffect ]> {
    let summary = "turns literals to SSA values.";
    let description = [{
            Same function as the ConstantOp in the Toy example.
            Turns bit vector literals to btor constant operations that have the
                literal's value as an operation property. 
        }]; 
        //Takes in a BoolArrayAttr named value 
        let arguments = (ins I1ElementsAttr: $value);

        let results = (outs I1Tensor);
        
        //These are defined in BtorDialect.cpp
        let parser = [{ return ::parseConstantOp(parser, result); }];
        let printer = [{ return ::print(p, (*this)); }];
        let verifier = [{ return ::verify(*this); }];

        let builders = [
            //Build a constant bit vector given a constant bitvetor value. 
            OpBuilderDAG<(ins "DenseIntElementsAttr"
                          : $value),
                         [{build($_builder, $_state, value.getType(),
                                  value);}]>,
            //Build a constant bitvector of length 1 given a constant boolean.
            //Defined in BtorDialect.cpp
            OpBuilderDAG<(ins "bool" : $value)>
        ];
}

def BvAndOp : Btor_Op<"bvand", [ NoSideEffect ]> {
        let summary = "bit vector AND operation";
        let description = [{Returns element-wise AND of 
            two bitvectors of the same size.}];

        let arguments = (ins I1Tensor: $lhs, F64Tensor:$rhs);
        let results = (outs I1Tensor);

        let parser = [{ return ::parseBinaryOp(parser, result); }];
        let printer = [{ return printBinaryOp(p, *this); }];

        let builders = [
            OpBuilderDAG<(ins "Value":$lhs, "Value": $rhs)>
        ];
}

def BvOrOp : Btor_Op<"bvor", [ NoSideEffect ]> {
        let summary = "Bit vector OR operation";
        let description = [{Returns element-wise OR of 
            two bitvectors of the same size.}];

        let arguments = (ins I1Tensor: $lhs, F64Tensor:$rhs);
        let results = (outs I1Tensor);

        let parser = [{ return ::parseBinaryOp(parser, result); }];
        let printer = [{ return printBinaryOp(p, *this); }];

        let builders = [
            OpBuilderDAG<(ins "Value":$lhs, "Value": $rhs)>
        ];
}

def BvNotOp : Btor_Op<"bvnot", [ NoSideEffect ]> {
        let summary = "Bit vector NOT operation";
        let description = [{Returns element-wise NOT of 
            a bitvectors.}];
        
        let arguments = (ins I1Tensor: $lhs, F64Tensor:$rhs);
        let results = (outs I1Tensor);

        let parser = [{ return ::parseBinaryOp(parser, result); }];
        let printer = [{ return printBinaryOp(p, *this); }];

        let builders = [
            OpBuilderDAG<(ins "Value":$lhs, "Value": $rhs)>
        ];
}

//Cutting ITE operations for now. Not sure how to take expressions as inputs
// def BvIteOp : Btor_Op<"bvIte", [ NoSideEffect ]> {
//         let summary = "Bit vector ite operation";
//         let description = [{Returns the bitvector result of an "if-then-else"
//             operation.}];

//         //let arguments = (ins ,)
// }

def GenericCallOp : Btor_Op<"generic_call"> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<I1Tensor>:$inputs);

  // The generic call operation returns a single value of TensorType.
  let results = (outs I1Tensor);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilderDAG<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

def PrintOp : Btor_Op<"print"> {
    let summary = "print operation";
    let description = [{
        Prints a given bitvector, produces no results.
    }];

    let arguments = (ins I1Tensor: $input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReturnOp : Btor_Op<"return", [NoSideEffect, HasParent<"FuncOp">, 
                                  Terminator]> {
    let summary = "return operation";
    let description = [{
        Custom return function. Copying from toy example.
    }];
    let arguments = (ins Variadic<I1Tensor>:$input);
    let assemblyFormat = "($input^ `:` type($input))? attr-dict";

    let builders = [
        OpBuilderDAG<(ins), [{build($_builder, $_state, llvm::None); }]>
    ];

    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

    let verifier = [{ return ::verify(*this); }];
}

#endif // BTOR_OPS