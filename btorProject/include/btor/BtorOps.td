#ifndef BTOR_OPS
#define BTOR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Btor_Dialect : Dialect {
    let name = "btor";
    let cppNamespace = "::mlir::btor";
}

class Btor_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Btor_Dialect, mnemonic, traits>;

def ConstantOp : Btor_Op<"constant", [ NoSideEffect ]> {
    let summary = "Turns literals to SSA values.";
    let description = [{
            Same function as the ConstantOp in the Toy example.
            Turns bit vector literals to btor.constant operations that have the
                literal's value as an operation property. 
        }]; 
        let arguments = (ins BoolArrayAttr
             : $value);

        let results = (outs I1MemRef);

        let parser = [{ return ::parseConstantOp(parser, result); }];
        let printer = [{ return ::print(p, (*this)); }];
        let verifier = [{ return ::verify(*this); }];

        let builders = [
            OpBuilderDAG<(ins "BoolArrayAttr"
                          : $value),
                         [{build($_builder, $_result, value.getType(),
                                  value)}]>,
            OpBuilderDAG<(ins "I1"
                          : $value)>
        ];
}

def BvAndOp : Btor_Op<"bvand", [ NoSideEffect ]> {
        let summary = "Bit vector and operation";
        let description = [{Performs element-wise AND on 
            two bitvectors of the same size.}];
}

#endif // BTOR_OPS