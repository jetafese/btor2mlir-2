//===- Btorc.cpp - The Btor Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Btor compiler.
//
//===----------------------------------------------------------------------===//

#include "btor/Dialect.h"
#include "btor/MLIRGen.h"
#include "btor/Parser.h"
#include <memory>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace btor;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input Btor file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Btor, MLIR };
}
static cl::opt<enum InputType> inputType(
    "x", cl::init(Btor), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Btor, "Btor", "load the input file as a Btor source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action { None, DumpAST, DumpMLIR };
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

/// Returns a Btor AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<btor::ModuleAST> parseInputFile(llvm::StringRef filename) {
  std::vector<std::unique_ptr<VariableExprAST>> args;
  std::string fnName = "main";
  btor::Location loc = {std::make_shared<std::string>("../../test/Examples/Btor2MLIR/ast.toy"), 0, 0};
  auto proto = std::make_unique<PrototypeAST>(std::move(loc), fnName, std::move(args));

  auto block = std::make_unique<ExprASTList>();

  btor::Location loc2 = {std::make_shared<std::string>("../../test/Examples/Btor2MLIR/ast.toy"), 2, 1};
  btor::Location loc4 = {std::make_shared<std::string>("../../test/Examples/Btor2MLIR/ast.toy"), 4, 1};
  llvm::StringRef var4 = "s0";
  std::unique_ptr<VarType> type4 = std::make_unique<VarType>();
  auto value4 = std::make_unique<NumberExprAST>(loc2, 0);
  auto init4 = std::make_unique<VarDeclExprAST>(loc4, var4, std::move(*type4), std::move(value4));
  block->push_back(std::move(init4));

  btor::Location loc5 = {std::make_shared<std::string>(std::move("../../test/Examples/Btor2MLIR/ast.toy")), 5, 1};
  llvm::StringRef var5 = "s0-next";
  std::unique_ptr<VarType> type5 = std::make_unique<VarType>();
  auto value5 = std::make_unique<NumberExprAST>(loc5, 1);
  auto init5 = std::make_unique<VarDeclExprAST>(loc5, var5, std::move(*type5), std::move(value5));
  block->push_back(std::move(init5));

  btor::Location loc6 = {std::make_shared<std::string>(std::move("../../test/Examples/Btor2MLIR/ast.toy")), 6, 1};
  auto s0Next = std::make_unique<VariableExprAST>(loc5, var5);
  auto s0 = std::make_unique<VariableExprAST>(loc4, var4);
  auto addOp = std::make_unique<BinaryExprAST>(loc6, '+', std::move(s0Next), std::move(s0));
  block->push_back(std::move(addOp));

  auto mainFunction = std::make_unique<FunctionAST>(std::move(proto), std::move(block));

  std::vector<FunctionAST> functions;
  functions.push_back(std::move(*mainFunction));

  return std::make_unique<ModuleAST>(std::move(functions));
}

int dumpMLIR() {
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::btor::BtorDialect>();

  // Handle '.Btor' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).endswith(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    mlir::OwningModuleRef module = mlirGen(context, *moduleAST);
    if (!module)
      return 1;

    module->dump();
    return 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningModuleRef module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }

  module->dump();
  return 0;
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Btor AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "Btor compiler\n");

  switch (emitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpMLIR:
    return dumpMLIR();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}
