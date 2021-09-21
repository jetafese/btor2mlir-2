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
  // llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
  //     llvm::MemoryBuffer::getFileOrSTDIN(filename);
  // if (std::error_code ec = fileOrErr.getError()) {
  //   llvm::errs() << "Could not open input file: " << ec.message() << "\n";
  //   return nullptr;
  // }
  // auto buffer = fileOrErr.get()->getBuffer();
  // LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  // Parser parser(lexer);
  // // llvm::errs() << parser.parseModule() << "\n";
  // return parser.parseModule();

  std::vector<std::unique_ptr<VariableExprAST>> args;
  std::string fnName = "main";
  btor::Location loc = {std::make_shared<std::string>("../../test/Examples/Btor2MLIR/ast.toy"), 0, 0};
  auto proto = std::make_unique<PrototypeAST>(std::move(loc), fnName, std::move(args));

  auto block = std::make_unique<ExprASTList>();

  btor::Location loc3 = {std::make_shared<std::string>("../../test/Examples/Btor2MLIR/ast.toy"), 3, 1};
  llvm::StringRef var4 = "s0";
  std::unique_ptr<VarType> type4 = std::make_unique<VarType>();
  auto value4 = std::make_unique<NumberExprAST>(loc3, 0);
  auto init4 = std::make_unique<VarDeclExprAST>(loc3, var4, std::move(*type4), std::move(value4));
  block->push_back(std::move(init4));

  std::vector<std::unique_ptr<ExprAST>> argsCallWhile;
  // argsCallWhile.push_back(std::make_unique<VariableExprAST>(loc3, "true"));
  argsCallWhile.push_back(std::make_unique<VariableExprAST>(loc3, "s0"));
  const std::string callee = "while";
  auto callWhile = std::make_unique<CallExprAST>(loc3, callee, std::move(argsCallWhile));
  block->push_back(std::move(callWhile));

  // represent while as function call
  btor::Location loc7 = {std::make_shared<std::string>("../../test/Examples/Btor2MLIR/ast.toy"), 7, 1};
  std::vector<std::unique_ptr<VariableExprAST>> argsWhile;
  argsWhile.push_back(std::make_unique<VariableExprAST>(loc3, "s0"));
  auto protoWhile = std::make_unique<PrototypeAST>(std::move(loc7), "while", std::move(argsWhile));
  auto blockWhile = std::make_unique<ExprASTList>();
    // add next
  btor::Location loc6 = {std::make_shared<std::string>(std::move("../../test/Examples/Btor2MLIR/ast.toy")), 6, 1};
  llvm::StringRef var6 = "s0-next";
  std::unique_ptr<VarType> type6 = std::make_unique<VarType>();
  auto value6 = std::make_unique<NumberExprAST>(loc6, 1);
  auto add6 = std::make_unique<VarDeclExprAST>(loc6, var6, std::move(*type6), std::move(value6));
  blockWhile->push_back(std::move(add6));
    // return next
  auto s0Next = std::make_unique<VariableExprAST>(loc6, var6);
  auto arg1 = std::make_unique<VariableExprAST>(loc3, var4);
  auto addOp = std::make_unique<BinaryExprAST>(loc6, '+', std::move(s0Next), std::move(arg1));
  // auto returnOp = std::make_unique<ReturnExprAST>(loc7, std::move(addOp));
  blockWhile->push_back(std::move(addOp));

  auto whileLoop = std::make_unique<FunctionAST>(std::move(protoWhile), std::move(blockWhile));
  auto mainFunction = std::make_unique<FunctionAST>(std::move(proto), std::move(block));

  std::vector<FunctionAST> functions;
  functions.push_back(std::move(*mainFunction));
  functions.push_back(std::move(*whileLoop));

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
