#ifndef TRANSACTION_H
#define TRANSACTION_H

#include "MediaType.h"
#include "Store.h"
#include <iostream>

using namespace std;

class Store;

class TransactionFactory;

class Transaction {

  friend ostream &operator<<(ostream &os, const Transaction &t);
  friend istream &operator>>(istream &is, Transaction &t);

protected:
  bool validFlag = false;

private:
  //====================Virtual Function =============================
  // read itself, remember to set stock, superclass will call this in operator>>
  virtual istream &read(istream &is) = 0;

  // print yourself, superclass will call operator<< on this function
  virtual ostream &printer(ostream &os) const = 0;

public:
  // copy not allowed
  Transaction(const Transaction &rhs) = delete;

  // move not allowed
  Transaction(Transaction &&rhs) = delete;

  // assignment not allowed
  Transaction &operator=(const Transaction &rhs) = delete;

  // move assignment not allowed
  Transaction &operator=(const Transaction &&rhs) = delete;

  Transaction() = default;

  virtual ~Transaction() = default;

  // call in class store
  virtual bool process(Store &s) = 0;

  //====================Factory======================================
  /* this method ensure transactionFactories is created before main
  and in time for resgester*/
  static map<string, TransactionFactory *> &getTransactionFactory();

  static void registerTransactionFactory(const string &type,
                                         TransactionFactory *factory);

  static bool hasTransactionFactory(const string &type);

  static Transaction *createTransaction(const string &type);
};

class TransactionFactory {

public:
  virtual Transaction *create() const = 0;
};

#endif