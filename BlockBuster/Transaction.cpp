#include "Transaction.h"

ostream &operator<<(ostream &os, const Transaction &t) { return t.printer(os); }

istream &operator>>(istream &is, Transaction &t) { return t.read(is); }

map<string, TransactionFactory *> &Transaction::getTransactionFactory() {
  static map<string, TransactionFactory *> transactionFactories;
  return transactionFactories;
}

void Transaction::registerTransactionFactory(const string &type,
                                             TransactionFactory *factory) {
  getTransactionFactory().emplace(type, factory);
}

bool Transaction::hasTransactionFactory(const string &type) {
  return getTransactionFactory().count(type) != 0;
}

Transaction *Transaction::createTransaction(const string &type) {
  if (!hasTransactionFactory(type)) {
    cout << "Don't know how to create " << type << " as a Transaction." << endl;
    return nullptr;
  }
  return getTransactionFactory().at(type)->create();
}