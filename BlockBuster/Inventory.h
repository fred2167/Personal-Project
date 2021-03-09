#ifndef INVENTORY_H
#define INVENTORY_H

#include "MediaType.h"
#include "Transaction.h"
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace std;

class Transaction;

// forward declare Inventory for InventoryFactory, fully define later
class Inventory;

// abstract base class factory.
// Sub-class of inventory with factory should be a sub-class of this factory
class InventoryFactory {

public:
  virtual Inventory *create() = 0;
};

/* a *base abstract* class with an abstract factory.
   this class contains a static factory for all inventory factories.
   map <identifier :string, InventoryFactory *> factories
*/
class Inventory {

  friend ostream &operator<<(ostream &os, const Inventory &inv);
  friend istream &operator>>(istream &is, Inventory &inv);

protected:
  map<string, MediaType *> stocks;

  // Transaction are deleted in customer class.
  vector<Transaction *> histories;

private:
  //====================Virtual Function =============================

  // read itself, remember to set stock, superclass will call this in operator>>
  virtual istream &read(istream &is) = 0;

  // print yourself, superclass will call operator<< on this function
  virtual ostream &printer(ostream &os) const = 0;

public:
  // read for transaction. when creating a transaction, it will also create a
  // dummyInventory. Use dummyInventory to find actual inventory to manipulate
  // stock
  virtual istream &transactionRead(istream &is) = 0;

  // print for transaction.
  virtual ostream &transactionPrinter(ostream &os) const = 0;

  virtual bool operator<(Inventory &rhs) = 0;
  virtual bool operator>(Inventory &rhs) = 0;
  virtual bool operator==(Inventory &rhs) = 0;

  // sub-class may override destructor, remember to delete stocks
  virtual ~Inventory();
  //====================Getter/ Setter=================================

  bool hasMedia(string &type);
  void setMedia(string &type, int stock);
  MediaType *getMedia(string &type);

  // register stock of different media type after create inventory
  void registerMediaType(const string &type, MediaType *m);

  // register transaction to inventory history
  void addToInventoryHistory(Transaction *t);

  //====================Factory/Inventories map=========================
  // this method regiester to both the factory and inventory map
  static void registerInventory(const string &type, InventoryFactory *factory);

  static Inventory *createInventory(const string &type);

  static void insertInventory(const string &type, Inventory *inv);

  static set<Inventory *, bool (*)(Inventory *, Inventory *)> &
  getInventorySet(const string &type);

  static void displayInventories();

  static void deleteInventories();

private:
  /* this method ensure inventoryFactories is created before main
and in time for resgester*/
  static map<string, InventoryFactory *> &getInventoryFactories();

  // comparetor for the each inventory set which is stored in getInventories()
  static bool compare(Inventory *a, Inventory *b);

  static map<string, set<Inventory *, bool (*)(Inventory *, Inventory *)>> &
  getInventoriesMap();
};

#endif