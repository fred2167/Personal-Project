#include "Inventory.h"

ostream &operator<<(ostream &os, const Inventory &inv) {
  return inv.printer(os);
}

istream &operator>>(istream &is, Inventory &inv) { return inv.read(is); }

Inventory::~Inventory() {
  for (auto &pair : stocks) {
    delete pair.second;
    pair.second = nullptr;
  }
}

bool Inventory::hasMedia(string &type) { return stocks.count(type) > 0; }

void Inventory::setMedia(string &type, int stock) {
  if (hasMedia(type)) {
    stocks.at(type)->setStock(stock);
  } else {
    cout << "Do NOT has " << type << " to set stock.";
  }
}
MediaType *Inventory::getMedia(string &type) {
  if (hasMedia(type)) {
    return stocks.at(type);
  }
  cout << "Do NOT has " << type << " to get stock.";
  return nullptr;
}

void Inventory::registerMediaType(const string &type, MediaType *m) {
  stocks.emplace(type, m);
}

void Inventory::addToInventoryHistory(Transaction *t) {
  histories.push_back(t);
}

map<string, InventoryFactory *> &Inventory::getInventoryFactories() {
  /* this method ensure inventoryFactories is created before main
  and in time for resgester*/
  static map<string, InventoryFactory *> inventoryFactories;
  return inventoryFactories;
}

map<string, set<Inventory *, bool (*)(Inventory *, Inventory *)>> &
Inventory::getInventoriesMap() {
  /* this method ensure inventories is created before main
  and in time for resgester*/
  static map<string, set<Inventory *, bool (*)(Inventory *, Inventory *)>>
      inventories;
  return inventories;
}

bool Inventory::compare(Inventory *a, Inventory *b) { return *a < *b; }

void Inventory::registerInventory(const string &type,
                                  InventoryFactory *factory) {
  if (getInventoryFactories().count(type) != 0 ||
      getInventoriesMap().count(type) != 0) {
    cout << "Inventory as type *" << type
         << "* already exists. Register Failed." << endl;
    return;
  }

  getInventoryFactories().emplace(type, factory);
  getInventoriesMap().emplace(type, compare);
}

Inventory *Inventory::createInventory(const string &type) {
  if (getInventoryFactories().count(type) == 0) {
    cout << "Don't know how to create " << type << " as an Inventory." << endl;
    return nullptr;
  }
  return getInventoryFactories().at(type)->create();
}

void Inventory::insertInventory(const string &type, Inventory *inv) {
  if (getInventoriesMap().count(type) == 0) {
    cout << "Don't know how to insert " << type << " as an Inventory." << endl;
    return;
  }
  getInventoriesMap().at(type).insert(inv);
}

set<Inventory *, bool (*)(Inventory *, Inventory *)> &
Inventory::getInventorySet(const string &type) {
  if (getInventoriesMap().find(type) == getInventoriesMap().end()) {
    cout << "Don't know  " << type << " as an Inventory identifier." << endl;
    // dummy stl;
    static set<Inventory *, bool (*)(Inventory *, Inventory *)> s(compare);
    return s;
  }
  return getInventoriesMap().find(type)->second;
}

void Inventory::displayInventories() {
  for (auto iter = getInventoriesMap().rbegin();
       iter != getInventoriesMap().rend(); iter++) {
    for (auto inv : iter->second) {
      cout << *inv;
    }
  }
}

void Inventory::deleteInventories() {
  for (auto &pair : getInventoriesMap()) {
    for (auto inv : pair.second) {
      delete inv;
      inv = nullptr;
    }
  }
}
