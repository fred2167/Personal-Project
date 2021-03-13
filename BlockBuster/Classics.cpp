#include "Classics.h"

ClassicsFactory::ClassicsFactory() {
  // identifer: C as Classics
  Inventory::registerInventory("C", this);
}

Inventory *ClassicsFactory::create() { return new Classics(); }

namespace {
// call constructor to register classicsFactory to the static factories in base
// class
ClassicsFactory classicsFactory;
} // namespace
//===========================================================

Classics::Classics() { type = "C"; }

bool Classics::operator<(Inventory &rhs) {
  auto &classics = dynamic_cast<Classics &>(rhs);
  if (yearRelease != classics.yearRelease) {
    return yearRelease < classics.yearRelease;
  }
  if (monthRelease != classics.monthRelease) {
    return monthRelease < classics.monthRelease;
  }
  return majorActor < classics.majorActor;
}

bool Classics::operator>(Inventory &rhs) {
  auto &classics = dynamic_cast<Classics &>(rhs);
  if (yearRelease != classics.yearRelease) {
    return yearRelease > classics.yearRelease;
  }
  if (monthRelease != classics.monthRelease) {
    return monthRelease > classics.monthRelease;
  }
  return majorActor > classics.majorActor;
}

bool Classics::operator==(Inventory &rhs) {
  auto &classics = dynamic_cast<Classics &>(rhs);
  return monthRelease == classics.monthRelease &&
         yearRelease == classics.yearRelease &&
         majorActor == classics.majorActor;
}

istream &Classics::read(istream &is) {
  // director, title, actor month year
  director = util::readNextItem(is, ',');
  title = util::readNextItem(is, ',');
  util::eatGarbage(is, ' ');
  majorActor = util::readNextItem(is, ' ') + " ";
  majorActor += util::readNextItem(is, ' ');
  is >> monthRelease;
  is >> yearRelease;
  util::eatGarbage(is, '\n');
  return is;
}

istream &Classics::transactionRead(istream &is) {
  // month year actor
  is >> monthRelease;
  is >> yearRelease;
  majorActor = util::readNextItem(is, '\n');

  return is;
}

ostream &Classics::printer(ostream &os) const {
  os << "Classics: " << title << ", " << director << ", " << monthRelease << " "
     << yearRelease << ", " << majorActor << ", ";
  for (auto &pair : stocks) {
    os << *pair.second << ", " << endl;
  }
  return os;
}

ostream &Classics::transactionPrinter(ostream &os) const {
  os << majorActor << ", " << monthRelease << " " << yearRelease << endl;
  return os;
}
