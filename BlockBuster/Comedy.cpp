#include "Comedy.h"

ComedyFactory::ComedyFactory() {
  // identifer: F as Funny
  Inventory::registerInventory("F", this);
}

Inventory *ComedyFactory::create() { return new Comedy(); }

namespace {
// call constructor to register comFactory to the static factories in base class
ComedyFactory comFactory;
} // namespace
//===========================================================

bool Comedy::operator<(Inventory &rhs) {
  Comedy &com = dynamic_cast<Comedy &>(rhs);
  if (title != com.title) {
    return title < com.title;
  }
  return yearRelease < com.yearRelease;
}

bool Comedy::operator>(Inventory &rhs) {
  Comedy &com = dynamic_cast<Comedy &>(rhs);
  if (title != com.title) {
    return title > com.title;
  }
  return yearRelease > com.yearRelease;
}

bool Comedy::operator==(Inventory &rhs) {
  Comedy &com = dynamic_cast<Comedy &>(rhs);
  return title == com.title && yearRelease == com.yearRelease;
}

istream &Comedy::read(istream &is) {
  // stock, director, title, year
  director = util::readNextItem(is, ',');
  title = util::readNextItem(is, ',');
  is >> yearRelease;
  util::eatGarbage(is, '\n');
  return is;
}

istream &Comedy::transactionRead(istream &is) {
  title = util::readNextItem(is, ',');
  is >> yearRelease;
  return is;
}

ostream &Comedy::printer(ostream &os) const {
  os << "Comedy: " << title << ", " << director << ", " << yearRelease << ", ";
  for (auto &pair : stocks) {
    os << *pair.second << ", " << endl;
  }
  return os;
}

ostream &Comedy::transactionPrinter(ostream &os) const {
  os << title << ", " << yearRelease << endl;
  return os;
}
