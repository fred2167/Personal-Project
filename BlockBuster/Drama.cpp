#include "Drama.h"

DramaFactory::DramaFactory() {
  // identifer: D as Drama
  Inventory::registerInventory("D", this);
}

Inventory *DramaFactory::create() { return new Drama(); }

namespace {
// call constructor to register dramaFactory to the static factories in base
// class
DramaFactory dramaFactory;
} // namespace
//===========================================================

Drama::Drama() { type = "D"; }

bool Drama::operator<(Inventory &rhs) {
  auto &drama = dynamic_cast<Drama &>(rhs);
  if (director != drama.director) {
    return director < drama.director;
  }
  return title < drama.title;
}

bool Drama::operator>(Inventory &rhs) {
  auto &drama = dynamic_cast<Drama &>(rhs);
  if (director != drama.director) {
    return director > drama.director;
  }
  return title > drama.title;
}

bool Drama::operator==(Inventory &rhs) {
  auto &drama = dynamic_cast<Drama &>(rhs);
  return title == drama.title && director == drama.director;
}

istream &Drama::read(istream &is) {
  // director, title, year
  director = util::readNextItem(is, ',');
  title = util::readNextItem(is, ',');
  is >> yearRelease;
  util::eatGarbage(is, '\n');
  return is;
}

istream &Drama::transactionRead(istream &is) {
  // director, title,\n
  director = util::readNextItem(is, ',');
  title = util::readNextItem(is, ',');
  util::eatGarbage(is, '\n');
  return is;
}

ostream &Drama::printer(ostream &os) const {
  os << "Drama: " << title << ", " << director << ", " << yearRelease << ", ";
  for (auto &pair : stocks) {
    os << *pair.second << ", " << endl;
  }
  return os;
}

ostream &Drama::transactionPrinter(ostream &os) const {
  os << director << ", " << title << endl;
  return os;
}
