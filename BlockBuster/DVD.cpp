#include "DVD.h"

DVDFactory::DVDFactory() { MediaType::registerMediaTypeFactory("D", this); }

MediaType *DVDFactory::create() const { return new DVD(); }

namespace {
// call constructor to register dvdFactory to the static factories in base class
DVDFactory dvdFactory;
} // namespace
//======================================================================
istream &DVD::read(istream &is) {
  is >> stock;
  return is;
}

ostream &DVD::printer(ostream &os) const {
  os << "DVD " << stock << "(" << onLoan << ")";
  return os;
}