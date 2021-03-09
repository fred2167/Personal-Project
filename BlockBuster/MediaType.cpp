#include "MediaType.h"

ostream &operator<<(ostream &os, const MediaType &m) { return m.printer(os); }

istream &operator>>(istream &is, MediaType &m) { return m.read(is); }

bool MediaType::hasStock() { return stock > 0; }

bool MediaType::borrowItem() {
  if (hasStock()) {
    stock -= 1;
    onLoan += 1;
    return true;
  }
  return false;
}

bool MediaType::returnItem() {
  stock += 1;
  onLoan -= 1;
  return true;
}

void MediaType::setStock(int stock) {
  assert(stock > 0);
  this->stock = stock;
}

int MediaType::getStock() { return stock; }

map<string, MediaTypeFactory *> &MediaType::getMediaTypeFactory() {
  static map<string, MediaTypeFactory *> mediaTypeFactories;
  return mediaTypeFactories;
}

void MediaType::registerMediaTypeFactory(const string &type,
                                         MediaTypeFactory *factory) {
  getMediaTypeFactory().emplace(type, factory);
}

bool MediaType::hasMediaTypeFactory(const string &type) {
  return getMediaTypeFactory().count(type) != 0;
}

MediaType *MediaType::createMediaType(const string &type) {
  if (!hasMediaTypeFactory(type)) {
    cout << "Don't know how to create " << type << " as a MediaType." << endl;
    return nullptr;
  }
  return getMediaTypeFactory().at(type)->create();
}