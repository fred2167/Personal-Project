#include "Person.h"

ostream &operator<<(ostream &os, const Person &p) { return p.printer(os); }

istream &operator>>(istream &is, Person &p) { return p.read(is); }

map<string, PersonFactory *> &Person::getPersonFactories() {
  /* this method ensure personFactories is created before main
  and in time for resgester*/
  static map<string, PersonFactory *> personFactories;
  return personFactories;
}

int Person::getId() { return id; }
string Person::getName() { return name; }

bool Person::operator<(Person &rhs) { return id < rhs.id; }

bool Person::operator>(Person &rhs) { return id > rhs.id; }

bool Person::operator==(Person &rhs) { return id == rhs.id; }

void Person::registerPersonFactory(const string &type, PersonFactory *factory) {
  getPersonFactories().emplace(type, factory);
}

bool Person::hasPersonFactory(const string &type) {
  return getPersonFactories().count(type) != 0;
}

Person *Person::createPerson(const string &type) {
  if (!hasPersonFactory(type)) {
    cout << "Don't know how to create " << type << "as an Person." << endl;
    return nullptr;
  }
  return getPersonFactories().at(type)->create();
}