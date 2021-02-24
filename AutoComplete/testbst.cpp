// /**
//  * Testing BST - Binary Search Tree functions
//  *
//  * This file has series of tests for BST
//  * Each test is independent and uses assert statements
//  * Test functions are of the form
//  *
//  *      test_netidXX()
//  *
//  * where netid is UW netid and XX is the test number starting from 01
//  *
//  * Test functions can only use the public functions from BST
//  * testBSTAll() is called from main in main.cpp
//  * testBSTAll calls all other functions
//  * @author Multiple
//  * @date ongoing
//  */

#include "bstmap.h"
#include <cassert>
#include <sstream>

using namespace std;

// global value for testing
// NOLINTNEXTLINE
stringstream globalSS;

// need to reset SS before calling this
void printer(const BSTMap::value_type &p) {
  globalSS << "[" << p.first << "=" << p.second << "]";
}

// // Testing == and []
void test01() {
  cout << "Starting test01" << endl;
  cout << "* Testing ==, !=, [] and copy constructor" << endl;
  BSTMap b1;
  auto val = b1["hello"];
  assert(val == 0);
  b1["hello"] = 5;
  val = b1["hello"];
  assert(val == 5);
  b1["world"] = 42;

  BSTMap b2;
  assert(b1 != b2);
  b2["hello"] = 5;
  b2["world"] = 42;
  assert(b1 == b2);

  BSTMap b3(b2);
  assert(b1 == b3);
  cout << "Ending tes01" << endl;
}

// Testing traversal
void test02() {
  cout << "Starting test02" << endl;
  cout << "* Testing traversal" << endl;
  BSTMap b;
  b["x"] = 10;
  b["f"] = 5;
  b["b"] = 3;
  b["e"] = 4;
  b["z"] = 50;
  // cout << b;

  globalSS.str("");
  b.inorder(printer);
  string order = globalSS.str();
  assert(order == "[b=3][e=4][f=5][x=10][z=50]");

  globalSS.str("");
  b.preorder(printer);
  order = globalSS.str();
  assert(order == "[x=10][f=5][b=3][e=4][z=50]");

  globalSS.str("");
  b.postorder(printer);
  order = globalSS.str();
  assert(order == "[e=4][b=3][f=5][z=50][x=10]");
  cout << "Ending test02" << endl;
}

// Testing rebalance
void test03() {
  cout << "Starting test03" << endl;
  cout << "* Testing rebalance" << endl;
  BSTMap b;
  b["1"] = 1;
  b["2"] = 2;
  b["3"] = 3;
  b["4"] = 4;
  b["5"] = 5;
  b["6"] = 6;
  assert(b.height() == 6);
  // cout << b << endl;
  b.rebalance();
  assert(b.height() == 3);
  // cout << b << endl;
  b.clear();
  assert(b.height() == 0);
  cout << "Ending test03" << endl;
}
// testing equality
void test04() {
  cout << "Starting test04" << endl;
  cout << "* Testing two trees equality" << endl;
  BSTMap b;
  b["1"] = 1;
  b["2"] = 2;
  b["3"] = 3;
  b["4"] = 4;
  b["5"] = 5;
  b["6"] = 6;

  BSTMap b2;
  b2["1"] = 1;
  b2["2"] = 2;
  b2["3"] = 3;
  b2["4"] = 4;
  b2["5"] = 5;
  b2["6"] = 6;
  assert(b == b2);

  cout << "Ending test04" << endl;
}
void test05() {
  cout << "Starting test05" << endl;
  cout << "* Testing empty, contains, count, size, height" << endl;
  BSTMap b;
  assert(b.empty() == true);
  b["y"] = 12;
  b["x"] = 13;
  b["z"] = 11;

  assert(b.empty() == false);
  assert(b.contains("x") == true);
  assert(b.contains("z") == true);
  assert(b.contains("a") == false);
  assert(b.count("x") == 1);
  assert(b.count("a") == 0);
  assert(b.size() == 3);
  assert(b.height() == 2);

  BSTMap b1;
  assert(b1.contains("a") == false);
  cout << "Ending test05" << endl;
}

void test07() {
  cout << "Starting test07" << endl;
  cout << "* Testing erase" << endl;
  BSTMap b;

  b["3"] = 3;
  b["2"] = 2;
  b["5"] = 5;
  b["1"] = 1;
  b["4"] = 4;
  b["6"] = 6;
  b["7"] = 7;

  // cout << b<< endl;
  bool noelement = b.erase("8");
  bool rightchild = b.erase("6");
  bool bothchild = b.erase("3");
  bool leftchild = b.erase("2");
  bool nochild = b.erase("4");

  // cout << b<< endl;

  assert(noelement == false);
  assert(rightchild == true);
  assert(leftchild == true);
  assert(nochild == true);
  assert(bothchild == true);
  assert(b.size() == 3);
  cout << "Ending test07" << endl;
}

void test08() {
  cout << "Starting test08" << endl;
  cout << "* Testing rebalance" << endl;
  BSTMap b;
  b["0"] = 0;
  b["1"] = 1;
  b["2"] = 2;
  b["3"] = 3;
  b["4"] = 4;
  b["5"] = 5;

  // cout << b << endl;
  b.rebalance();
  // cout << b << endl;
  assert(b.height() == 3);
  cout << "Ending test08" << endl;
}
void test09() {
  cout << "Starting test09" << endl;
  cout << "* Testing vector constructor" << endl;
  vector<pair<string, uint64_t>> v;
  v.reserve(4);
  for (int i = 4; i > 0; i--) {
    v.emplace_back(make_pair(to_string(i), i));
  }
  BSTMap b(v);
  // cout << b << endl;
  assert(b.height() == 3);
  assert(b.size() == 4);
  cout << "Ending test09" << endl;
}

void test10() {
  cout << "Starting test10" << endl;
  cout << "* Testing getAll" << endl;

  BSTMap b;
  b["hello"] = 10;
  b["help"] = 20;
  b["heal"] = 3;
  b["herd"] = 5;
  b["hertz"] = 7;
  b["look"] = 20;
  b["like"] = 20;
  b["lerp"] = 2;
  b["lemon"] = 10;
  b["lease"] = 6;

  b.rebalance();
  // cout << b << endl;
  vector<BSTMap::value_type> v;
  globalSS.str("");

  v = b.getAll("he");
  for (auto &pair : v) {
    globalSS << pair.first << pair.second;
  }
  string order = globalSS.str();
  assert(order == "heal3hello10help20herd5hertz7");

  v = b.getAll("ab");
  assert(v.empty() == true);
  cout << "Ending test10" << endl;
}

// // Calling all test functions
void testBSTAll() {
  test01();
  test02();
  test03();
  test04();
  test05();
  test07();
  test08();
  test09();
  test10();
}

// int main() { testBSTAll(); }