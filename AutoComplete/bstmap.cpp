// Yusuf Pisan pisan@uw.edu
// 17 Jan 2021

// BST class
// Creates a BST to store values
// Uses Node which holds the data

#include "bstmap.h"
#include <cassert>

using namespace std;

// copy constructor
BSTMap::BSTMap(const BSTMap &bst) {

  // delete self
  clear();
  copy(root, bst.root);
}

void BSTMap::copy(Node *&thisNodePtr, Node *otherNodePtr) {
  if (otherNodePtr != nullptr) {
    Node *insNode = new Node();
    insNode->data =
        make_pair((otherNodePtr->data).first, (otherNodePtr->data).second);
    thisNodePtr = insNode;
    size_ += 1;

    copy(thisNodePtr->left, otherNodePtr->left);
    copy(thisNodePtr->right, otherNodePtr->right);
  }
}

// given an array of length n
// create a tree to have all items in that array
// with the minimum height (uses same helper as rebalance)
BSTMap::BSTMap(const vector<value_type> &v) {

  vector<Node *> nodes;
  nodes.reserve(v.size());
  for (auto &pair : v) {
    Node *node = new Node();
    node->data = make_pair(pair.first, pair.second);
    nodes.push_back(node);
  }

  auto sortByKey = [](Node *&a, Node *&b) {
    return a->data.first < b->data.first;
  };
  sort(nodes.begin(), nodes.end(), sortByKey);
  rebalance(nodes, root, 0, v.size() - 1);
  size_ = v.size();
}

// destructor
BSTMap::~BSTMap() { clear(); }

// delete all nodes in tree
void BSTMap::clear() { clear(root); }

void BSTMap::clear(Node *&nodePtr) {
  if (nodePtr != nullptr) {
    clear(nodePtr->left);
    clear(nodePtr->right);
    delete nodePtr;
    nodePtr = nullptr;
    size_ -= 1;
  }
}

// true if no nodes in BST
bool BSTMap::empty() const { return size_ == 0; }

// Number of nodes in BST
int BSTMap::size() const { return size_; }

// true if item is in BST
bool BSTMap::contains(const key_type &key) const {
  if (empty()) {
    return false;
  }
  Node *current = root;

  while (current != nullptr) {

    if (key > (current->data).first) {
      current = current->right;
    } else if (key < (current->data).first) {
      current = current->left;
    } else {
      return true;
    }
  }

  return false;
}

// If k matches the key returns a reference to its value
// If k does not match any key, inserts a new element
// with that key and returns a reference to its mapped value.
BSTMap::mapped_type &BSTMap::operator[](const key_type &k) {
  Node *current = root;
  Node *previous = nullptr;

  while (current != nullptr) {

    previous = current;

    if (k > (current->data).first) {
      current = current->right;
    } else if (k < (current->data).first) {
      current = current->left;
    } else { // already has  key
      return (current->data).second;
    }
  }

  Node *insNode = new Node();
  insNode->data = make_pair(k, 0);
  size_ += 1;

  // first case (empty tree)
  if (previous == nullptr) {
    root = insNode;
  } else if (k > (previous->data).first) {
    previous->right = insNode;
  } else {
    previous->left = insNode;
  }
  return (insNode->data).second;
}

vector<BSTMap::value_type> BSTMap::getAll(const key_type &k) const {

  Node *current = root;

  // find the key
  while (current != nullptr) {

    if ((current->data).first.rfind(k, 0) != 0 && k > (current->data).first) {
      current = current->right;
    } else if ((current->data).first.rfind(k, 0) != 0 &&
               k < (current->data).first) {
      current = current->left;
    } else if ((current->data).first.rfind(k, 0) == 0) {
      break;
    }
  }

  vector<value_type> v;
  // did not found key, return empty vector to conform with API
  if (current == nullptr) {
    return v;
  }

  vector<Node *> nodes;
  vectorize(nodes, current, k);

  for (auto &node : nodes) {
    v.push_back(node->data);
  }

  return v;
}

// 0 if empty, 1 if only root, otherwise
// height of root is max height of subtrees + 1
int BSTMap::height() const { return getHeight(root); }

// height of a Node, nullptr is 0, root is 1, static, no access to 'this'
// helper function to height(), used by printVertical
int BSTMap::getHeight(const Node *n) {
  if (n == nullptr) {
    return 0;
  }
  return 1 + max(getHeight(n->left), getHeight(n->right));
}

// same as contains, but returns 1 or 0
// compatibility with std::map
size_t BSTMap::count(const string &k) const {
  return static_cast<size_t>(contains(k));
}

// inorder traversal: left-root-right
// takes a function that takes a single parameter of type T
void BSTMap::inorder(void visit(const value_type &item)) const {
  inorder(visit, root);
}

void BSTMap::inorder(void visit(const value_type &item), Node *nodePtr) const {

  if (nodePtr != nullptr) {
    inorder(visit, nodePtr->left);
    visit(nodePtr->data);
    inorder(visit, nodePtr->right);
  }
}

// preorder traversal: root-left-right
void BSTMap::preorder(void visit(const value_type &item)) const {
  preorder(visit, root);
}

void BSTMap::preorder(void visit(const value_type &item), Node *nodePtr) const {

  if (nodePtr != nullptr) {
    visit(nodePtr->data);
    preorder(visit, nodePtr->left);
    preorder(visit, nodePtr->right);
  }
}

// postorder traversal: left-right-root
void BSTMap::postorder(void visit(const value_type &item)) const {
  postorder(visit, root);
}
void BSTMap::postorder(void visit(const value_type &item),
                       Node *nodePtr) const {

  if (nodePtr != nullptr) {
    postorder(visit, nodePtr->left);
    postorder(visit, nodePtr->right);
    visit(nodePtr->data);
  }
}

// in order traversal to push nodes that match the key into a vector
void BSTMap::vectorize(vector<Node *> &v, Node *nodePtr,
                       const key_type &key) const {
  if (nodePtr != nullptr) {
    vectorize(v, nodePtr->left, key);

    // get all nodes if key is empty
    if (key.empty()) {
      v.push_back(nodePtr);
    } else {
      // only get nodes where prefix of keys match
      if ((nodePtr->data).first.rfind(key, 0) == 0) {
        v.push_back(nodePtr);
      }
    }
    vectorize(v, nodePtr->right, key);
  }
}

// balance the BST by saving all nodes to a vector inorder
// and then recreating the BST from the vector
void BSTMap::rebalance() {

  vector<Node *> v;
  key_type s; // empty string to match API/ No use
  vectorize(v, root, s);
  rebalance(v, root, 0, v.size() - 1);
}

void BSTMap::rebalance(vector<Node *> &v, Node *&nodePtr, int startIdx,
                       int endIdx) {
  if (startIdx >= 0 && startIdx < v.size() && endIdx >= 0 &&
      endIdx < v.size() && startIdx <= endIdx) {
    int mid = (startIdx + endIdx + 1) / 2;
    nodePtr = v[mid];
    rebalance(v, nodePtr->left, startIdx, mid - 1);
    rebalance(v, nodePtr->right, mid + 1, endIdx);
  } else {
    // since we are rearrange nodes, nodes still have dangle links.
    // if the condition not met, its a dangle link, set to nullptr
    nodePtr = nullptr;
  }
}

// trees are equal if they have the same structure
// AND the same item values at all the nodes
bool BSTMap::operator==(const BSTMap &other) const {

  if (size() != other.size()) {
    return false;
  }
  return isEqual(root, other.root);
}

// check equality in each node. return *right away* when data are not the same
bool BSTMap::isEqual(Node *thisNodePtr, Node *otherNodePtr) const {
  if (thisNodePtr != nullptr && otherNodePtr != nullptr) {

    bool flag = (thisNodePtr->data == otherNodePtr->data);
    if (flag) {
      isEqual(thisNodePtr->left, otherNodePtr->left);
      isEqual(thisNodePtr->right, otherNodePtr->right);
    }
    return flag;
  }

  // case 1: both are nullptr. return true since they are equivalent in  BST
  // case 2: One is nullptr one is not. return false where two tree are NOT
  // equivalent
  return thisNodePtr == otherNodePtr;
}

// not == to each other
bool BSTMap::operator!=(const BSTMap &other) const {
  return !operator==(other);
}

bool BSTMap::erase(const key_type &k) {
  int sizeBefore = size();
  root = erase(k, root);
  return sizeBefore != size();
}
BSTMap::Node *BSTMap::erase(const key_type &k, Node *nodePtr) {

  // fall of tree
  if (nodePtr == nullptr) {
    return nullptr;
  }

  // go right sub-tree
  if (k > (nodePtr->data).first) {
    nodePtr->right = erase(k, nodePtr->right);
  }
  // go left sub-tree
  else if (k < (nodePtr->data).first) {
    nodePtr->left = erase(k, nodePtr->left);
  } else {
    // no child
    if (nodePtr->left == nullptr && nodePtr->right == nullptr) {
      delete nodePtr;
      size_ -= 1;
      return nullptr;
    }

    // one left child
    if (nodePtr->left != nullptr && nodePtr->right == nullptr) {
      Node *leftChild = nodePtr->left;

      // delete self and decrement size
      delete nodePtr;
      size_ -= 1;

      // promote left child
      return leftChild;
    }

    // one right child
    if (nodePtr->left == nullptr && nodePtr->right != nullptr) {
      Node *rightChild = nodePtr->right;

      // delete self and decrement size
      delete nodePtr;
      size_ -= 1;

      // promote right child
      return rightChild;
    }

    // has both child
    Node *successor = nodePtr->right;
    while (successor->left != nullptr) {
      successor = successor->left;
    }
    // copy data
    nodePtr->data =
        make_pair((successor->data).first, (successor->data).second);

    // delete successor node
    nodePtr->right = erase((successor->data).first, nodePtr->right);
    return nodePtr;
  }

  return nodePtr;
}