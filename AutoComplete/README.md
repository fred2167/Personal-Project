# autocomplete

Use a Binary Search Tree to store key-value pairs where the values represent the frequency of keys (cities, words, movie names, etc)

Traverse the BST to find all entries that match the given prefix. Sort them by weight

### cpp files

- bstmap.h/cpp: Binary Search Tree header and implementation
- bstmapprinter.cpp: Functions for printing BST on screen
- autocomple.h/cpp: Autocomplete
- testbst.cpp: Testing BST functions
- testac.cpp: Testing Autocomplete functions
- main.cpp: Accepts inputs to return strings sorted by their frequency

### Text files for testing
- small.txt Short list of words
- wiktionary.txt Large list of words
- cities.txt Cities

### Shell scripts

- runit.sh: compile and run he program
- create-output.sh: compile, run, check for clang-tidy, clang-format and valgrind warnings
- check-code-coverage: examine if any lines not executed during tests


### Explaination
- None of the function from bstmapprinter.cpp and operator<< are ran during tests. Therefore, there is a large block in check code coverage.