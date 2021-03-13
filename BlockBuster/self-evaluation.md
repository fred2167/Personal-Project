# Self-Evaluation

## Name(s): Fred Chan

Out of 25 points. Use output.txt created using 
`./create-output.sh > output.txt 2>&1` for guidance.

Complete all questions with "Q:"

Q: Does the program compile and run to completion: Yes

- If the program does not compile or gives a segmentation error when run, 
the maximum possible grade is 50%. No need to continue with the rest of self-evaluation

Q: All public functions have been implemented: 0

- -2 for each functionality not implemented

For each command, state Full, Partial or None to indicate 
if it has been fully, partially or not implemented at all.
Explain any partial implementations.

Inventory: Full
History: Full
Borrow: Full
Return: Full


Q: -1 for each compilation warning, min -3: 0

- Check under *1. Compiles without warnings*
- If the warning message is addressed in README.md, including how the programmer tried to address it, no deductions

Q: -1 for each clang-tidy warning, min -3: 0

- Check under *3. clang-tidy warnings*
- If the warning message is addressed in README.md, including how the programmer tried to address it, no deductions

Factories are declared in some cpp files and have raised warning of possible throwing exception and throwing constructor. 

I try to use try catch block but the warning remains. Therefore, I remove try catch block for the readability of the code.

Q: -1 for each clang-format warning, min -3: 0

- Check under *4. clang-format does not find any formatting issues*


Q: -2 for any detected memory leak: 0

- Check under *5. No memory leaks using g++*
- Check under *6. No memory leaks using valgrind*

Q: Do the tests sufficiently test the code: 0

- -1 for each large block of code not executed
- -2 for each function that is never called when testing
- Check under *7. Tests have full code coverage* paying attention to *The lines below were never executed*

-  Person.cpp, Classics.cpp, Comedy.cpp, Drama.cpp
    operator <,>, == are implemented for class completeness.  
- Customer.cpp 
    printer() is implemented for class completeness.
- Person.cpp
    getName() is implemented for class completeness.

Q: Are all functions in .h and .cpp file documents (min -3): 0

- -1 for each function not documented

## Location of error message or functionality

State the file and function where the information can be found

invalid command code: 
    Transaction.cpp
        createTransaction(string& type)

invalid movie type: 
    Inventory.cpp
        createInventory(string& type)

invalid customer ID: 
    Borrow.cpp 
        process(Store& s)
    Return.cpp 
        process(Store& s)
 
invalid movie: 
    Borrow.cpp 
        process(Store& s)
    Return.cpp 
        process(Store& s)

factory classes: 
    Inventory.cpp
    as a self-register static map in Inventory.cpp

hashtable: (explain what the hashtable is used for)
    Store.cpp
    Used to store customers information after reading customer data from file

container used for comedy movies: 
    All movie containers are static set and stored in Inventory.cpp.
    Same self-register feature just as factories.

function for sorting comedy movies: 
    Comedy.cpp:
        operator<, >, ==
    Many comparing operator are implemented for class completeness but not called in test.

function where comedy movies are sorted: 
    Inventory.cpp: 
        bool compare(Inventory* a, Inventory* b)

    Inventories such as comedy, drama, classics are all stored in each set in the inventory. compare are called when register the set in inventory.

functions called when retrieving a comedy movie based on title and year:
    Comedy.cpp:
        operator<
    I create a dummy comedy obj while reading transaction. During retrieving, transaction has a process function that call set.find(Comedy& dummyComedy) to find inventory. Since STL sets use !(a < b) && !(b < a) to determine equality, operator< got called.

functions called for retrieving and printing customer history: 
    Customer.cpp
        displayCustomerHistory()

container used for customer history: 
    Customer.h
        Vector< Transaction* > histories

functions called when borrowing a movie: 
    MediaType.cpp
        borrowItem()

explain borrowing a movie that does not exist: 
    Borrow.cpp
        Process(Store& s)
    1. retrieve customer
    2. retrieve Inventory
        If inventory do not exist, print error message. 
        Set TransactionValidFlag to false and return false;
    3. Store will delete transactions that return false during transaction processing.
    

explain borrowing a movie that has 0 stock: 
   Borrow.cpp
        Process(Store& s)
    1. retrieve Customer
    2. retrieve Inventory
    3. retrieve MediaType in inventory
        If MediaType has stock of 0, print error message.
        Set TransactionValidFlag to false and return false;
    3. Store will delete transactions that return false during   transaction processing.

explain returning a movie that customer has not checked out: 
    Return.cpp
        Process(Store& s)
    Customer.cpp
        hasBorrow(Inventory* inv)

    1. retrieve Customer
    2. check customer has borrow item by call hasBorrow() in customer.cpp
    3. hasBorrow() iterate through vector<Inventory> onLoan to find a match

any static_cast or dynamic_cast used: 
    Return.cpp, Borrow.cpp
        Process(Store& s)
        Used dynamic_cast to cast Person to Customer when retrieve customer.
    
    Classics.cpp, Drama.cpp, Comedy.cpp
        operator<, >, ==


Q: Total points: 25 (max 25)