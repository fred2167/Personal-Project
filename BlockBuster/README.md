## Moive Project - Polymorphism
3-14-2021 Final Project

## Requirements

- Create a movie store that read-in three .txt files that create inventories(Movies), customers, and transactions.

- Process each transaction with different customers and inventories.

- Create abstract factories and each sub-class need to self-regester its own. Adding new classes mustn't alter original code.


## Implement

- Factories are stored as a static map with different single character identifier in Inventory.cpp as singleton. 

- Inventories(Movies) are also stored the same as factories with self-registering feature.

- Inventory 
  - Movie
    - Comedy
    - Classical
    - Drama

- MediaType -> DVD

- Person -> Customer

- Transaction 
  - Return
  - Borrow
  - History
  - DisplayInventory
