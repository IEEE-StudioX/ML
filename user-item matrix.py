# User-Item Matrix:
# A user-item matrix is a matrix used in recommendation systems to represent the relationship between 
# users and items (e.g., movies, products, etc.). 
# Each row represents a user, and each column represents an item. 
# The value at the intersection of a row and column represents the interaction
# or rating of a user for an item.

          | Item 1 | Item 2 | Item 3 | Item 4 | Item 5 |
-------------------------------------------------------
User 1    |   5    |   3    |   0    |   1    |   0    |
User 2    |   4    |   0    |   2    |   5    |   3    |
User 3    |   0    |   2    |   4    |   0    |   1    |
User 4    |   3    |   0    |   1    |   4    |   0    |
-------------------------------------------------------

# Rows: Represent different users (User 1, User 2, etc.).
# Columns: Represent different items (Item 1, Item 2, etc.).
# Values: Represent the interaction between a user and an item, such as:
#   1- Ratings (e.g., 1-5 stars)
#   2- Binary interactions (e.g., 1 for purchased or liked, 0 for not interacted)