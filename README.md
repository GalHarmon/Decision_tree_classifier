# Decision_tree_classifier
Full implementation of a decision tree in Python using numpy and pandas

Also entropy and χ2 tests functions implemented by myself

The tree in the project tries to predict if a certain hour of a certain day is going to be busy or NOT busy in Seoul Bike rental.
<br>
<b>Busy hour</b> = over 650 rented bikes.
<br>
<b>NOT busy hour</b> = less or equal to 650

Main functions:
1. "build tree(<float> ratio)" -> build a decision tree, using ratio ratio of the data, and validate it on the remainder.
2. "tree error(<int> k)" -> report the quality of the decision tree by building k-fold cross validation
3. "is busy(<array> row input)" ->  return 1 if you think the day will be a busy one (more than 650 rentals), and 0 if not.




Assignment instructions can be found in requirementsFile.pdf.
<br>
The raw date set can be found in SeoulBikeData.csv.
