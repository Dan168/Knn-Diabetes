import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# import the data using pandas
df = pd.read_csv('data.csv')

# Split the data into X and Y
# Dropping the diabetes column as that is the label
# x is now split into just the features, y now only contains the Labels
x = df.drop(columns=['diabetes'])
y = df['diabetes'].values

# Splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

# Set the classifier up
# The classifier determines how many neighbours to look at on the model
# Changing this number will change the overall accuracy of the model
knn = KNeighborsClassifier(n_neighbors=14)
# Training the model
knn.fit(x_train, y_train)

# user info can be inputted here
# pregnancies,glucose,diastolic,triceps,insulin,bmi,dpf,age,diabetes
arrayToPredict = [[0,128,75,50,35,29,1.101,34]]

if knn.predict(arrayToPredict) == 0:
    print("There is low chance you're diabetic")
else:
    print("There is an increased chance you're diabetic")

print("Overall model accuracy score:",knn.score(x_test, y_test))

