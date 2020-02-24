#function to generate a feature set for classification of name into gender 'male' or 'female'
#function returns a dictionary as feature set where 'last_letter' is the feature name and 'name[-1]' gives feature value
def gender_featureset(name):
    return {'last_letter':name[-1]}

#creating list of examples and corresponding class labels
from nltk.corpus import names
labeled_names = ([(name,'male') for name in names.words('male.txt')]+[(name,'female') for name in names.words('female.txt')])

#mixing the examples created for labels 'male' and 'female'
import random
random.shuffle(labeled_names)
#to print first 10 elements of 'labeled names'
print("---Labeled Names---")
print(labeled_names[0:10])

print("\n")

#processing examples as per the featureset
features = [(gender_featureset(name),gender) for (name,gender) in labeled_names]
#to print first 10 elements of 'features'
print("---Processed Names---")
print(features[0:10])

#preparing training and testing datasets
#training dataset - contains processed examples from 500 to end
#teting dataset - contains processed examples from 0 to 500
train_set, test_set = features[500:], features[:500]

#creating and training the classifier
import nltk
classifier = nltk.NaiveBayesClassifier.train(train_set)

print("\n")

#testing and measuring accuracy of the classifier
print("---Accuracy---")
print(nltk.classify.accuracy(classifier,test_set))

#classifying unknown names
name = input("Enter name to identify gender: ")
print(name,"is",classifier.classify(gender_featureset(name)))

