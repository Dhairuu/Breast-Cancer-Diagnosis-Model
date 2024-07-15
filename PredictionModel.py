from LogisticRegression import *

with open('BreastCancer.json') as file:
    data = JS.load(file)
    Parameters = data[0]

count = 0
Wrong = 0
while count < 569:
    if(Result(Parameters,TestInput[count]) > 0.55 and Data.iloc[count].diagnosis != 'M'):
        Wrong +=1
    elif(Result(Parameters,TestInput[count]) < 0.55 and Data.iloc[count].diagnosis != 'B'):
        Wrong += 1
    count +=1

print(f"Totoal: 569\nIncorrect: {Wrong}\nError:{100 * (Wrong/569)}")


# while True:
#     index = int(input('Enter the index: '))
#     if index == -1:

#         break
#     print(f"ID: {Data.iloc[index].id}\nDiagnosis: {Data.iloc[index].diagnosis}\nPrediction: {Result(Parameters,TestInput[index])}")