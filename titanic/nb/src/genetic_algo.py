import sympy
from sympy import simplify, cos, sin, Symbol, Function, tanh, pprint, init_printing, exp
from sympy.functions import Min,Max
import numpy as np
import pandas as pd

A = 0.058823499828577    
B = 0.841127 # 0.885868
C = 0.138462007045746 
D = 0.31830988618379069
E = 2.810815 # 2.675679922103882 
F = 0.63661977236758138
G = 5.428569793701172   
H = 3.1415926535897931
I = 0.592158 #0.623655974864960
J = 4.869778 #  2.770736 # 2.212120056152344
K = 0.063467 # 1.5707963267948966
L = -0.091481 # 0.094339601695538 
M = 0.0821533 
N = 0.720430016517639
O = 0.230145 
P = 9.89287 
Q = 785 
R = 1.07241 
S = 281
T = 734
U = 5.3
V = 67.0
W = 2.484848
X = 8.48635 
Y = 63
Z = 12.6275 
AA = 0.735354 # 0.7
AB = 727
AC = 2.5
AD = 2.6 
AE = 0.3
AF = 3.0
AG = 0.226263 #0.1
AH = 2.0
AI = 12.4148
AJ = 96
AK = 0.130303 # 0.2
AL = 176
AM = 3.2
BIG = [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA,AB,AC,AD,AE,AF,AG,AH,AI,AJ,AK,AL,AM]

# Now may I present: The winning gen function, Inspired by Akshat's notebook:
# https://www.kaggle.com/akshat113/titanic-dataset-analysis-level-2
def GeneticFunction(data,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA,AB,AC,AD,AE,AF,AG,AH,AI,AJ,AK,AL,AM):
    return ((np.minimum( ((((A + data["Sex"]) - np.cos((data["Pclass"] / AH))) * AH)),  ((B))) * AH) +
            np.maximum( ((data["SibSp"] - AC)),  ( -(np.minimum( (data["Sex"]),  (np.sin(data["Parch"]))) * data["Pclass"]))) +
            (AG * ((np.minimum( (data["Sex"]),  (((data["Parch"] / AH) / AH))) * data["Age"]) - data["Cabin"])) +
            np.minimum( ((np.sin((data["Parch"] * ((data["Fare"] - AA) * AH))) * AH)),  ((data["SibSp"] / AH))) +
            np.maximum( (np.minimum( ( -np.cos(data["Embarked"])),  (C))),  (np.sin(((data["Cabin"] - data["Fare"]) * AH)))) +
            -np.minimum( ((((data["Age"] * data["Parch"]) * data["Embarked"]) + data["Parch"])),  (np.sin(data["Pclass"]))) +
            np.minimum( (data["Sex"]),  ((np.sin( -(data["Fare"] * np.cos((data["Fare"] * W)))) / AH))) +
            np.minimum( ((O)),  (np.sin(np.minimum( (((V / AH) * np.sin(data["Fare"]))),  (D))))) +
            np.sin((np.sin(data["Cabin"]) * (np.sin((Z)) * np.maximum( (data["Age"]),  (data["Fare"]))))) +
            np.sin(((np.minimum( (data["Fare"]),  ((data["Cabin"] * data["Embarked"]))) / AH) *  -data["Fare"])) +
            np.minimum( (((AD * data["SibSp"]) * np.sin(((AJ) * np.sin(data["Cabin"]))))),  (data["Parch"])) +
            np.sin(np.sin((np.maximum( (np.minimum( (data["Age"]),  (data["Cabin"]))),  ((data["Fare"] * AK))) * data["Cabin"]))) +
            np.maximum( (np.sin(((AI) * (data["Age"] / AH)))),  (np.sin((-AF * data["Cabin"])))) +
            (np.minimum( (np.sin((((np.sin(((data["Fare"] * AH) * AH)) * AH) * AH) * AH))),  (data["SibSp"])) / AH) +
            ((data["Sex"] - data["SibSp"]) * (np.cos(((data["Embarked"] - AA) + data["Age"])) / AH)) +
            ((np.sin(data["Cabin"]) / AH) - (np.cos(np.minimum( (data["Age"]),  (data["Embarked"]))) * np.sin(data["Embarked"]))) +
            np.minimum( (AE),  ((data["Sex"] * (J * (N - np.sin((data["Age"] * AH))))))) +
            (np.minimum( (np.cos(data["Fare"])),  (np.maximum( (np.sin(data["Age"])),  (data["Parch"])))) * np.cos((data["Fare"] / AH))) +
            np.sin((data["Parch"] * np.minimum( ((data["Age"] - K)),  ((np.cos((data["Pclass"] * AH)) / AH))))) +
            (data["Parch"] * (np.sin(((data["Fare"] * (I * data["Age"])) * AH)) / AH)) +
            (D * np.cos(np.maximum( ((0.5 * data["Fare"])),  ((np.sin(N) * data["Age"]))))) +
            (np.minimum( ((data["SibSp"] / AH)),  (np.sin(((data["Pclass"] - data["Fare"]) * data["SibSp"])))) * data["SibSp"]) +
            np.tanh((data["Sex"] * np.sin((U * np.sin((data["Cabin"] * np.cos(data["Fare"]))))))) +
            (np.minimum( (data["Parch"]),  (data["Sex"])) * np.cos(np.maximum( ((np.cos(data["Parch"]) + data["Age"])),  (AM)))) +
            (np.minimum( (np.tanh(((data["Cabin"] / AH) + data["Parch"]))),  ((data["Sex"] + np.cos(data["Age"])))) / AH) +
            (np.sin((np.sin(data["Sex"]) * (np.sin((data["Age"] * data["Pclass"])) * data["Pclass"]))) / AH) +
            (data["Sex"] * (np.cos(((data["Sex"] + data["Fare"]) * ((X) * (Y)))) / AH)) +
            np.minimum( (data["Sex"]),  ((np.cos((data["Age"] * np.tanh(np.sin(np.cos(data["Fare"]))))) / AH))) +
            (np.tanh(np.tanh( -np.cos((np.maximum( (np.cos(data["Fare"])),  (L)) * data["Age"])))) / AH) +
            (np.tanh(np.cos((np.cos(data["Age"]) + (data["Age"] + np.minimum( (data["Fare"]),  (data["Age"])))))) / AH) +
            (np.tanh(np.cos((data["Age"] * ((-AH + np.sin(data["SibSp"])) + data["Fare"])))) / AH) +
            (np.minimum( (((S) - data["Fare"])),  (np.sin((np.maximum( ((AL)),  (data["Fare"])) * data["SibSp"])))) * AH) +
            np.sin(((np.maximum( (data["Embarked"]),  (data["Age"])) * AH) * (((Q) * H) * data["Age"]))) +
            np.minimum( (data["Sex"]),  (np.sin( -(np.minimum( ((data["Cabin"] / AH)),  (data["SibSp"])) * (data["Fare"] / AH))))) +
            np.sin(np.sin((data["Cabin"] * (data["Embarked"] + (np.tanh( -data["Age"]) + data["Fare"]))))) +
            (np.cos(np.cos(data["Fare"])) * (np.sin((data["Embarked"] - ((T) * data["Fare"]))) / AH)) +
            ((np.minimum( (data["SibSp"]),  (np.cos(data["Fare"]))) * np.cos(data["SibSp"])) * np.sin((data["Age"] / AH))) +
            (np.sin((np.sin((data["SibSp"] * np.cos((data["Fare"] * AH)))) + (data["Cabin"] * AH))) / AH) +
            (((data["Sex"] * data["SibSp"]) * np.sin(np.sin( -(data["Fare"] * data["Cabin"])))) * AH) +
            (np.sin((data["SibSp"] * ((((G + V) * AH) / AH) * data["Age"]))) / AH) +
            (data["Pclass"] * (np.sin(((data["Embarked"] * data["Cabin"]) * (data["Age"] - (R)))) / AH)) +
            (np.cos((((( -data["SibSp"] + data["Age"]) + data["Parch"]) * data["Embarked"]) / AH)) / AH) +
            (D * np.sin(((data["Age"] * ((data["Embarked"] * np.sin(data["Fare"])) * AH)) * AH))) +
            ((np.minimum( ((data["Age"] * A)),  (data["Sex"])) - F) * np.tanh(np.sin(data["Pclass"]))) +
            -np.minimum( ((np.cos(((AB) * ((data["Fare"] + data["Parch"]) * AH))) / AH)),  (data["Fare"])) +
            (np.minimum( (np.cos(data["Fare"])),  (data["SibSp"])) * np.minimum( (np.sin(data["Parch"])),  (np.cos((data["Embarked"] * AH))))) +
            (np.minimum( (((data["Fare"] / AH) - E)),  (C)) * np.sin((K * data["Age"]))) +
            np.minimum( ((M)),  (((np.sin(data["Fare"]) + data["Embarked"]) - np.cos((data["Age"] * (P)))))))


def CleanData(data):
    # Sex
    data.drop(['Ticket', 'Name'], inplace=True, axis=1)
    data.Sex.fillna('0', inplace=True)
    data.loc[data.Sex != 'male', 'Sex'] = 0
    data.loc[data.Sex == 'male', 'Sex'] = 1
    # Cabin
    data.Cabin.fillna('0', inplace=True)
    data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 2
    data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 3
    data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 4
    data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 5
    data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 6
    data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 7
    data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 8
    # Embarked
    data.loc[data.Embarked == 'C', 'Embarked'] = 1
    data.loc[data.Embarked == 'Q', 'Embarked'] = 2
    data.loc[data.Embarked == 'S', 'Embarked'] = 3
    data.Embarked.fillna(0, inplace=True)
    data.fillna(-1, inplace=True)
    return data.astype(float)

# This function rounds values to either 1 or 0, because the GeneticFunction below returns floats and no
# definite values
def Outputs(data):
    return np.round(1.-(1./(1.+np.exp(-data))))


# データ取得
raw_train = pd.read_csv("../../data/input/train.csv") # 学習データ
raw_test = pd.read_csv("../../data/input/test.csv")   # テストデータ

cleanedTrain = CleanData(raw_train)
cleanedTest = CleanData(raw_test)
# run a check on the Training dataset. See section "Programm your own gen. algorithm" below on how to 
# construct your own genetic algorithm
thisArray = BIG.copy()
testPredictions = Outputs(GeneticFunction(cleanedTrain,thisArray[0],thisArray[1],thisArray[2],thisArray[3],thisArray[4],thisArray[5],thisArray[6],thisArray[7],thisArray[8],thisArray[9],thisArray[10],thisArray[11],thisArray[12],thisArray[13],thisArray[14],thisArray[15],thisArray[16],thisArray[17],thisArray[18],thisArray[19],thisArray[20],thisArray[21],thisArray[22],thisArray[23],thisArray[24],thisArray[25],thisArray[26],thisArray[27],thisArray[28],thisArray[29],thisArray[30],thisArray[31],thisArray[32],thisArray[33],thisArray[34],thisArray[35],thisArray[36],thisArray[37],thisArray[38]))
pdcheck = pd.DataFrame({'Survived': testPredictions.astype(int)})
ret = pdcheck.Survived.where(pdcheck["Survived"].values==cleanedTrain["Survived"].values).notna()
t,f = ret.value_counts()
score = 100/(t+f)*t
print("Training set score: ",score)
scores=[]
scores.append(score/100)

testPredictions = Outputs(GeneticFunction(cleanedTest,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA,AB,AC,AD,AE,AF,AG,AH,AI,AJ,AK,AL,AM))
pdtest = pd.DataFrame({'PassengerId': cleanedTest.PassengerId.astype(int),
                        'Survived': testPredictions.astype(int)})
pdtest.to_csv("../../data/output/003/submission_ga.csv", index=False)
pdtest.head()