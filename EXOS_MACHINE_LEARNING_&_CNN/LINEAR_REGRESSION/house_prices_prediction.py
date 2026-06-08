import pandas as pd #Pandas library is used for creating Dataframe
import numpy as np#NumPy library is used for mathematical computation
from sklearn.metrics import r2_score #Sklearn is used for compute score prediction

def create_dataframe():#Import file.csv and create DataFrame
    train = pd.read_csv("./house-prices-advanced-regression-techniques/train.csv") #Train.csv
    final_test = pd.read_csv("./house-prices-advanced-regression-techniques/test.csv") #Final_Test.csv

    dataframe = {
        "train" : train,
        "final_test" : final_test
    }

    return (dataframe)


def replace_variable_to_numeric(dataframe): #Classifying features existing into three categories and replacing string values by numerical values.

    binary_variable = ['Street', 'Alley', 'CentralAir'] #Binary category ==> 0/1 values

    variables_ordinales = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
    'LotShape', 'LandSlope', 'GarageFinish', 'BsmtFinType1', 'BsmtFinType2',
    'BsmtExposure', 'LandContour', 'Utilities', 'PavedDrive'
    ] #Ordinale category ==> values [0, 10] according to comments: excellent, low, etc

    variables_nominales = [
    'MSSubClass', 'MSZoning', 'BldgType', 'HouseStyle', 'RoofStyle', 'GarageType',
    'Neighborhood', 'Condition1', 'Condition2', 'RoofMatl', 'Exterior1st', 
    'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 
    'Functional', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'LotConfig'
    ] #Nominale category. This category inform us about the position, the stat, the shape, the design, etc of a house.
    # Here for each feature, 1 value is for true and can confirm that a house is an "Agriculture" Zoning classification for example while 0 value is for false.

    notes_qualite = {
    'NA': 0, 'IR3': 0, 'Sev': 0, 'NA': 0, 'N': 0,'ELO': 0, 'Low': 0,
    'Po': 1, 'P': 1, 'No': 1, 'Unf': 1, 'IR2': 1, 'Mod': 1, 'HLS': 1,
    'Unf': 1, 'NoSeWa': 1, 'Y': 2, 'Mn': 2, 'LwQ': 2, 'IR1': 2, 'Bnk': 2,
    'Gtl': 2, 'RFn': 2, 'NoSewr': 2, 'Fa': 2, 'Av': 3, 'Rec': 3, 'Lvl': 3,
    'Reg': 3, 'Fin': 3, 'AllPub': 3, 'TA': 3, 'Gd': 4, 'BLQ': 4, 'Ex': 5,
    'ALQ': 5, 'GLQ': 6,
    } #There is a dictionary composed of different values

    notes_street = {'NA': 0, 'Grvl': 0, 'Pave': 1} #This is a binary dictionary
    notes_alley = {'NA': 0, 'Grvl': 1, 'Pave': 1} #This is a binary dictionary
    notes_central_air = {'NA':0, 'N': 0, 'Y': 1} #This is a binary dictionary

    dataframe['Street'] = dataframe['Street'].fillna('NA').map(notes_street) # Replace Street values by 0/1
    dataframe['Alley'] = dataframe['Alley'].fillna('NA').map(notes_alley) # Replace Alley values by 0/1
    dataframe['CentralAir'] = dataframe['CentralAir'].fillna('NA').map(notes_central_air) # Replace CentralAir values by 0/1

    for col in variables_ordinales:
        dataframe[col] = dataframe[col].fillna('NA').map(notes_qualite) #Replace variable ordinales values by their values in notes_qualite

    dataframe = pd.get_dummies(dataframe, columns=variables_nominales, drop_first=True, dtype=int) #Apply One-Hot Encoding method to make correspond 0/1 to existance of one house characteristics

    return (dataframe)


def clean_dataframe(train, final_test):#Clean NaN values from dataframes
    train_clean = train.dropna()
    final_test_clean = final_test.dropna()

    dataframe_cleaned = {
        "train_clean" : train_clean,
        "final_test_clean" : final_test_clean
    }

    return (dataframe_cleaned)


def correlation_with_saleprice(dataframe):#Find best features which correlation are strongest with SalePrice
    big_corr = dataframe.corr()['SalePrice'].abs().sort_values(ascending=False)

    best_features = big_corr[big_corr >= 0.3]

    return (best_features)

def conserv_best_features(dataframe, best_features):#Conserv best features detected

    val_bool = False

    for col in dataframe:
        for feature, value in best_features.items():
            if (col == feature):
                val_bool = True
                break
            
        if (val_bool == False):
            dataframe = dataframe.drop(columns=col)

        val_bool = False

    return(dataframe)

def split_dataframe_into_train_test(dataframe):#Split train dataframe into train and first_test
    dataframe_size = int(len(dataframe) * 0.8)

    train_set = dataframe.iloc[:dataframe_size, :]
    test_set = dataframe.iloc[dataframe_size:, :]

    data_set = {
        "train_set": train_set,
        "test_set": test_set
    }

    return (data_set)


def split_dataframe(dataframe): #Split dataframe into X and y

    index = 0

    for col in dataframe:
        if (col == "SalePrice"):
            y = dataframe.iloc[:, index:index + 1]
            break
        index += 1
    
    X = dataframe.drop(columns="SalePrice")

    data = {
        "X": X,
        "y" : y
    }

    return (data)


def normalize_data_train(X_train):#Normalize X_train values to have X values between [-1, 1]
    moy = X_train.mean(axis=0)
    stand_d = X_train.std(axis=0)

    X_norm = (X_train - moy) / stand_d

    dict_norm = {
        "X_norm": X_norm,
        "moy": moy,
        "std": stand_d
    }

    return (dict_norm)


def normalize_data_test(X_test, moy, stand_d): #Normalize X_test values to have X values between [-1, 1]
    X_test_norm = (X_test - moy) / stand_d

    return (X_test_norm)

def train_model(X_norm, y_train, w, b, learning_rate):#Training my AI model

    y_pred = 0
    error = 0
    loss = 0

    for i in range(10000):
        #Model for prediction
        y_pred = np.dot(X_norm, w) + b

        #Error
        error = y_pred - y_train

        #Loss
        loss = np.mean(error ** 2)

        #Gradients
        dw = (2 * np.dot(X_norm.T, error)) / len(X_norm)
        db = np.mean(2 * error)

        #Gradients Descente
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if (i % 100 == 0):
            print(f"Loss: \n{loss}")

    params = {
        "weights": w,
        "bias": b,
        "train_pred": y_pred
    }

    return (params)


def prediction(X_test, w, b):#Prediction function
    real_pred = np.dot(X_test, w) + b

    return (real_pred)

if __name__=="__main__": #Main function
    df = create_dataframe() #Create and stock dataframe

    train = df['train']
    final_test = df['final_test']

    train = replace_variable_to_numeric(train) #Classifying features existing into three categories and replacing string values by numerical values.
    final_test = replace_variable_to_numeric(final_test) #Classifying features existing into three categories and replacing string values by numerical values.

    df_cleaned = clean_dataframe(train, final_test) #Clean NaN values from dataframes

    train_clean = df_cleaned['train_clean'] #shape(1121, 40)
    final_test_clean = df_cleaned['final_test_clean'] #shape(1146, 39)

    pd.set_option('display.max_columns', None)
    #print(f"Train Dataframe: \n{train_clean}\n") 
    #print(f"final_test Dataframe: \n{final_test_clean.shape}\n")

    best_features_train = correlation_with_saleprice(train_clean) #Find best features which correlation are strongest with SalePrice
    pd.set_option('display.max_rows', None)
    #print(f"Correlation: \n{best_features_train}\n")
    
    train_clean = conserv_best_features(train_clean, best_features_train) #Conserv best features detected
    
    best_features_final_test_clean = conserv_best_features(final_test_clean, best_features_train) #shape(1146, 39)
    #print(f"Correlation Final test: \n{best_features_final_test_clean}\n")

    data_set = split_dataframe_into_train_test(train_clean) #Split train dataframe into train and first_test

    train_set = data_set['train_set']
    test_set = data_set['test_set']

    data = split_dataframe(train_set)
    X_train = data['X'] #shape(896, 39)
    y_train = data['y'] #shape(896, 1)

    data = split_dataframe(test_set)
    X_test = data['X'] #shape(225, 39)
    y_test = data['y'] #shape(225, 1)


    dict_norm = normalize_data_train(X_train) #Normalize X_train values to have X values between [-1, 1]
    X_norm = dict_norm['X_norm'] #shape(896, 39)
    moy = dict_norm['moy']
    stand_d = dict_norm['std']

    # print(f"Best_Features_Train: \n{train_clean.shape}\n")

    # print(f"Best_Features_Final_Test: \n{final_test_clean.shape}\n")

    # print(f"X_Train Dataframe: \n{X_train.shape}\n")
    # print(f"Y_Train Dataframe: \n{y_train.shape}\n")

    # print(f"X_Train_norm Dataframe: \n{X_norm.shape}\n")

    # print(f"X_Test Dataframe: \n{X_test.shape}\n")
    # print(f"Y_Test Dataframe: \n{y_test.shape}\n")

    w = np.zeros((39, 1)) #shape(39, 1)
    b = 0 #shape(1,)

    params = train_model(X_norm, y_train, w, b, learning_rate=0.001) #Training my AI model

    w = params['weights']#shape(39, 1)
    b = params['bias']#shape(1,)
    y_pred = params['train_pred'] #shape(896, 1)

    print(f"Weights: \n{w.shape}\n{w}\n")
    print(f"bias: \n{b.shape}\n{b}\n")
    print(f"Train prediction: \n{y_pred.shape}\n{y_pred}\n")

    score_train = r2_score(y_train, y_pred)
    print(f"The accuracy train is: {score_train}\n")
    print(f"score >= 0.8 ==> excellent prediction!\n")
    print(f"score >= 0.5 ==> good prediction!\n")
    print(f"score == 0 ==> bad prediction!\n")
    print(f"score < 0 ==> very bad prediction!\n")

    X_test_norm = normalize_data_test(X_test, moy, stand_d) #Normalize X_test values to have X values between [-1, 1]
    real_pred = prediction(X_test_norm, w, b) #Prediction function
    score_test = r2_score(y_test, real_pred)
    print(f"The accuracy test is: {score_test}\n")

    id_col = final_test_clean.iloc[:, :1] #id_col.shape = shape(1146, 1)
    #id_col = pd.DataFrame(id_col).to_numpy() #new id_col_shape = shape(1146, )
    final_test_norm = normalize_data_test(best_features_final_test_clean, moy, stand_d)

    final_pred = prediction(final_test_norm, w, b) #final_pred.shape = shape(1146, 1)
    #final_pred = pd.DataFrame(final_pred).to_numpy() #new final_pred_shape = shape(1146, )
    print(f"{id_col.shape}\n")
    pd.set_option('display.max_rows', None)
    print(f"Final prediction values: \n{final_pred.shape}\n")

    sample_submission = pd.DataFrame({
        "id": id_col.to_numpy().flatten(),
        "SalePrice": final_pred.flatten()
    })
    sample_submission.to_csv("sample_submission.csv", index=False)
