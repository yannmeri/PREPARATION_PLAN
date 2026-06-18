import os #connect my local code to my docker-container
import pandas as pd #Pandas library is used for creating Dataframe
import numpy as np #NumPy library is used for mathematical computation
from sklearn.metrics import r2_score #Sklearn is used for compute score prediction
from tqdm import tqdm #loading bar


mode = os.getenv("MODE_PROJECT")

BINARY_VARIABLES = ['Street', 'Alley', 'CentralAir'] #Binary category ==> 0/1 values

ORDINALES_VARIABLES = [
'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
'LotShape', 'LandSlope', 'GarageFinish', 'BsmtFinType1', 'BsmtFinType2',
'BsmtExposure', 'LandContour', 'Utilities', 'PavedDrive'
] #Ordinale category ==> values [0, 10] according to comments: excellent, low, etc

NOMINALES_VARIABLES = [
'MSSubClass', 'MSZoning', 'BldgType', 'HouseStyle', 'RoofStyle', 'GarageType',
'Neighborhood', 'Condition1', 'Condition2', 'RoofMatl', 'Exterior1st', 
'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 
'Functional', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'LotConfig'
] #Nominale category. This category inform us about the position, the stat, the shape, the design, etc of a house.
# Here for each feature, 1 value is for true and can confirm that a house is an "Agriculture" Zoning classification for example while 0 value is for false.

QUALITY_NOTES = {
'NA': 0, 'IR3': 0, 'Sev': 0, 'NA': 0, 'N': 0,'ELO': 0, 'Low': 0,
'Po': 1, 'P': 1, 'No': 1, 'Unf': 1, 'IR2': 1, 'Mod': 1, 'HLS': 1,
'Unf': 1, 'NoSeWa': 1, 'Y': 2, 'Mn': 2, 'LwQ': 2, 'IR1': 2, 'Bnk': 2,
'Gtl': 2, 'RFn': 2, 'NoSewr': 2, 'Fa': 2, 'Av': 3, 'Rec': 3, 'Lvl': 3,
'Reg': 3, 'Fin': 3, 'AllPub': 3, 'TA': 3, 'Gd': 4, 'BLQ': 4, 'Ex': 5,
'ALQ': 5, 'GLQ': 6,
} #There is a dictionary composed of different values

NOTES_STREET = {'NA': 0, 'Grvl': 0, 'Pave': 1} #This is a binary dictionary
NOTES_ALLEY = {'NA': 0, 'Grvl': 1, 'Pave': 1} #This is a binary dictionary
NOTES_CENTRAL_AIR = {'NA':0, 'N': 0, 'Y': 1} #This is a binary dictionary

def load_datasets(): #Load the raw train and test CSV files.

    df_train = pd.read_csv("./house-prices-advanced-regression-techniques/train.csv") #Train.csv
    df_test = pd.read_csv("./house-prices-advanced-regression-techniques/test.csv") #Final_Test.csv

    return (df_train, df_test)


def encode_categorical_features(df_train, df_test): #Clean, map and encode categorical variables using a combined pipeline.

    # Add tags to track splits and align target columns
    df_train["is_train"] = 1
    df_test["is_train"] = 0
    df_test["SalePrice"] = 0

    # Temporal concatenation to sync encoding shapes
    combined = pd.concat([df_train, df_test], axis=0)

    # 1. Binary conversion
    combined['Street'] = combined['Street'].fillna('NA').map(NOTES_STREET) # Replace Street values by 0/1
    combined['Alley'] = combined['Alley'].fillna('NA').map(NOTES_ALLEY) # Replace Alley values by 0/1
    combined['CentralAir'] = combined['CentralAir'].fillna('NA').map(NOTES_CENTRAL_AIR) # Replace CentralAir values by 0/1

    # 2. Ordinal mapping
    for col in ORDINALES_VARIABLES:
        combined[col] = combined[col].fillna('NA').map(QUALITY_NOTES) #Replace variable ordinales values by their values in QUALITY_NOTES

    # 3. One-Hot encoding for nominal features
    combined = pd.get_dummies(combined, columns=NOMINALES_VARIABLES, drop_first=True, dtype=int) #Apply One-Hot Encoding method to make correspond 0/1 to existance of one house characteristics

    # 4. Handle remaining NaN values with numerical column median
    numerical_cols = combined.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != "SalePrice":
            combined[col] = combined[col].fillna(combined[col].median())

    # Split back into individual preprocessed sets
    train_treated = combined[combined["is_train"] == 1].drop(columns=["is_train"])
    test_treated = combined[combined["is_train"] == 0].drop(columns=["is_train", "SalePrice"])

    return (train_treated, test_treated)


def clean_dataframe(train_treated, test_treated):#Clean NaN values from dataframes
    
    train_clean = train_treated.dropna()
    test_clean = test_treated.dropna()

    return (train_clean, test_clean)


def compute_top_correlations(dataframe):#Filter and return features with a strong correlation to SalePrice.
    
    big_corr = dataframe.corr()['SalePrice'].abs().sort_values(ascending=False)
    best_features = big_corr[big_corr >= 0.3]

    return (best_features)


def drop_low_correlation_features(dataframe, best_features):#Drop any column from the dataframe that is not listed in top_features.

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


def train_val_split(dataframe):#Split the dataset into internal training and validation sets.
    
    dataframe_size = int(len(dataframe) * 0.8)

    train_set = dataframe.iloc[:dataframe_size, :]
    validation_set = dataframe.iloc[dataframe_size:, :]

    return (train_set, validation_set)


def separate_features_and_target(dataframe):#Isolate features data (X) from target variable matrix (y).

    index = 0

    for col in dataframe:
        if (col == "SalePrice"):
            y = dataframe.iloc[:, index:index + 1]
            break
        index += 1

    X = dataframe.drop(columns="SalePrice")

    return (X, y)


def fit_transform_scaling(X_train):#Compute mean and std from training set and standardize features.
    moy = X_train.mean(axis=0)
    stand_d = X_train.std(axis=0)

    X_norm = (X_train - moy) / stand_d

    dict_norm = {
        "X_norm": X_norm,
        "moy": moy,
        "std": stand_d
    }

    return (dict_norm)


def transform_scaling_val_test(X_test, moy, stand_d):# Standardize validation or test features using predefined mean and std.
    X_test_norm = (X_test - moy) / stand_d

    return (X_test_norm)


def train_linear_model(X_norm, y_train, w, b, learning_rate):#Train linear regression weights using batch Gradient Descent.

    y_pred = 0
    error = 0
    loss = 0

    for i in tqdm(range(10000)):
        #Model for prediction
        y_pred = np.dot(X_norm, w) + b

        #Error
        error = y_pred - y_train

        #Loss
        loss = np.mean(error ** 2)

        #Gradients
        dw = (2 / len(X_norm)) * np.dot(X_norm.T, error)
        db = np.mean(2 * error)

        #Gradients Descente
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # if (i % 100 == 0):
        #     print(f"Loss: \n{loss}")

    params = {
        "weights": w,
        "bias": b,
        "train_pred": y_pred
    }

    return (params)


def make_prediction(X, w, b):#Compute dot product for model inference predictions.
    
    real_pred = np.dot(X, w) + b

    return (real_pred)


if __name__=="__main__": #Main function

    #1- Load the raw train and test CSV files.
    df_train, df_test = load_datasets()

    #2- Clean, map and encode categorical variables using a combined pipeline.
    train_treated, test_treated = encode_categorical_features(df_train, df_test)

    #3- Clean NaN values from dataframes
    train_clean, test_clean = clean_dataframe(train_treated, test_treated)
    #train_clean shape ==> shape(1460, 219)
    #test_clean shape ==> shape(1459, 218)

    #4- Filter and return features with a strong correlation to SalePrice.
    best_features_train = compute_top_correlations(train_clean)

    #5- Drop any column from the dataframe that is not listed in top_features.
    top_features_train = drop_low_correlation_features(train_clean, best_features_train)
    top_features_test = drop_low_correlation_features(test_clean, best_features_train)

    #6- Split the dataset into internal training and validation sets.
    train_set, validation_set = train_val_split(top_features_train)

    #7- Isolate features data (X) from target variable matrix (y).
    X_train, y_train = separate_features_and_target(train_set)
    # X_train shape ==> shape(1168, 38)
    # y_train shape ==> shape(1168, 1)

    X_validation, y_validation = separate_features_and_target(validation_set)
    # X_validation shape ==> shape(292, 38)
    # y_validation shape ==> shape(292, 1)


    #8- Compute mean and std from training set and standardize features.
    dict_norm = fit_transform_scaling(X_train)
    X_norm = dict_norm['X_norm'] #shape(1168, 38)
    moy = dict_norm['moy']
    stand_d = dict_norm['std']

    #9- Standardize validation or test features using predefined mean and std.
    X_val_norm = transform_scaling_val_test(X_validation, moy, stand_d)
    test_norm = transform_scaling_val_test(top_features_test, moy, stand_d)

    #Initialiaze weights and bias values
    n_features = X_norm.shape[1]
    w = np.random.randn(n_features, 1) #shape(38, 1)
    b = 0 #shape(1,)

    if (mode == "train"):
        #10- Train linear regression weights using batch Gradient Descent
        params = train_linear_model(X_norm, y_train, w, b, learning_rate=0.001)

        w = params['weights']#shape(38, 1)
        b = params['bias']#shape(1,)
        y_pred = params['train_pred'] #shape(292, 1)

        save_weights = pd.DataFrame({
                "weights": w.flatten()
        })
        save_weights.to_csv("save_weights.csv", index=False)

        save_bias = pd.DataFrame({
                "bias": b.flatten()
        })
        save_bias.to_csv("save_bias.csv", index=False)

        #Compute train score values
        print(f"score >= 0.8 ==> excellent prediction!\n")
        print(f"score >= 0.5 ==> good prediction!\n")
        print(f"score == 0 ==> bad prediction!\n")
        print(f"score < 0 ==> very bad prediction!\n")

        score_train = r2_score(y_train, y_pred)
        print(f"The accuracy train is: {score_train}\n")

        #11- Compute dot product for model inference predictions (validation dataframe or test dataframe).
        validations_pred = make_prediction(X_val_norm, w, b) #validation_pred shape ==> shape(292,1)
        score_validation = r2_score(y_validation, validations_pred)

        print(f"The accuracy validation is: {score_validation}\n")


    elif (mode == "predict"):
        
        save_weights = pd.read_csv("save_weights.csv")
        save_bias = pd.read_csv("save_bias.csv")
        w = save_weights['weights'].to_numpy()
        b = save_bias['bias'].to_numpy()

        final_pred = make_prediction(test_norm, w, b) #final_pred shape ==> shape(1459, 1)
        
        #Get id column from test clean dataframe
        id_col = test_clean.iloc[:, :1] #id_col.shape = shape(1459, 1)

        #12- Kaggle submission
        
        sample_submission = pd.DataFrame({
                "id": id_col.to_numpy().flatten(),
                "SalePrice": final_pred.flatten()
        })
        sample_submission.to_csv("sample_submission.csv", index=False)