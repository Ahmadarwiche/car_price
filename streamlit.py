import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler, RobustScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer


df = pd.read_csv('cleaned_data.csv')


# #################################{MODELISATION}##################################################################
y = df['price']
X = df.drop('price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


# categorial_features = ['symboling', 'fueltype', 'aspiration', 'doornumber',  'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem', 'marck', 'model'   ]
# numeric_features = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
#         'enginesize', 'boreratio', 'stroke',
#        'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'cylindernumber']



# numeric_transformer = Pipeline([
#         #('imputer', SimpleImputer(strategy='mean')),
#         ('rbscaler' , RobustScaler()),  
#         ])
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')


# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorial_features)
#     ]
# )

# model = Ridge()
# pipe = Pipeline([
#      ('prep', preprocessor),
#      ('model', model)
# ])

# from sklearn.model_selection import GridSearchCV
# parameters = {'model__alpha':[1, 10]}

# # define the grid search
# grid = GridSearchCV(pipe, parameters,cv=5)
# #fit the grid search
# grid.fit(X_train,y_train)
# best_model = grid.best_estimator_
# best_model.fit(X_train,y_train)
# st.write(best_model.score(X_test,y_test))
###############################{LOAD OUR MODEL VIA PICKLE}################################################################
import pickle
with open('car.pkl', 'rb') as file:
    best_modell = pickle.load(file)
# best_modell.fit(X_train,y_train)
# st.write(best_modell.score(X_test,y_test))
###################{STREAMLIT APPLICATION }#######################################################################
st.title('Predicting Car Prices')

# Définition des variables
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    symboling = st.selectbox('Symboling', df['symboling'].unique())
    fueltype = st.selectbox('Fuel Type', df['fueltype'].unique())
    aspiration = st.selectbox('Aspiration', df['aspiration'].unique())
    doornumber = st.selectbox('Door Number', df['doornumber'].unique())
    carbody = st.selectbox('Car Body', df['carbody'].unique())
with col2:   
    drivewheel = st.selectbox('Drive Wheel', df['drivewheel'].unique())
    enginelocation = st.selectbox('Engine Location', df['enginelocation'].unique())
    wheelbase = st.number_input('Wheel Base')
    carlength = st.number_input('Car Length')
    carwidth = st.number_input('Car Width')
with col3:
    carheight = st.number_input('Car Height')
    curbweight = st.number_input('Curb Weight')
    enginetype = st.selectbox('Engine Type', df['enginetype'].unique())
    cylindernumber = st.selectbox('Cylinder Number', df['cylindernumber'].unique())
    enginesize = st.number_input('Engine Size')
with col4:
    fuelsystem = st.selectbox('Fuel System', df['fuelsystem'].unique())
    boreratio = st.number_input('Bore Ratio')
    stroke = st.number_input('Stroke')
    compressionratio = st.number_input('Compression Ratio')
    horsepower = st.number_input('Horsepower')
with col5:
    peakrpm = st.number_input('Peak RPM')
    citympg = st.number_input('City MPG')
    highwaympg = st.number_input('Highway MPG')
    marck = st.text_input('Marck')
    model = st.text_input('Model')
    
    
# add a button to trigger prediction
if st.button('Predict Price'):
    # create a dictionary with user inputs
    input_data = {
        'symboling': symboling,
        'fueltype': fueltype,
        'aspiration': aspiration,
        'doornumber': doornumber,
        'carbody': carbody,
        'drivewheel': drivewheel,
        'enginelocation': enginelocation,
        'enginetype': enginetype,
        'fuelsystem': fuelsystem,
        'marck': marck,
        'model': model,
        'wheelbase': wheelbase,
        'carlength': carlength,
        'carwidth': carwidth,
        'carheight': carheight,
        'curbweight': curbweight,
        'enginesize': enginesize,
        'boreratio': boreratio,
        'stroke': stroke,
        'compressionratio': compressionratio,
        'horsepower': horsepower,
        'peakrpm': peakrpm,
        'citympg': citympg,
        'highwaympg': highwaympg,
        'cylindernumber': cylindernumber
    }
    
    # convert the dictionary to a dataframe
    input_df = pd.DataFrame([input_data])
    
    # use the pre-trained model to predict the price
    predicted_price = best_modell.predict(input_df)[0]
    
    # show the predicted price on the app
    if predicted_price>0:
         st.info(f'Predicted price: {predicted_price:.2f} $')
    else:
        st.info('The trained data is not reasonable.')
###############################################################################################