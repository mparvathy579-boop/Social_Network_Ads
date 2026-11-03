import streamlit as st
import pandas as pd
import pickle



st.set_page_config(
    page_title =" Customer Purchase Prediction",
    layout = "centered"
)

model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open("scaler.pkl",'rb'))
feature_colunms = pickle.load(open("feature_colunms",'rb'))


# age gender from user side
st.title('Social Network ads Prediction')
st.subheader('Purchaces Predicton')
st.write(
    'Enter Customer Details to predict'
    'where they are likely to Purchase or Not'
)

gender = st.selectbox(
    "Select Gender",
    ['Male','Female']
)
age = st.number_input(
    "Enter Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)

salary = st.number_input(
    "Enter Estimated Salary (₹)",
    min_value=0,
    max_value=1000000,
    value=50000,
    step=1000
) 

# Prediction button

if st.button("Prediction Purchase", type  = "primary"):
    input_data = pd.DataFrame({
        "Age":[age],
        "Gender":[gender],
        "EstimatedSalary":[salary]
    })
    #onehot encoding
    input_data = pd.get_dummies(
        input_data,
        columns = ['Gender'],dtype = int
    )
     # Match the training columns and their order
    input_data = input_data.reindex(
        columns=feature_colunms,
        fill_value=0
    )

    # Scale input data
    input_scaled = scaler.transform(input_data)


    # just print 
    st.write(input_data)

    # Scale input data

    input_scaled = scaler.transform(input_data)
    # prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("The customer is likely to purchase the product!")
    else:
        st.info("The customer is unlikely to purchase the product.")
    st.write("Purchase Probability: {:.2f}%".format(probability * 100))