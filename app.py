import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

# Set page title and icon
st.set_page_config(page_title="Airline Passenger Satisfaction", page_icon='✈️')

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions"])

# Load dataset
df = pd.read_csv('train_clean.csv')

# Home Page
if page == "Home":
    st.title("📊 Airline Passenger Satisfaction")
    st.subheader("Welcome to our Airline Passenger Satisfaction app!")
    st.write("""
        This app provides an interactive platform to explore an airline passenger satisfaction survey. 
        You can visualize the distribution of data, explore relationships between features, and even make predictions on new data!
        Use the sidebar to navigate through the sections.
    """)
    st.image('https://cdn.mos.cms.futurecdn.net/Tpwmmfo3CiAJvwd4vXGzvn-1200-80.jpg')
    st.write("Use the sidebar to navigate between different sections")


# Data Overview
elif page == "Data Overview":
    st.title("🔢 Data Overview")

    st.subheader("About the Data")
    st.write("""
        This airline passenger satisfaction dataset is used to determine what makes a satisfied airline passenger. 
        Are we able to predict passenger satisfaction? The dataset contains over 100,000 rows of passenger satisfaction survey results. 
        For each passenger, the dataset has various features scored from 1-5 with 5 being very satisfied. Example of features include:
        Inflight WiFi service, Ease of Online Booking, Seat comfort, On-board service, Baggage handling, Cleanliness and several more we'll explore.
        
    """)
    st.image('https://static1.simpleflyingimages.com/wordpress/wp-content/uploads/2020/05/K65704.jpg', caption = '737 Airplanes')

    # Dataset Display
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)
    

    # Shape of Dataset
    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")


# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])

    obj_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()

    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show by Customer Type"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, color='Customer Type', title=chart_title, barmode='overlay'))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))

    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for the box plot:", num_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.box(df, x='Customer Type', y=b_selected_col, title=chart_title, color='Customer Type'))
            if st.checkbox("Show by Gender"):
                st.plotly_chart(px.box(df, x='Customer Type', y=b_selected_col, title=chart_title, color='Gender'))
            if st.checkbox("Show by Class"):
                st.plotly_chart(px.box(df, x='Customer Type', y=b_selected_col, title=chart_title, color='Class'))
            if st.checkbox("Show by Type of Travel"):
                st.plotly_chart(px.box(df, x='Customer Type', y=b_selected_col, title=chart_title, color='Type of Travel'))

    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols)
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='Customer Type', title=chart_title))

    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        selected_col = st.selectbox("Select a categorical variable:", obj_cols)
        if selected_col:
            chart_title = f'Distribution of {selected_col.title()}'
            st.plotly_chart(px.histogram(df, x=selected_col, color='Customer Type', title=chart_title))

# Model Training and Evaluation Page
elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")

    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = df.drop(columns = ['satisfaction', 'Gender', 'Customer Type', 'Type of Travel', 'Class'])
    y = df['satisfaction']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.3f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.3f}")
    st.write(f"Precision Score: {precision_score(y_test, y_pred, average = 'weighted'):.3f}")
    st.write(f"Recall Score: {recall_score(y_test, y_pred, average = 'weighted'):.3f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
    st.pyplot(fig)

# Make Predictions Page
elif page == "Make Predictions":
    st.title("✈️  Make Predictions")

    st.subheader("Adjust the values below to make predictions on if a passenger is neutral/dissatisfied or satisfied with their flying experience:")

    # User inputs for prediction
    online_boarding = st.slider("Online Boarding", min_value=1, max_value=5, value=3)
    inflight_entertainment = st.slider("Inflight Entertainment", min_value=1, max_value=5, value=3)
    seat_comfort = st.slider("Seat Comfort", min_value=1, max_value=5, value=3)
    onboard_service = st.slider("Onboard Service", min_value=1, max_value=5, value=3)
    leg_room = st.slider("Leg Room Service", min_value=1, max_value=5, value=3)
    cleanliness = st.slider("Cleanliness", min_value=1, max_value=5, value=3)
    inflight_wifi = st.slider("Inflight WiFi Service", min_value=1, max_value=5, value=3)
    inflight_service = st.slider("Inflight Service", min_value=1, max_value=5, value=3)
    baggage_handling = st.slider("Baggage Handling", min_value=1, max_value=5, value=3)
    food_drink = st.slider("Food and Drink", min_value=1, max_value=5, value=3)
    checkin_service = st.slider("Check-In Service", min_value=1, max_value=5, value=3)
    ease_of_booking = st.slider("Ease of Online Booking", min_value=1, max_value=5, value=3)
    departure_arrival = st.slider("Departure/Arrival Time Convenient", min_value=1, max_value=5, value=3)
    age = st.slider("Age", min_value=10, max_value=75, value=25)
    departure_delay = st.slider("Departure Delay in Minutes", min_value=1, max_value=100, value=20)
    arrival_delay = st.slider("Arrival Delay in Minutes", min_value=1, max_value=100, value=20)
    flight_distance = st.slider("Flight Distance (miles)", min_value=100, max_value=3000, value=1000)
    neighbors = st.slider("Select amount of KNN neighbors", min_value = 1, max_value = 19, value = 9)

    # User input dataframe
    user_input = pd.DataFrame({
        'Online boarding': [online_boarding],
        'Inflight entertainment': [inflight_entertainment],
        'Seat comfort': [seat_comfort],
        'On-board service': [onboard_service],
        'Leg room service': [leg_room],
        'Cleanliness': [cleanliness],
        'Inflight wifi service': [inflight_wifi],
        'Inflight service': [inflight_service],
        'Baggage handling': [baggage_handling],
        'Food and drink': [food_drink],
        'Checkin service': [checkin_service],
        'Ease of Online booking': [ease_of_booking],
        'Departure/Arrival time convenient': [departure_arrival],
        'Age': [age],
        'Departure Delay in Minutes': [departure_delay],
        'Arrival Delay in Minutes': [arrival_delay],
        'Flight Distance': [flight_distance]

    })

    st.write("### Your Input Values")
    st.dataframe(user_input)

    # Use KNN (k=9) as the model for predictions
    model = KNeighborsClassifier(n_neighbors=neighbors)
    features = ['Online boarding', 'Inflight entertainment', 'Seat comfort', 'On-board service', 'Leg room service', 'Cleanliness',
                'Inflight wifi service', 'Inflight service', 'Baggage handling', 'Food and drink', 'Checkin service', 'Ease of Online booking',
                'Departure/Arrival time convenient', 'Age', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Flight Distance']
    X = df[features]
    y = df['satisfaction']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Train the model on the scaled data
    model.fit(X_scaled, y)

    # Make predictions
    prediction = model.predict(user_input_scaled)[0]


    # Display the result
    st.write(f"The model predicts that this customer is: **{prediction}**")