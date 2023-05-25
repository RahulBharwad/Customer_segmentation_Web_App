import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit config option to disable the warning
st.set_option('deprecation.showPyplotGlobalUse', False)


# Load the KMeans model from the pickle file
filename = 'clusters_kmean_f.pkl'
loaded_kmeans = pickle.load(open(filename, 'rb'))

def preprocess_inputs(idf):
    
    # Map education categories
    idf['Education']=np.where(idf['Education']== 'Basic',0,np.where(idf['Education']=='2n Cycle',0,
            np.where(idf['Education']=='Graduation',1,
                        np.where(idf['Education']=='Master',2,
                                 np.where(idf['Education']=='PhD',3,idf['Education'])))))
    
    # Map marital status categories
    idf['Married'] = np.where(idf['Married'] == 'Married', 'Married',
                                      np.where(idf['Married'] == 'Together', 'Married',
                                               np.where(idf['Married'] == 'Single', 'Unmarried',
                                                        np.where(idf['Married'] == 'Divorced', 'Unmarried',
                                                                 np.where(idf['Married'] == 'Widow', 'Unmarried',
                                                                          idf['Married'])))))

   
    #Creating new column Using numpy's where function to replace string values with numeric values
    idf['Married']=np.where (idf['Married']== 'Married', 1,
        np.where (idf['Married']== 'Unmarried', 0, 1))
    
    
    month_mapping = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
     }
    
    
    # Replace month values with numeric labels
    idf["Dt_Customer_Month"] = np.where(
        idf["Dt_Customer_Month"].isin(month_mapping.keys()),
        idf["Dt_Customer_Month"].map(month_mapping),
        idf["Dt_Customer_Month"]
    )
    
    return idf



# Function to preprocess the input data
def preprocess_input(df):
    
    #dropping all the null values
    data=df.dropna()
    
    # Calculate age from year of birth
    data['Age'] = 2023 - data['Year_Birth']
    data.drop('Year_Birth', axis=1, inplace=True)

    # Drop unnecessary columns
    data.drop(['ID', 'Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)

    # Calculate total spending
    data['Total_Sped'] = data[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                               'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

    # Extract year and month from Dt_Customer column
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])
    data['Dt_Customer_year'] = data['Dt_Customer'].dt.year
    data['Dt_Customer_Month'] = data['Dt_Customer'].dt.month
    data.drop('Dt_Customer', axis=1, inplace=True)

    # Map education categories
    data['Education']=np.where(data['Education']== 'Basic',0,np.where(data['Education']=='2n Cycle',0,
            np.where(data['Education']=='Graduation',1,
                        np.where(data['Education']=='Master',2,
                                 np.where(data['Education']=='PhD',3,data['Education'])))))
    
    # Map marital status categories
    data['Marital_Status'] = np.where(data['Marital_Status'] == 'Married', 'Married',
                                      np.where(data['Marital_Status'] == 'Together', 'Married',
                                               np.where(data['Marital_Status'] == 'Single', 'Unmarried',
                                                        np.where(data['Marital_Status'] == 'Divorced', 'Unmarried',
                                                                 np.where(data['Marital_Status'] == 'Widow', 'Unmarried',
                                                                          data['Marital_Status'])))))

   
    #Creating new column Using numpy's where function to replace string values with numeric values
    data['Married']=np.where (data['Marital_Status']== 'Married', 1,
        np.where (data['Marital_Status']== 'Unmarried', 0, 1))

    # Drop the original marital status column
    data.drop(['Marital_Status'], axis=1, inplace=True)

    return data




    
    # Create the form in the sidebar using st.sidebar.form()
with st.sidebar.form("User Input Form"):
    # Education
    education = st.selectbox("Education", ("Graduation", "PhD", "Master", "2n Cycle", "Basic"))

    # Income
    income = st.number_input("Income", step=1000.0)

    # Kidhome
    kidhome = st.selectbox("Kidhome", (0, 1, 2))

    # Teenhome
    teenhome = st.selectbox("Teenhome", (0, 1, 2))

    # Recency
    recency = st.number_input("Recency", step=1)

    # MntWines
    mnt_wines = st.number_input("MntWines", step=1)

    # MntFruits
    mnt_fruits = st.number_input("MntFruits", step=1)

    # MntMeatProducts
    mnt_meat_products = st.number_input("MntMeatProducts", step=1)

    # MntFishProducts
    mnt_fish_products = st.number_input("MntFishProducts", step=1)

    # MntSweetProducts
    mnt_sweet_products = st.number_input("MntSweetProducts", step=1)

    # MntGoldProds
    mnt_gold_prods = st.number_input("MntGoldProds", step=1)

    # NumDealsPurchases
    num_deals_purchases = st.number_input("NumDealsPurchases", step=1)

    # NumWebPurchases
    num_web_purchases = st.number_input("NumWebPurchases", step=1)

    # NumCatalogPurchases
    num_catalog_purchases = st.number_input("NumCatalogPurchases", step=1)

    # NumStorePurchases
    num_store_purchases = st.number_input("NumStorePurchases", step=1)

    # NumWebVisitsMonth
    num_web_visits_month = st.number_input("NumWebVisitsMonth", step=1)

    # AcceptedCmp3
    accepted_cmp3 = st.number_input("AcceptedCmp3", step=1)

    # AcceptedCmp4
    accepted_cmp4 = st.number_input("AcceptedCmp4", step=1)

    # AcceptedCmp5
    accepted_cmp5 = st.number_input("AcceptedCmp5", step=1)

    # AcceptedCmp1
    accepted_cmp1 = st.number_input("AcceptedCmp1", step=1)

    # AcceptedCmp2
    accepted_cmp2 = st.number_input("AcceptedCmp2", step=1)

    # Complain
    complain = st.number_input("Complain", step=1)

    # Z_CostContact
    z_cost_contact = st.number_input("Z_CostContact", step=1)

    # Z_Revenue
    z_revenue = st.number_input("Z_Revenue", step=1)

    # Response
    response = st.number_input("Response", step=1)

    # Age
    age = st.number_input("Age", step=1)

    # Total_Sped
    total_spent = st.number_input("Total_Sped", step=1.0)

    # Dt_Customer_year
    dt_customer_year = st.number_input("Dt_Customer_year", step=1)

    # Dt_Customer_Month
    dt_customer_month = st.selectbox("Dt_Customer_Month", ("January", "February", "March", "April", "May", "June",
                                                            "July", "August", "September", "October", "November", "December"))

    # Married
    married = st.selectbox("Married", ("Married", "Together", "Single", "Divorced", "Widow"))

    # Submit button
    submit_button = st.form_submit_button("Submit")
    
    # Process the form submission and create the DataFrame
    if submit_button:
        # Create a dictionary with the input values
        data = {
            "Education": [education],
            "Income": [income],
            "Kidhome": [kidhome],
            "Teenhome": [teenhome],
            "Recency": [recency],
            "MntWines": [mnt_wines],
            "MntFruits": [mnt_fruits],
            "MntMeatProducts": [mnt_meat_products],
            "MntFishProducts": [mnt_fish_products],
            "MntSweetProducts": [mnt_sweet_products],
            "MntGoldProds": [mnt_gold_prods],
            "NumDealsPurchases": [num_deals_purchases],
            "NumWebPurchases": [num_web_purchases],
            "NumCatalogPurchases": [num_catalog_purchases],
            "NumStorePurchases": [num_store_purchases],
            "NumWebVisitsMonth": [num_web_visits_month],
            "AcceptedCmp3": [accepted_cmp3],
            "AcceptedCmp4": [accepted_cmp4],
            "AcceptedCmp5": [accepted_cmp5],
            "AcceptedCmp1": [accepted_cmp1],
            "AcceptedCmp2": [accepted_cmp2],
            "Complain": [complain],
            "Z_CostContact": [z_cost_contact],
            "Z_Revenue": [z_revenue],
            "Response": [response],
            "Age": [age],
            "Total_Sped": [total_spent],
            "Dt_Customer_year": [dt_customer_year],
            "Dt_Customer_Month": [dt_customer_month],
            "Married": [married]
            }
        
        # Create the DataFrame
        idf1 = pd.DataFrame(data)
        
        # Create fit DataFrame
        idf2 = pd.read_excel('fit.xlsx')
        idf2 = idf2.dropna()
        
        # Preprocess the input data
        idf3 = preprocess_inputs(idf1)
        
        # Combine the two datasets by adding rows
        preprocessed_data = idf2.append(idf3)
        
        # Reset the index if needed
        preprocessed_data = preprocessed_data.reset_index(drop=True)
        
        #standardising data
        from sklearn.preprocessing import StandardScaler
        scaler= StandardScaler()
        preprocessed_data = preprocessed_data.fillna(preprocessed_data.mean())
        data_std=scaler.fit_transform(preprocessed_data)
        
        
        #Fitting data in model
        loaded_kmeans.fit(data_std)

        # Perform clustering on preprocessed data
        #loaded_kmeans.labels_
        l = loaded_kmeans.labels_.tolist()

        preprocessed_data['clusters'] =loaded_kmeans.labels_
        
        
        
        # Get the last value of column B
        last_value_B = l[-1] 
        st.info(f'customer lies in Cluster Number = {last_value_B }')
        
        
    
    

    
# Main function to run the Streamlit app
def main():
    # Set Streamlit app title
    st.title('Customer Personality Analysis')    
    
    st.text("For individual use, please fill form from side bar")
    

    # Add a file uploader to upload the input Excel file
    uploaded_file = st.file_uploader('It can also work on a large number of customer data by uploading a file here you will get costomers segments.', type=['csv', 'xls','xlsx'])

    if uploaded_file is not None:
        # Read the uploaded Excel file
        df = pd.read_excel(uploaded_file)
        
         # Preprocess the input data
        preprocessed_data = preprocess_input(df)
        
        #standardising data
        from sklearn.preprocessing import StandardScaler
        scaler= StandardScaler()
        data_std=scaler.fit_transform(preprocessed_data)
        
        
        #Fitting data in model
        loaded_kmeans.fit(data_std)

        # Perform clustering on preprocessed data
        #loaded_kmeans.labels_
        clusters =loaded_kmeans.labels_

        # Add the clusters column to the input data
        preprocessed_data['Clusters'] = clusters

        # Display the clustering results
        st.write(f'shape of data :- {preprocessed_data.shape}')
        
        st.subheader('Cluster Column added which shows label of cluster number:-')
        st.write(preprocessed_data)
        
        
        st.subheader('count of customer in each cluster numbers:-')
        # distribution of data in all clusters
        #preprocessed_data["Clusters"].value_counts().plot(kind='bar')
        #plt.xlabel('Cluster Number')
        #plt.ylabel('Count')
        sns.countplot(x='Clusters', data=preprocessed_data, palette="pastel")
        plt.title('Distribution of Data in Clusters')
        st.pyplot()
        print(preprocessed_data.value_counts())
        
        
        st.subheader('Visualization of Clusters')
        # Create a figure with subplots for histogram and box plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot histogram of Clusters variable
        ax1.hist(preprocessed_data["Clusters"], bins=len(np.unique(loaded_kmeans.labels_)), edgecolor='black')
        ax1.set_xlabel('Cluster Number')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Histogram of Clusters')
        
        # Plot box plot of Clusters variable
        ax2.boxplot(preprocessed_data["Clusters"])
        ax2.set_xlabel('Clusters')
        ax2.set_ylabel('Cluster Number')
        ax2.set_title('Box Plot of Clusters')
        
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.4)
        
        # Display the figure
        st.pyplot(fig)
        
        #Create a scatter plot with income on the y-axis and total spend on the x-axis
        #sns.scatterplot(x=preprocessed_data["Total_Sped"], y=preprocessed_data["Income"], hue=preprocessed_data["Clusters"])
        # Set labels and title
        #plt.xlabel('Total Spend')
        #plt.ylabel('Income')
        #plt.title('Scatter Plot with Clusters')
        
        
        
if __name__ == '__main__':
    main()
