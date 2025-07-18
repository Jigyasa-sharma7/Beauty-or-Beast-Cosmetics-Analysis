import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load your dataset (replace with your actual path or loader)
@st.cache_data
def load_data():
    return pd.read_csv("final_cleaned_data.csv")

df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["Overview", "Descriptive Statistics", "Visualizations", "Product Search", "Chemical Analysis","Discontinuation Prediction"])

# Page 1: Overview
if option == "Overview":
    st.title("Dataset Overview")
    st.write("Total records:", df.shape[0])
    st.write("Total Columns:", df.shape[1])
    st.write("Unique products:", df['ProductName'].nunique())
    st.write("Unique chemicals:", df['ChemicalId'].nunique())
    st.dataframe(df.head())

# Page 2: Descriptive Statistics
elif option == "Descriptive Statistics":
    st.title("Descriptive Statistics")
    numeric_cols = df.select_dtypes(include='number').columns
    selected_col = st.selectbox("Select a numeric column", numeric_cols)
    st.write("Mean:", df[selected_col].mean())
    st.write("Median:", df[selected_col].median())
    st.write("Mode:", df[selected_col].mode()[0])
    st.write("Standard Deviation:", df[selected_col].std())
    st.write("Variance:", df[selected_col].var())
    st.write("Range:", df[selected_col].max() - df[selected_col].min())


# Page 3: Visualizations
elif option == "Visualizations":
    st.title("Visualizations")
    st.title('Distribution of Chemical Count per Product')
#first
# Plot the histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['ChemicalCount'], bins=20, kde=True, ax=ax, color='skyblue')
    ax.set_xlabel('Number of Chemicals in Product')
    ax.set_ylabel('Number of Products')
    ax.set_title('Histogram: How Many Chemicals Are Used in Each Product?')
    st.pyplot(fig)

#second
# Count products per brand
    st.title('Top 10 Brands by Number of Products')
    top_brands = df['BrandName'].value_counts().head(10).reset_index()
    top_brands.columns = ['BrandName', 'ProductCount']
# Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='BrandName', y='ProductCount', data=top_brands, ax=ax, palette='viridis')
    ax.set_xlabel('Brand Name')
    ax.set_ylabel('Number of Products')
    ax.set_title('Top 10 Brands by Product Count')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

#third
    st.title('Chemical Introduction Trends Over Time')

# Convert date column to datetime
    df['ChemicalCreatedAt'] = pd.to_datetime(df['ChemicalCreatedAt'], errors='coerce')

# Extract year
    df['ChemicalCreatedYear'] = df['ChemicalCreatedAt'].dt.year

# Drop NaNs in year
    yearly_chemicals = df.dropna(subset=['ChemicalCreatedYear'])

# Group by year and count distinct chemicals
    chem_trend = yearly_chemicals.groupby('ChemicalCreatedYear')['ChemicalId'].nunique().reset_index()
    chem_trend.columns = ['Year', 'NewChemicals']

# Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=chem_trend, x='Year', y='NewChemicals', marker='o', ax=ax, color='teal')
    ax.set_title('Number of New Chemicals Introduced per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Unique Chemicals')
    ax.grid(True)

# Show plot
    st.pyplot(fig)

# Fourth: Distribution of Primary Categories
    st.title('Distribution of Primary Categories')
    fig = plt.figure(figsize=(12, 6))
    df['PrimaryCategory'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Primary Categories')
    plt.xlabel('Primary Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

#fifth
    st.title('Top 10 Subcategories Proportion')

    # Count subcategories
    subcategory_counts = df['SubCategory'].value_counts()

    # Plot pie chart using matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(subcategory_counts[:10],
           labels=subcategory_counts.index[:10],
           autopct='%1.1f%%',
           startangle=140,
           colors=sns.color_palette('pastel'))
    ax.set_title('Top 10 Subcategories Proportion')
    st.pyplot(fig)

# Sixth: Heatmap of PrimaryCategory vs SubCategory
    st.title('Relationship Between PrimaryCategory and SubCategory')

# Create crosstab
    category_cross_tab = pd.crosstab(df['PrimaryCategoryId'], df['SubCategory'])

# Plot heatmap
    fig, ax = plt.subplots(figsize=(20, 12))  # Larger figure size
    sns.heatmap(category_cross_tab, cmap='Blues', cbar=True, ax=ax)

    # Improve axis label size
    ax.set_title('Relationship Between PrimaryCategory and SubCategory', fontsize=20)
    ax.set_xlabel('SubCategory', fontsize=20)
    ax.set_ylabel('PrimaryCategory', fontsize=20)

# Rotate and resize tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)

#seventh
   
    st.title("Chemical Count Across Primary Categories")

# Create the box plot
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(x='PrimaryCategory', y='ChemicalCount', data=df, ax=ax)
    ax.set_title('Chemical Count Across Primary Categories')
    ax.set_xlabel('Primary Category')
    ax.set_ylabel('Chemical Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

#eight 
    st.title("Pairplot: Relationships Among Numerical Features")

# Create the pairplot
    pairplot_fig = sns.pairplot(df[['ChemicalCount', 'PrimaryCategoryId', 'SubCategoryId']])

# Render using Streamlit
    st.pyplot(pairplot_fig.figure)

#ninth
    st.title("Correlation Heatmap")

# Calculate the correlation matrix for numerical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).corr()

# Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numerical_columns, annot=True, cmap='coolwarm', ax=ax)

# Add title and layout
    ax.set_title('Correlation Heatmap', fontsize=16)
    plt.tight_layout()

# Render in Streamlit
    st.pyplot(fig)

#tenth
    st.title("Word Cloud: Most Common Chemicals")

# Join all chemical names into one string
    text = ' '.join(df['ChemicalName'].dropna().astype(str))

# Generate the word cloud
    chemical_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(chemical_wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Most Common Chemicals')

# Show in Streamlit
    st.pyplot(fig)

# eleventh
    st.title("Product Reporting Trends Over the Years")

# Preprocess the data
    df['YearReported'] = pd.to_datetime(df['InitialDateReported']).dt.year

# Count of products reported each year
    yearly_trends = df['YearReported'].value_counts().sort_index()

# Display a line plot in Streamlit
    st.subheader("Trend of Products Reported Each Year")
    if not yearly_trends.empty:
     fig, ax = plt.subplots(figsize=(10, 4))
     sns.lineplot(x=yearly_trends.index, y=yearly_trends.values, marker="o", color="b", ax=ax)
     ax.set_title("Number of Products Reported Over the Years", fontsize=16)
     ax.set_xlabel("Year")
     ax.set_ylabel("Number of Products")
     ax.grid(True)
     st.pyplot(fig)
    else:
      st.write("No data available for the selected time period.")



# Page 4: Product Search
elif option == "Product Search":
    st.title("Search Product")
    query = st.text_input("Enter product name")
    if query:
        results = df[df['ProductName'].str.contains(query, case=False, na=False)]
        st.write(f"Found {results.shape[0]} results:")
        st.dataframe(results)

# Page 5: Chemical Analysis
elif option == "Chemical Analysis":
    st.title("Chemical Analysis")
    st.markdown("---")
    st.subheader("Chemical Counts in Specific Products")

    # Keyword input
    keyword = st.text_input("Enter a product keyword (e.g., 'hairs', 'shampoo')", 'hairs')

    def get_chemical_counts_in_product(keyword):
        # Filter the dataframe for matching products
        chemicals_in_product = df[df['ProductName'].str.contains(keyword, case=False, na=False)]['ChemicalName']
        chemical_counts = chemicals_in_product.value_counts()
        chemical_counts.index = chemical_counts.index.str.slice(0, 20)  # Trim long names
        return chemical_counts

    # If input is provided
    if keyword:
        chemical_counts = get_chemical_counts_in_product(keyword)
        if not chemical_counts.empty:
            st.subheader(f"Total unique chemicals found in products containing '{keyword}': {chemical_counts.count()}")

            # Show as a table
            chemical_counts_df = pd.DataFrame({
                'Chemical Name': chemical_counts.index,
                'Count': chemical_counts.values
            })
            chemical_counts_df.index = range(1, len(chemical_counts_df) + 1)
            st.dataframe(chemical_counts_df)

            # Pie chart for top 10 chemicals
            top_10_chemicals = chemical_counts.head(10)
            values = top_10_chemicals.values
            labels = top_10_chemicals.index

            fig, ax = plt.subplots(figsize=(10, 10))  # Bigger figure
            wedges, texts, autotexts = ax.pie(
                values,
                autopct='%1.1f%%',
                startangle=90,
                textprops=dict(color="black"),
                pctdistance=1.05  # Pulls percentage labels closer in
)

            ax.axis('equal')  # Makes sure it's a circle
            ax.set_title(f"Top 10 Chemicals in Products Containing '{keyword}'", fontsize=14)

# Adjust the legend to prevent overlap
            ax.legend(
                wedges,
                labels,
                title="Chemicals",
                loc="upper left",
                bbox_to_anchor=(1, 0.9),
                fontsize=8
)

            plt.tight_layout()
            st.pyplot(fig)


# Page 6: Discontinuation Prediction
elif option == "Discontinuation Prediction":
    st.title("Product Discontinuation Prediction")

    st.write("This tool predicts whether a product is likely to be discontinued based on selected attributes.")

    #Prepare data
    if 'DiscontinuedDate' in df.columns:
     df['Discontinued'] = df['DiscontinuedDate'].notnull().astype(int)
    else:
     st.error("⚠️ The column 'DiscontinuedDate' is missing from the dataset.")
     st.stop()
    model_df = df.dropna(subset=['CompanyId', 'PrimaryCategoryId', 'SubCategoryId', 'ChemicalCount'])

    X = model_df[['CompanyId', 'PrimaryCategoryId', 'SubCategoryId', 'ChemicalCount']]
    y = model_df['Discontinued']

    #Train model (you could cache or save this later)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

  

    # Select a product from dropdown
    product_names = model_df['ProductName'].dropna().unique()
    selected_product = st.selectbox("Select a product to predict discontinuation status:", product_names)

# Get its features
    product_row = model_df[model_df['ProductName'] == selected_product].iloc[0]
    input_data = pd.DataFrame([{
    'CompanyId': product_row['CompanyId'],
    'PrimaryCategoryId': product_row['PrimaryCategoryId'],
    'SubCategoryId': product_row['SubCategoryId'],
    'ChemicalCount': product_row['ChemicalCount']
}])

# Show selected values
    st.write("Selected product details:")
    st.json(input_data.to_dict(orient='records')[0])

    if st.button("Predict"):
        proba = model.predict_proba(input_data)[0][1]
        if proba > 0.5:
           st.error(f"⚠️ This product is likely to be discontinued. (Probability: {proba:.2f})")
        else:
           st.success(f"✅ This product is likely to remain available. (Probability: {proba:.2f})")

    #Confusion Matrix
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Not Discontinued', 'Discontinued'], 
                yticklabels=['Not Discontinued', 'Discontinued'])
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)



   