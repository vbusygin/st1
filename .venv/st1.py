import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import streamlit as st

# Load the iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Display the DataFrame in Streamlit
st.dataframe(df)

# Show general information about the dataset
st.text(df.info())

# Show statistical information about the dataset
st.write(df.describe())

# Select a feature to display histogram
feature = st.selectbox('Select a feature', df.columns)

# Plot histogram
fig, ax = plt.subplots()
ax.hist(df[feature], bins=20)

# Set the title and labels
ax.set_title(f'Histogram of {feature}')
ax.set_xlabel(feature)
ax.set_ylabel('Frequency')

# Display the plot
st.pyplot(fig)


import matplotlib.pyplot as plt
import seaborn as sns

# Select features to display scatter plot
feature_x = st.selectbox('Select feature for x axis', df.columns)
feature_y = st.selectbox('Select feature for y axis', df.columns)

# Display scatter plot
fig, ax = plt.subplots()
sns.scatterplot(data=df, x=feature_x, y=feature_y, hue=iris.target, ax=ax)
st.pyplot(fig)