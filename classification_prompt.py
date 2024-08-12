import streamlit as st
import pandas as pd
from tqdm import tqdm
from langchain_openai import AzureChatOpenAI
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import io
import os

# Load environment variables from .env file
load_dotenv()

# Set up environment variables for Azure OpenAI API
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Initialize the AzureChatOpenAI model
llm = AzureChatOpenAI(
    azure_deployment="GPT35TURBO16K",
    api_version="2023-03-15-preview",
    temperature=0,
    max_tokens=10,
    max_retries=2,
)

# Function to classify a question using the AzureChatOpenAI model
def classify_question(question, prompt_template):
    messages = [
        ("system", prompt_template.format(question=question)),
        ("human", question),
    ]
    response = llm.invoke(messages)
    category = response.content.strip()

    # Map the response to a category label
    category_map = {
        "Technical Issues": "0",
        "Individual User Issues": "1",
        "System, Tool, Program Features": "2",
        "Process and Guidance": "3"
    }

    return category_map.get(category, "Unknown")

# Function to validate the uploaded Excel file
def validate_excel_file(df):
    required_columns = ['Question', 'manually_tagged']
    return all(col in df.columns for col in required_columns)

# Function to process the Excel file and calculate accuracy
def process_excel_file(input_file, start_row, end_row, prompt_template):
    df = pd.read_excel(input_file)
    
    # Validate the file format
    if not validate_excel_file(df):
        return None, "Error: The uploaded file does not have the required columns ('Question' and 'manually_tagged'). Please upload a file in the correct format."

    questions = df.iloc[start_row:end_row+1, df.columns.get_loc('Question')].tolist()
    actual_categories = df.iloc[start_row:end_row+1]['manually_tagged'].astype(str).tolist()

    predicted_categories = []
    for question in tqdm(questions, desc="Classifying questions..."):
        category = classify_question(question, prompt_template)
        predicted_categories.append(category)
    
    # Map actual categories to Y/N based on the logic you provided
    actual_binary_tags = ['Y' if item in ['2', '3'] else 'N' for item in actual_categories]
    
    # Map predicted categories to Y/N
    predicted_binary_tags = ['Y' if category in ['2', '3'] else 'N' for category in predicted_categories]
    
    # Add the predicted category and binary tag columns to the DataFrame
    df_subset = df.iloc[start_row:end_row+1].copy()
    # Create the new 'manually_tagged_YN' column based on 'manually_tagged'
    if not all(tag in ['Y', 'N'] for tag in actual_categories):
        df_subset['manually_tagged_YN'] = actual_binary_tags
    df_subset['Predicted_Category'] = predicted_categories
    df_subset['Predicted_YN'] = predicted_binary_tags
    
    # Calculate accuracy using binary 'Y/N' labels
    accuracy = accuracy_score(actual_binary_tags, predicted_binary_tags)

    return df_subset, accuracy

# Streamlit App
def main():
    st.title("Classification Model Testing with Prompt Adjustment")

    # Default prompt template
    prompt_template = """You are an expert in classifying incoming requests originating from users inside an enterprise who are dealing with process, system, policy, best practices, and guidelines related issues when accomplishing a given task. 
        Examine the requests framed in the form of questions into one of the four categories and respond **only with the category names**:

        **Technical Issues**: Questions related to error messages, system malfunctions, or issues users encounter while using the system. Examples include questions about specific error codes, login problems, and issues with infrastructure, connectivity, or local setup.
        **Individual User Issues**: Questions that involve specific user actions, personal access, individual settings, unique user permissions, or requests for specific records. These questions focus on the unique needs or challenges faced by an individual user.
        **System, Tool, Program Features**: These questions pertain to how certain features or functions work, including how to export data to Excel and adjust settings within the system. The program could also mean a Marketing Program and questions around market program eligibility, target audience, or Sales Program to include compensation structure, payment method, etc.
        **Process and Guidance**: Questions seek guidance on the process steps to follow, including best practices, or procedures for handling particular scenarios. Process steps could include registration, quoting, ordering, configuration, shipping, enrollment, etc.

        Classify the following requests into one of these four categories and provide **only the category names as the response**. Do not justify the bucket in any way. Simply return the name of the bucket:

        Question: {question}
        0-3:
    """
    
    # Display category annotations
    st.header("Category Annotations")
    st.markdown("""
| **Category** | **Description** |
|--------------|-----------------|
| 0            | Technical Issues |
| 1            | Individual User Issues |
| 2            | System, Tool, Program Features |
| 3            | Process and Guidance |
""")

    # Single Question Classification Section
    st.header("Single Question Classification")
    
    # Input field for a single question
    single_question = st.text_input("Enter a question to classify:")
    
    # Editable prompt template
    prompt_template = st.text_area(
        "Modify the Prompt Template:",
        value=prompt_template,
        height=300  # Adjust height to increase the text box size
    )

    if st.button("Classify Single Question"):
        if single_question:
            predicted_category = classify_question(single_question, prompt_template)
            predicted_binary_tag = 'Y' if predicted_category in ['2', '3'] else 'N'
            st.write(f"Predicted Category: **{predicted_category}**")
            st.write(f"Predicted Tag: **{predicted_binary_tag}**")
        else:
            st.warning("Please enter a question to classify.")

    # Batch Classification Section
    st.header("Batch Classification with Excel File")

    st.subheader("Instructions for File Upload:")
    st.markdown("""
Please ensure your Excel file contains the following columns:

- **Question (1st column)**: The question to be classified.
- **manually_tagged (2nd column)**: This column can contain either:
  - **Category Labels (0, 1, 2, 3)**: For direct comparison with predicted categories, or 
  - **Binary Tags (Y, N)**: For comparison with binary predictions derived from categories 2 and 3.

The classification will be based on the questions in the 1st column, and accuracy will be calculated using the `manually_tagged` column as the reference. Depending on the format provided in the `manually_tagged` column, accuracy will be calculated based on either category labels (`0,1,2,3`) or binary `Y/N` tags.

The names of the two columns must be exactly 'Questions' and 'manually_tagged'.

**Note:** To avoid incurring high costs, please select the start and end row indices for a subset of questions to analyze. Once you are confident with the prompt and its accuracy, you can analyze the entire file.
""")

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        st.write("File uploaded successfully!")

        # Input for start and end rows
        start_row = st.number_input("Start Row (0-indexed)", min_value=0, value=0, step=1)
        end_row = st.number_input("End Row (0-indexed)", min_value=0, value=5, step=1)

        if st.button("Classify and Evaluate"):
            # Process the file and get accuracy
            df_classified, accuracy = process_excel_file(uploaded_file, start_row, end_row, prompt_template)

            if isinstance(accuracy, str):  # Error message
                st.error(accuracy)
            else:
                # Store accuracy and classified DataFrame in session state
                st.session_state['accuracy'] = accuracy
                st.session_state['df_classified'] = df_classified

    if 'accuracy' in st.session_state:
        st.write(f"Accuracy of Classification (Reusability - Y,N) :  **{st.session_state['accuracy'] * 100:.2f}%**")

        # Allow download of classified DataFrame
        output = io.BytesIO()
        st.session_state['df_classified'].to_excel(output, index=False, engine='openpyxl')
        output.seek(0)

        st.download_button(
            label="Download Classified Results",
            data=output,
            file_name="classified_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
