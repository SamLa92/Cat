import streamlit as st
import pandas as pd
import numpy as np
import spacy
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import base64
import time
import logging
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from langdetect import detect
from googletrans import Translator
import io

# Set page configuration
st.set_page_config(
    page_title="Call Summary Categorization Tool",
    page_icon="📞",
    layout="wide",
)

# Initialize logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    import spacy
    from spacy.cli import download
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        # Model not found, download it
        download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    return nlp

nlp = load_spacy_model()


# Load BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, bert_model = load_bert_model()

# Initialize translator
translator = Translator()

# Title and description
st.title("📞 Call Summary Categorization Tool")
st.markdown(
    """
    Upload a CSV file containing call summaries to categorize them into customer demand buckets.
    Customize preprocessing steps, select clustering algorithms, and adjust model parameters to suit your needs.
    """
)

# In-app tutorial
with st.expander("How to Use This App"):
    st.write("""
    1. **Upload Data**: Upload a CSV file containing your call summaries. Ensure it has a 'summary' column.
    2. **Customize Settings**: Use the sidebar to adjust preprocessing steps, clustering parameters, and feature extraction methods.
    3. **Run Analysis**: The app will process the data, perform clustering, and generate visualizations.
    4. **Review Results**: Explore the categorized data, edit category names if needed, and download the results.
    """)

# Sidebar for user preferences
st.sidebar.header("User Preferences")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    help="Upload a CSV file containing call summaries. The file must have a 'summary' column."
)

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    # Check if 'summary' column exists
    if 'summary' not in df.columns:
        st.error("The uploaded CSV file must contain a 'summary' column.")
    else:
        # Advanced Preferences
        st.sidebar.header("Advanced Preferences")

        # Preprocessing options
        preprocessing_options = st.sidebar.multiselect(
            "Preprocessing Steps",
            options=["Lowercase", "Remove Punctuation", "Remove Stop Words", "Lemmatization"],
            default=["Lowercase", "Remove Stop Words", "Lemmatization"],
            help="Select the preprocessing steps to apply to the text data."
        )

        # Custom stop words
        custom_stop_words_input = st.sidebar.text_area(
            "Custom Stop Words",
            help="Enter custom stop words separated by commas."
        )

        if custom_stop_words_input:
            custom_stop_words = [word.strip() for word in custom_stop_words_input.split(',')]
        else:
            custom_stop_words = []

        # Clustering parameters
        st.sidebar.subheader("Clustering Parameters")

        algorithm = st.sidebar.selectbox(
            "Choose Clustering Algorithm",
            ["KMeans", "Agglomerative Clustering", "DBSCAN"],
            help="Select the clustering algorithm to group similar call summaries."
        )

        if algorithm != "DBSCAN":
            num_categories = st.sidebar.slider(
                "Number of Categories (Clusters)",
                min_value=2, max_value=20, value=9,
                help="Set the number of clusters to form."
            )

        if algorithm == "DBSCAN":
            eps_value = st.sidebar.slider(
                "DBSCAN: eps (distance for clustering)",
                min_value=0.1, max_value=5.0, value=0.5, step=0.1,
                help="Maximum distance between two samples for them to be considered as in the same neighborhood."
            )
            min_samples = st.sidebar.slider(
                "DBSCAN: min_samples (minimum points in a cluster)",
                min_value=1, max_value=10, value=5,
                help="Number of samples in a neighborhood for a point to be considered as a core point."
            )

        # Preprocessing function
        def preprocess(text):
            # Language detection and translation
            try:
                if detect(text) != 'en':
                    text = translator.translate(text, dest='en').text
            except:
                pass  # If detection fails, proceed without translation

            # Apply selected preprocessing steps
            if "Lowercase" in preprocessing_options:
                text = text.lower()
            doc = nlp(text)
            tokens = []
            for token in doc:
                if "Remove Punctuation" in preprocessing_options and token.is_punct:
                    continue
                if "Remove Stop Words" in preprocessing_options and (token.is_stop or token.text in custom_stop_words):
                    continue
                if "Lemmatization" in preprocessing_options:
                    tokens.append(token.lemma_)
                else:
                    tokens.append(token.text)
            return ' '.join(tokens)

        # Function to get BERT embeddings
        @st.cache_data
        def get_bert_embeddings(text_list):
            embeddings = []
            for text in text_list:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                outputs = bert_model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy()[0])
            return np.array(embeddings)

        # Initialize progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Start time
        start_time = time.time()

        # Total steps in the processing pipeline
        total_steps = 5
        current_step = 0

        # Step 1: Preprocessing
        status_text.text("Preprocessing data...")
        try:
            df['processed_summary'] = df['summary'].astype(str).apply(preprocess)
            logging.info('Preprocessing completed successfully.')
        except Exception as e:
            st.error(f"An error occurred during preprocessing: {e}")
            logging.error(f"Preprocessing error: {e}")
        # Update progress
        current_step += 1
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / current_step) * total_steps
        time_remaining = estimated_total_time - elapsed_time
        progress = int((current_step / total_steps) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Preprocessing completed. Estimated time remaining: {int(time_remaining)} seconds.")

        # Step 2: Feature Extraction
        status_text.text("Extracting features using BERT embeddings...")
        try:
            df['processed_summary'] = df['processed_summary'].fillna('')
            texts = df['processed_summary'].tolist()
            X = get_bert_embeddings(texts)
            logging.info('Feature extraction completed successfully.')
        except Exception as e:
            st.error(f"An error occurred during feature extraction: {e}")
            logging.error(f"Feature extraction error: {e}")
        # Update progress
        current_step += 1
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / current_step) * total_steps
        time_remaining = estimated_total_time - elapsed_time
        progress = int((current_step / total_steps) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Feature extraction completed. Estimated time remaining: {int(time_remaining)} seconds.")

        # Step 3: Clustering
        status_text.text("Clustering data...")
        try:
            if algorithm == "KMeans":
                model = KMeans(n_clusters=int(num_categories), random_state=42)
                df['category'] = model.fit_predict(X)
            elif algorithm == "Agglomerative Clustering":
                model = AgglomerativeClustering(n_clusters=int(num_categories))
                df['category'] = model.fit_predict(X)
            elif algorithm == "DBSCAN":
                model = DBSCAN(eps=eps_value, min_samples=min_samples)
                df['category'] = model.fit_predict(X)
            logging.info('Clustering completed successfully.')
        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")
            logging.error(f"Clustering error: {e}")
        # Update progress
        current_step += 1
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / current_step) * total_steps
        time_remaining = estimated_total_time - elapsed_time
        progress = int((current_step / total_steps) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Clustering completed. Estimated time remaining: {int(time_remaining)} seconds.")

        # Step 4: Automated Labeling
        status_text.text("Assigning category names based on top terms...")
        try:
            # Since we're using BERT embeddings, we'll use the original text for labeling
            category_names = {}
            for category in sorted(df['category'].unique()):
                if category == -1:
                    category_names[category] = 'Noise'
                    continue
                texts_in_category = df[df['category'] == category]['processed_summary']
                all_words = ' '.join(texts_in_category).split()
                most_common_words = pd.Series(all_words).value_counts().head(5).index.tolist()
                category_name = ', '.join(most_common_words)
                category_names[category] = category_name
                st.write(f"Category {category}: {category_name}")
            logging.info('Automated labeling completed successfully.')
        except Exception as e:
            st.error(f"An error occurred during labeling: {e}")
            logging.error(f"Labeling error: {e}")
        # Map numerical labels to names in the DataFrame
        df['category_name'] = df['category'].map(category_names)

        # Update progress
        current_step += 1
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / current_step) * total_steps
        time_remaining = estimated_total_time - elapsed_time
        progress = int((current_step / total_steps) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Automated labeling completed. Estimated time remaining: {int(time_remaining)} seconds.")

        # Step 5: Visualization
        status_text.text("Generating visualization...")
        try:
            # PCA for dimensionality reduction
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            # Interactive Plotly scatter plot
            st.subheader("Interactive Cluster Visualization (PCA)")
            vis_df = pd.DataFrame({
                'PCA1': X_pca[:, 0],
                'PCA2': X_pca[:, 1],
                'Category': df['category_name'],
                'Summary': df['summary']
            })

            fig = px.scatter(
                vis_df,
                x='PCA1',
                y='PCA2',
                color='Category',
                hover_data=['Summary'],
                title='Clusters of Call Summaries'
            )
            st.plotly_chart(fig)
            logging.info('Visualization generated successfully.')
        except Exception as e:
            st.error(f"An error occurred during visualization: {e}")
            logging.error(f"Visualization error: {e}")

        # Update progress
        current_step += 1
        progress_bar.progress(100)
        status_text.text("Processing complete.")

        # Clear progress bar and status text after completion
        progress_bar.empty()
        status_text.empty()

        # Clustering evaluation metric
        if len(set(df['category'])) > 1:
            score = silhouette_score(X, df['category'])
            st.write(f"**Silhouette Score:** {score:.2f}")
            logging.info(f'Silhouette Score: {score:.2f}')
        else:
            st.write("Silhouette Score: Cannot compute with only one cluster.")
            logging.warning('Silhouette Score could not be computed due to a single cluster.')

        # Editable Category Names
        st.subheader("Edit Category Names")
        if 'category_names' not in st.session_state:
            st.session_state['category_names'] = category_names

        for category in sorted(df['category'].unique()):
            default_name = st.session_state['category_names'].get(category)
            new_name = st.text_input(
                f"Category {category}",
                value=default_name,
                key=f"rename_{category}"
            )
            st.session_state['category_names'][category] = new_name

        # Update the DataFrame with new category names
        df['category_name'] = df['category'].map(st.session_state['category_names'])

        # Display results
        st.subheader("Categorized Data with Named Categories")
        st.write(df[['summary', 'category_name']].head(20))  # Show first 20 rows

        # Word Clouds for each category
        st.subheader("Word Clouds for Each Category")
        for category in sorted(df['category'].unique()):
            texts = df[df['category'] == category]['processed_summary']
            text_combined = ' '.join(texts)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
            st.write(f"Category: {st.session_state['category_names'][category]}")
            st.image(wordcloud.to_array())

        # Download results
        st.markdown("### Download Categorized Data")

        # Download as CSV
        csv = df.to_csv(index=False)
        b64_csv = base64.b64encode(csv.encode()).decode()
        href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="categorized_call_summaries.csv">Download CSV File</a>'
        st.markdown(href_csv, unsafe_allow_html=True)

        # Download as Excel
        towrite = io.BytesIO()
        df.to_excel(towrite, index=False)
        towrite.seek(0)
        st.download_button(
            label="Download Excel",
            data=towrite,
            file_name='categorized_call_summaries.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        # Download as JSON
        json_data = df.to_json(orient='records')
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name='categorized_call_summaries.json',
            mime='application/json'
        )

        # Feedback mechanism
        st.sidebar.subheader("Feedback")
        feedback = st.sidebar.text_area("Please provide your feedback here.")
        if st.sidebar.button("Submit Feedback"):
            try:
                with open('feedback.txt', 'a') as f:
                    f.write(feedback + '\n')
                st.sidebar.success("Thank you for your feedback!")
                logging.info('Feedback submitted.')
            except Exception as e:
                st.sidebar.error(f"An error occurred while submitting feedback: {e}")
                logging.error(f"Feedback submission error: {e}")
else:
    st.info("Awaiting CSV file to be uploaded.")
