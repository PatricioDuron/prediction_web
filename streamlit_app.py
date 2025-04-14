import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import base64

# Set page configuration
st.set_page_config(
    page_title="Conseiller d'orientation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Utility function to find resources
def resource_find(filename):
    # First try the current directory
    if os.path.exists(filename):
        return filename
    
    # Try resource directories (you can add more paths as needed)
    resource_dirs = [".", "resources", "assets"]
    for directory in resource_dirs:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    
    # Return the original filename if not found (will cause error later)
    return filename

# Function to display horizontal bar chart of predictions
def display_career_chart(predictions, translation_map=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Sort and get top 3 predictions
    sorted_careers = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
    careers, probabilities = zip(*sorted_careers)
    
    # Translate if a map is given
    if translation_map:
        careers = [translation_map.get(c, c) for c in careers]
    
    # Create plot
    bars = ax.barh(careers, probabilities, color='#1f77b4')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width*100:.1f}%', va='center', ha='left', fontsize=12)
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilité')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    ax.tick_params(axis='x', colors='#666666')
    ax.tick_params(axis='y', colors='#666666')
    ax.set_facecolor('#f5f5f5')
    fig.patch.set_facecolor('#ffffff')
    fig.tight_layout()
    
    return fig

# Function to get PDF download link
def get_pdf_download_link(pdf_path, filename="Presentation.pdf"):
    with open(pdf_path, "rb") as file:
        pdf_bytes = file.read()
    
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Télécharger le PDF</a>'

# Define field translations
field_translation = {
    "Engineering": "Ingénierie",
    "Chemistry": "Chimie",
    "Physics": "Physique",
    "Law": "Droit",
    "Marketing": "Marketing",
    "Medicine": "Médecine",
    "Computer Science": "Informatique",
    "Education": "Éducation",
    "Business": "Affaires",
    "Architecture": "Architecture",
    "Music": "Musique",
    "Biology": "Biologie",
    "Finance": "Finance",
    "Psychology": "Psychologie",
    "Art": "Art"
}

# Define career translations
career_translation = {
    "AI Researcher": "Chercheur en IA",
    "Accountant": "Comptable",
    "Acoustics Specialist": "Spécialiste de l'acoustique",
    "Actuary": "Actuaire",
    "Advertising Manager": "Responsable de la publicité",
    "Aerospace Engineer": "Ingénieur aérospatial",
    "Analytical Chemist": "Chimiste analytique",
    "Animator": "Animateur 2D/3D",
    "Architect": "Architecte",
    "Architectural Technologist": "Technologue en architecture",
    "Art Director": "Directeur artistique",
    "Art Therapist": "Art-thérapeute",
    "Artist": "Artiste",
    "Astronomer": "Astronome",
    "Biochemist": "Biochimiste",
    "Biologist": "Biologiste",
    "Biomedical Engineer": "Ingénieur biomédical",
    "Biotechnologist": "Biotechnologiste",
    "Brand Manager": "Chef de marque",
    "Chemical Engineer": "Ingénieur chimiste",
    "Chemist": "Chimiste",
    "Civil Engineer": "Ingénieur civil",
    "Clinical Psychologist": "Psychologue clinicien",
    "Composer": "Compositeur",
    "Conductor": "Chef d'orchestre",
    "Construction Manager": "Chef de chantier",
    "Counselor": "Conseiller d'orientation",
    "Credit Analyst": "Analyste de crédit",
    "Curriculum Developer": "Concepteur pédagogique",
    "Cybersecurity Analyst": "Analyste en cybersécurité",
    "Data Scientist": "Scientifique des données",
    "Dentist": "Dentiste",
    "Digital Marketing Specialist": "Spécialiste du marketing numérique",
    "Doctor": "Médecin",
    "Ecologist": "Écologiste",
    "Education Administrator": "Administrateur scolaire",
    "Electrical Engineer": "Ingénieur électricien",
    "Entrepreneur": "Entrepreneur",
    "Financial Advisor": "Conseiller financier",
    "Financial Analyst": "Analyste financier",
    "Financial Controller": "Contrôleur financier",
    "Fluid Mechanics Engineer": "Ingénieur en mécanique des fluides",
    "Forensic Psychologist": "Psychologue judiciaire",
    "Game Developer": "Développeur de jeux vidéo",
    "Geneticist": "Généticien",
    "Graphic Designer": "Graphiste",
    "Human Resources Specialist": "Spécialiste RH",
    "Illustrator": "Illustrateur",
    "Industrial-Organizational Psychologist": "Psychologue du travail",
    "Inorganic Chemist": "Chimiste inorganicien",
    "Interior Designer": "Décorateur d'intérieur",
    "Investment Banker": "Banquier d'investissement",
    "Judge": "Juge",
    "Landscape Architect": "Architecte paysagiste",
    "Lawyer": "Avocat",
    "Legal Analyst": "Analyste juridique",
    "Legal Consultant": "Conseiller juridique",
    "Legal Secretary": "Secrétaire juridique",
    "Manager": "Gestionnaire",
    "Market Research Analyst": "Analyste en études de marché",
    "Marketing Manager": "Responsable marketing",
    "Marketing Specialist": "Spécialiste marketing",
    "Mechanical Engineer": "Ingénieur mécanique",
    "Microbiologist": "Microbiologiste",
    "Music Teacher": "Professeur de musique",
    "Music Therapist": "Musicothérapeute",
    "Musician": "Musicien",
    "Nuclear Physicist": "Physicien nucléaire",
    "Nurse": "Infirmier / Infirmière",
    "Organic Chemist": "Chimiste organicien",
    "Paralegal": "Technicien juridique",
    "Pharmacist": "Pharmacien",
    "Physical Chemist": "Chimiste physique",
    "Physician Assistant": "Assistant médical",
    "Physicist": "Physicien",
    "Principal": "Directeur d'école",
    "Psychologist": "Psychologue",
    "Quantum Physicist": "Physicien quantique",
    "Risk Analyst": "Analyste de risques",
    "School Counselor": "Conseiller scolaire",
    "School Psychologist": "Psychologue scolaire",
    "Social Media Manager": "Responsable des réseaux sociaux",
    "Software Developer": "Développeur logiciel",
    "Sound Engineer": "Ingénieur du son",
    "Special Education Teacher": "Enseignant en éducation spécialisée",
    "Surgeon": "Chirurgien",
    "Teacher": "Enseignant",
    "Urban Planner": "Urbaniste",
    "Web Developer": "Développeur web",
    "Zoologist": "Zoologiste"
}

# Define questions
try:
    csv_path = resource_find("ml_data.csv")
    df = pd.read_csv(csv_path)
    field_options = sorted(
        [field_translation.get(field, field) for field in np.unique(df["Field"])]
    )
except Exception as e:
    st.warning(f"Could not load field options: {str(e)}")
    field_options = list(field_translation.values())

# Define questions
questions = [
    {"column": "Field", "text": "Quel est ton domaine d'études ?", "type": "dropdown", "options": field_options},
    {"column": "GPA", "text": "Quel est ton GPA actuel ? (0-10)", "type": "slider", "min": 0.0, "max": 10.0, "step": 0.1, "round": 'float'},
    {"column": 'Extracurricular_Activities', 'text': 'Participes-tu à des activités parascolaires (combien) ?', 'type': 'slider',
     'min': 0, 'max': 10, "step": 1, 'round': 'int'},
    {"column": "Internships", "text": "Combien de stages as-tu faits ?", "type": "slider", "min": 0, "max": 5, "step": 1, "round": "int"},
    {"column": "Projects", "text": "Combien de projets as-tu réalisés ?", "type": "slider", "min": 0, "max": 10, "step": 1, "round": "int"},
    {"column": "Leadership_Positions", "text": "Combien de rôles de leader as-tu occupés ?", "type": "slider", "min": 0, "max": 5, "step": 1, "round": "int"},
    {"column": "Field_Specific_Courses", "text": "Combien de cours spécialisés as-tu suivis ? (nombre)", "type": "slider", "min": 0, "max": 30, "step": 1, "round": 'int'},
    {"column": "Research_Experience", "text": "Combien d'expériences de recherche as-tu eues ?", "type": 'slider', 'min': 0, 'max': 5, "step": 1, 'round': 'int'},
    {"column": "Coding_Skills", "text": "Évalue tes compétences en codage (0-10)", 'type': 'slider', 'min': 0, 'max': 10, "step": 1, 'round': 'int'},
    {"column": "Communication_Skills", "text": "Évalue tes compétences en communication (0-10)", 'type': 'slider', 'min': 0, 'max': 10, "step": 1, 'round': 'int'},
    {"column": "Problem_Solving_Skills", "text": "Évalue tes compétences en résolution de problèmes (0-10)", 'type': 'slider', 'min': 0, 'max': 10, "step": 1, 'round': 'int'},
    {"column": "Teamwork_Skills", "text": "Évalue ton esprit d'équipe (0-10)", 'type': 'slider', 'min': 0, 'max': 10, "step": 1, 'round': 'int'},
    {"column": "Analytical_Skills", "text": "Évalue ta capacité analytique (0-10)", 'type': 'slider', 'min': 0, 'max': 10, "step": 1, 'round': 'int'},
    {"column": "Presentation_Skills", "text": "Évalue ta capacité à faire des présentations (0-10)", 'type': 'slider', 'min': 0, 'max': 10, "step": 1, 'round': 'int'},
    {"column": "Networking_Skills", "text": "Évalue ton réseau et ta capacité à te connecter (0-10)", 'type': 'slider', 'min': 0, 'max': 10, "step": 1, 'round': 'int'},
    {"column": "Industry_Certifications", "text": "Combien de certifications possèdes-tu ? (nombre)", 'type': 'slider', 'min': 0, 'max': 30, "step": 1, 'round': 'int'}
]

# Initialize session state for form responses
if 'responses' not in st.session_state:
    st.session_state.responses = {}

if 'predictions' not in st.session_state:
    st.session_state.predictions = None

if 'group_name' not in st.session_state:
    st.session_state.group_name = None

# Function to convert GPA to tier
def gpa_to_tier(gpa):
    if gpa >= 3.5:
        return 'Excellent'
    elif gpa >= 3.0:
        return 'Good'
    elif gpa >= 2.5:
        return 'Average'
    else:
        return 'Below Average'

# Function to make predictions
def predict_career():
    responses = st.session_state.responses
    
    # Build input data
    input_data = responses.copy()
    user_df = pd.DataFrame([input_data])
    
    # Derived features
    user_df["GPA_Tier"] = user_df["GPA"].apply(gpa_to_tier)
    user_df["SoftSkills"] = user_df[["Teamwork_Skills", "Communication_Skills", "Presentation_Skills"]].mean(axis=1)
    user_df["HardSkills"] = user_df[["Coding_Skills", "Research_Experience", "Projects", "Field_Specific_Courses"]].mean(axis=1)
    user_df["Engagement"] = user_df[["Internships", "Leadership_Positions", "Extracurricular_Activities"]].mean(axis=1)
    user_df["Academic"] = user_df[["GPA", "Field_Specific_Courses"]].mean(axis=1)
    user_df["LeadershipComposite"] = user_df[["Leadership_Positions", "Presentation_Skills"]].mean(axis=1)
    user_df["AnalyticalThinking"] = user_df[["Problem_Solving_Skills", "Analytical_Skills"]].mean(axis=1)
    user_df["CommunicationComposite"] = user_df[["Communication_Skills", "Presentation_Skills", "Networking_Skills"]].mean(axis=1)
    user_df["InitiativeScore"] = user_df[["Projects", "Internships", "Research_Experience"]].mean(axis=1)
    user_df["SocialCapital"] = user_df[["Teamwork_Skills", "Networking_Skills", "Extracurricular_Activities"]].mean(axis=1)
    user_df["GPA_Internship_Impact"] = user_df["GPA"] * user_df["Internships"]
    user_df["SoftSkill_Execution"] = user_df["SoftSkills"] * user_df["Projects"]
    
    # Load helpers and models
    helper_path = resource_find("f_model_helpers.pkl")
    helper = joblib.load(open(helper_path, "rb"))
    
    preprocessor = helper["preprocessor"]
    group_encoder = helper["group_encoder"]
    career_encoders = helper["career_encoders"]
    
    group_model_path = resource_find("f_group_model.h5")
    group_model = load_model(group_model_path)
    
    # Transform and predict group
    X_user = preprocessor.transform(user_df)
    group_pred = group_model.predict(X_user)
    group_name = group_encoder.inverse_transform([np.argmax(group_pred)])[0]
    
    # Load career model
    career_model_path = resource_find(f"f_career_model_{group_name}.h5")
    career_model = load_model(career_model_path)
    career_encoder = career_encoders[group_name]
    
    # Predict career
    career_preds = career_model.predict(X_user)
    top_3_indices = np.argsort(career_preds[0])[::-1][:3]
    top_3_careers = career_encoder.inverse_transform(top_3_indices)
    top_3_confidences = career_preds[0][top_3_indices]
    
    # Convert to dictionary
    predictions_dict = dict(zip(top_3_careers, top_3_confidences))
    
    st.session_state.predictions = predictions_dict
    st.session_state.group_name = group_name

# Create tabs
tab1, tab2 = st.tabs(["Modèle", "Sur l'appli"])

# Tab 1: Prediction Model
with tab1:
    # Title with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            logo_path = resource_find('logo.png')
            st.image(logo_path, width=300)
        except:
            st.title("Conseiller d'orientation")
    
    # Form for user input
    with st.form("career_prediction_form"):
        st.subheader("Complète le formulaire pour prédire ta carrière idéale")
        
        # Create form fields
        for q in questions:
            if q["type"] == "slider":
                st.session_state.responses[q["column"]] = st.slider(
                    q["text"], 
                    min_value=q["min"], 
                    max_value=q["max"],
                    step=q["step"],
                    key=f"slider_{q['column']}"
                )
            elif q["type"] == "dropdown":
                selected = st.selectbox(
                    q["text"],
                    options=["Choisir..."] + q["options"],
                    key=f"dropdown_{q['column']}"
                )
                if selected != "Choisir...":
                    st.session_state.responses[q["column"]] = selected
        
        # Submit button
        submit_button = st.form_submit_button("Soumettre")
        if submit_button:
            # Check for missing responses
            missing_columns = []
            for q in questions:
                column = q["column"]
                if column not in st.session_state.responses or st.session_state.responses[column] == "" or st.session_state.responses[column] == "Choisir...":
                    missing_columns.append(column)
            
            if missing_columns:
                missing_fields = ", ".join(missing_columns)
                st.error(f"Veuillez remplir les champs suivants : {missing_fields}")
            else:
                try:
                    predict_career()
                except Exception as e:
                    st.error(f"Erreur dans l'analyse: {str(e)}")
    
    # Display results
    if st.session_state.predictions:
        st.subheader("Résultats de la prédiction")
        
        # Show group name
        st.write(f"**Groupe de carrières prévu**: {st.session_state.group_name}")
        
        # Display chart
        st.pyplot(display_career_chart(st.session_state.predictions, translation_map=career_translation))
        
        # Display detailed results
        st.write("**Top 3 Carrières:**")
        for i, (career, confidence) in enumerate(sorted(st.session_state.predictions.items(), key=lambda x: x[1], reverse=True), 1):
            career_fr = career_translation.get(career, career)
            st.write(f"{i}. {career_fr} — {confidence*100:.2f}%")

# Tab 2: About
with tab2:
    st.title("Sur l'application")
    
    explanatory_text = """
    ## Pourquoi on a décidé de faire cela ?

    De nos jours, il y a beaucoup des adolescents qui n'ont aucune idée de leur futur (métier), étant à peu près 55% des jeunes de 14 à 20 ans. Et maintenant, il y a une hausse de 10% par rapport à 2022. Cette problème peut s'étendre en leur donnant du stress, en baissant leur motivation et compliquant leur prise de décision.

    En voyant ceux-ci, on voulait investiguer pour leur montrer des solutions. Finalement, on a fini par entendre qu'ils peuvent analyser des sources gouvernementales et vérifiées ou faire des questionnaires pour mieux savoir son choix.

    Mais celui-ci n'était pas suffisant, par conséquent on a décidé de leur faire une application qui pourrait prédire la carrière la plus appropriée pour eux.

    En conclusion, notre sensibilité en voyant cette problématique dans laquelle les adolescents ne savent pas quoi faire et notre manque de satisfaction dans la recherche nous ont mené à créer cette application.
    """
    
    st.markdown(explanatory_text)
    
    # PDF Download
    try:
        pdf_path = resource_find("Test.pdf")
        st.markdown(get_pdf_download_link(pdf_path), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur de chargement du PDF: {str(e)}")

# Apply custom styling
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: white;
    }
    .stButton>button {
        background-color: #333;
        color: white;
        border: 1px solid #444;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #444;
    }
    .stSelectbox, .stSlider {
        background-color: #1e1e1e;
        border-radius: 5px;
        padding: 5px;
    }
    h1, h2, h3 {
        color: white;
    }
    .stTab {
        background-color: #1e1e1e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    pass  # This is already being run by Streamlit