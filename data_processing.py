
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import fuzz
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
import os
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Requirements dictionary remains the same
job_requirements = {
    'Software Developer / Engineer': {
        'programming_languages': ['java', 'python', 'c', 'javascript'],
        'other_skills': ['.NET', 'react', 'angular', 'databases', 'sql', 'nosql', 'version control systems', 'problem-solving skills', 'software development life cycle', 'communication', 'teamwork'],
        'education': ['bachelor’s degree in computer science', 'related field'],
        'experience': ['previous experience or internships in software development'],
        'optional': ['portfolio', 'github repository', 'certifications like AWS, Microsoft']
    },
    'Data Scientist': {
        'programming_languages': ['python', 'r'],
        'other_skills': ['data analysis tools', 'machine learning frameworks', 'statistics', 'data preprocessing', 'data visualization', 'SQL'],
        'education': ['bachelor’s or master’s degree in data science', 'computer science', 'statistics', 'related field'],
        'experience': ['experience in handling large datasets'],
        'optional': ['project experience', 'analytical skills']
    },
    'Project Manager': {
        'programming_languages': [],
        'other_skills': ['project management tools', 'risk management', 'budgeting', 'scheduling', 'communication', 'leadership', 'problem-solving', 'agile/scrum methodologies'],
        'education': ['bachelor’s degree in business administration', 'management', 'related field'],
        'experience': ['experience in project management'],
        'optional': ['PMP certification', 'track record of delivering projects on time and within budget']
    },
    'UI/UX Designer': {
        'programming_languages': [],
        'other_skills': ['adobe xd', 'sketch', 'figma', 'user-centered design principles', 'wireframes', 'prototypes', 'communication'],
        'education': ['bachelor’s degree in design', 'fine arts', 'related field'],
        'experience': ['experience in UI/UX design'],
        'optional': ['knowledge of HTML/CSS', 'portfolio']
    },
    'DevOps Engineer': {
        'programming_languages': ['python', 'ruby', 'go', 'bash'],
        'other_skills': ['aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'ci/cd pipelines', 'terraform', 'ansible', 'prometheus', 'grafana'],
        'education': ['bachelor’s degree in computer science', 'engineering', 'related field'],
        'experience': ['experience in software development and system operations'],
        'optional': ['AWS Certified DevOps Engineer', 'problem-solving skills']
    },
    'Marketing Specialist': {
        'programming_languages': [],
        'other_skills': ['google analytics', 'seo tools', 'social media platforms', 'content creation', 'communication', 'writing', 'analytical skills'],
        'education': ['bachelor’s degree in marketing', 'communications', 'related field'],
        'experience': ['previous experience or internships in marketing'],
        'optional': ['portfolio of marketing campaigns', 'knowledge of marketing trends and best practices']
    },
    'Human Resources Manager': {
        'programming_languages': [],
        'other_skills': ['interpersonal skills', 'communication', 'hr software', 'workday', 'adp', 'labor laws', 'recruitment', 'onboarding', 'conflict resolution', 'organizational skills'],
        'education': ['bachelor’s degree in human resources', 'business administration', 'related field'],
        'experience': ['previous HR experience'],
        'optional': ['SHRM-CP', 'PHR certifications', 'track record in managing HR functions']
    },
    'Financial Analyst': {
        'programming_languages': [],
        'other_skills': ['financial modeling', 'forecasting', 'analytical skills', 'quantitative skills', 'financial software', 'excel', 'quickbooks', 'accounting principles'],
        'education': ['bachelor’s degree in finance', 'accounting', 'related field'],
        'experience': ['previous experience or internships in finance'],
        'optional': ['CFA certification', 'attention to detail']
    },
    'Customer Support Specialist': {
        'programming_languages': [],
        'other_skills': ['communication', 'interpersonal skills', 'problem-solving', 'customer support software', 'zendesk', 'salesforce', 'patience', 'empathy', 'high-stress situations'],
        'education': ['high school diploma or equivalent; bachelor’s degree preferred'],
        'experience': ['previous experience in customer service or support'],
        'optional': ['multilingual abilities', 'track record of resolving customer issues']
    },
    'Cybersecurity Analyst': {
        'programming_languages': [],
        'other_skills': ['security tools', 'firewalls', 'IDS/IPS', 'network security', 'protocols', 'regulatory standards', 'GDPR', 'HIPAA', 'vulnerability assessments', 'penetration testing'],
        'education': ['bachelor’s degree in cybersecurity', 'computer science', 'related field'],
        'experience': ['previous experience in cybersecurity roles'],
        'optional': ['CISSP', 'CEH certifications', 'attention to detail', 'investigative skills']
    },
    'Field Technician': {
        'programming_languages': [],
        'other_skills': ['mechanical aptitude', 'problem-solving skills', 'communication', 'teamwork'],
        'education': ['high school diploma or equivalent', 'technical certification'],
        'experience': ['previous experience in field work or related technical role'],
        'optional': ['certifications in relevant field']
    },
    'Project Engineer': {
        'programming_languages': [],
        'other_skills': ['project management', 'engineering principles', 'communication', 'teamwork', 'problem-solving skills'],
        'education': ['bachelor’s degree in engineering', 'related field'],
        'experience': ['experience in project management or engineering projects'],
        'optional': ['PMP certification', 'relevant engineering certifications']
    },
    'Senior Project Manager': {
        'programming_languages': [],
        'other_skills': ['advanced project management', 'leadership', 'risk management', 'budgeting', 'communication', 'teamwork'],
        'education': ['bachelor’s degree in engineering', 'project management', 'related field'],
        'experience': ['extensive experience in project management'],
        'optional': ['PMP certification', 'track record of successful project delivery']
    },
    'Chief Operations Officer (COO)': {
        'programming_languages': [],
        'other_skills': ['executive leadership', 'strategic planning', 'financial acumen', 'communication', 'teamwork', 'problem-solving skills'],
        'education': ['bachelor’s degree in business administration', 'engineering', 'related field'],
        'experience': ['significant executive experience in operations management'],
        'optional': ['MBA', 'track record of improving operational efficiency']
    },
    'Operator': {
        'programming_languages': [],
        'other_skills': ['mechanical aptitude', 'equipment operation', 'communication', 'teamwork'],
        'education': ['high school diploma or equivalent'],
        'experience': ['previous experience as an operator or in a similar role'],
        'optional': ['certifications in equipment operation']
    },
    'Electrical Engineer': {
        'programming_languages': [],
        'other_skills': ['electrical engineering principles', 'problem-solving skills', 'communication', 'teamwork'],
        'education': ['bachelor’s degree in electrical engineering', 'related field'],
        'experience': ['experience in electrical engineering'],
        'optional': ['PE license', 'relevant certifications']
    },
    'Mechanical Engineer': {
        'programming_languages': [],
        'other_skills': ['mechanical engineering principles', 'problem-solving skills', 'communication', 'teamwork'],
        'education': ['bachelor’s degree in mechanical engineering', 'related field'],
        'experience': ['experience in mechanical engineering'],
        'optional': ['PE license', 'relevant certifications']
    },
    'Drilling Supervisor': {
        'programming_languages': [],
        'other_skills': ['drilling operations', 'leadership', 'problem-solving skills', 'communication', 'teamwork'],
        'education': ['bachelor’s degree in engineering', 'related field'],
        'experience': ['experience in drilling operations'],
        'optional': ['relevant certifications', 'track record of successful drilling projects']
    },
    'Senior Electrical Engineer': {
        'programming_languages': [],
        'other_skills': ['advanced electrical engineering principles', 'leadership', 'problem-solving skills', 'communication', 'teamwork'],
        'education': ['bachelor’s degree in electrical engineering', 'related field'],
        'experience': ['extensive experience in electrical engineering'],
        'optional': ['PE license', 'relevant certifications']
    },
    'Chief Executive Officer (CEO)': {
        'programming_languages': [],
        'other_skills': ['executive leadership', 'strategic planning', 'financial acumen', 'communication', 'teamwork', 'problem-solving skills'],
        'education': ['bachelor’s degree in business administration', 'engineering', 'related field'],
        'experience': ['significant executive experience in operations management'],
        'optional': ['MBA', 'track record of successful company leadership']
    },
    'Site Assistant': {
        'programming_languages': [],
        'other_skills': ['site management', 'communication', 'teamwork', 'problem-solving skills'],
        'education': ['high school diploma or equivalent'],
        'experience': ['previous experience in site management or related field'],
        'optional': ['certifications in site management']
    },
    'Site Engineer': {
        'programming_languages': [],
        'other_skills': ['site management', 'engineering principles', 'communication', 'teamwork', 'problem-solving skills'],
        'education': ['bachelor’s degree in engineering', 'related field'],
        'experience': ['experience in site engineering'],
        'optional': ['PE license', 'relevant certifications']
    },
    'Senior Site Engineer': {
        'programming_languages': [],
        'other_skills': ['advanced site management', 'engineering principles', 'leadership', 'communication', 'teamwork', 'problem-solving skills'],
        'education': ['bachelor’s degree in engineering', 'related field'],
        'experience': ['extensive experience in site engineering'],
        'optional': ['PE license', 'relevant certifications']
    },
    'Chief Engineering Officer': {
        'programming_languages': [],
        'other_skills': ['executive leadership', 'strategic planning', 'financial acumen', 'communication', 'teamwork', 'problem-solving skills'],
        'education': ['bachelor’s degree in engineering', 'related field'],
        'experience': ['significant executive experience in engineering management'],
        'optional': ['MBA', 'track record of successful engineering projects']
    },
    'Sales Associate': {
        'programming_languages': [],
        'other_skills': ['customer service', 'sales techniques', 'communication', 'interpersonal skills', 'product knowledge'],
        'education': ['high school diploma or equivalent'],
        'experience': ['previous retail or sales experience'],
        'optional': ['multilingual abilities', 'track record of achieving sales targets']
    },
    'Store Manager': {
        'programming_languages': [],
        'other_skills': ['retail management', 'leadership', 'inventory management', 'budgeting', 'customer service', 'communication'],
        'education': ['bachelor’s degree in business administration', 'related field'],
        'experience': ['experience in retail management'],
        'optional': ['track record of improving store performance']
    },
    'Regional Manager': {
        'programming_languages': [],
        'other_skills': ['multi-store management', 'leadership', 'strategic planning', 'budgeting', 'communication', 'teamwork'],
        'education': ['bachelor’s degree in business administration', 'related field'],
        'experience': ['significant experience in retail management'],
        'optional': ['track record of managing multiple locations']
    },
    'Director of Retail Operations': {
        'programming_languages': [],
        'other_skills': ['executive leadership', 'strategic planning', 'financial acumen', 'communication', 'teamwork', 'problem-solving skills'],
        'education': ['bachelor’s degree in business administration', 'related field'],
        'experience': ['extensive experience in retail management'],
        'optional': ['MBA', 'track record of successful retail operations management']
    }
    # same as before
}

general_requirements = {
    'programming_languages': [],
    'other_skills': ['communication', 'problem-solving', 'teamwork', 'adaptability', 'time management'],
    'education': ['bachelor’s degree in any field'],
    'experience': ['previous relevant experience'],
    'optional': []
}

def calculate_match_percentage(candidate_skills, required_skills, general_skills, years_of_experience):
    candidate_skills_set = set(skill.strip().lower() for skill in candidate_skills.split(','))
    required_skills_set = set(skill.strip().lower() for skill in required_skills)

    if required_skills_set:
        match_count = len(candidate_skills_set.intersection(required_skills_set))
        skill_match_percentage = (match_count / len(required_skills_set)) * 100
    else:
        general_skills_set = set(skill.strip().lower() for skill in general_skills)
        match_count = len(candidate_skills_set.intersection(general_skills_set))
        skill_match_percentage = (match_count / len(general_skills_set)) * 100 if general_skills_set else 0

    experience_match_percentage = min(years_of_experience / 10, 100)
    return (skill_match_percentage + experience_match_percentage) / 2

def evaluate_programming_languages(candidate_languages, required_languages, general_languages):
    return calculate_match_percentage(candidate_languages, required_languages, general_languages, 0)

def evaluate_other_skills(candidate_skills, required_skills, general_skills):
    return calculate_match_percentage(candidate_skills, required_skills, general_skills, 0)

def evaluate_education(candidate_education, required_education):
    if any(fuzz.partial_ratio(candidate_education.lower(), edu.lower()) > 80 for edu in required_education):
        return 100
    return 0

def evaluate_experience(candidate_experience, required_experience):
    if any(fuzz.partial_ratio(str(candidate_experience).lower(), exp.lower()) > 80 for exp in required_experience):
        return 100
    return 0

def calculate_overall_match(candidate_row, job_requirements, general_requirements):
    job_title = candidate_row['Job Title You Are Applying For \'If Not Write in Other\'']
    job_specific_requirements = job_requirements.get(job_title, general_requirements)

    programming_languages_match = evaluate_programming_languages(candidate_row['Skillset'], job_specific_requirements['programming_languages'], general_requirements['programming_languages'])
    other_skills_match = evaluate_other_skills(candidate_row['Skillset'], job_specific_requirements['other_skills'], general_requirements['other_skills'])
    education_match = evaluate_education(candidate_row['Education'], job_specific_requirements['education'])
    experience_match = evaluate_experience(candidate_row['Years of Experience'], job_specific_requirements['experience'])

    overall_match = (
        programming_languages_match +
        other_skills_match +
        education_match +
        experience_match
    ) / 4

    return overall_match

def get_latest_file_with_keyword(folder_path, keyword):
    files = glob.glob(os.path.join(folder_path, f"*{keyword}*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No files found with keyword '{keyword}' in folder '{folder_path}'")
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def process_applications(folder_path, job_requirements, general_requirements):
    latest_application_file = get_latest_file_with_keyword(folder_path, "Application")
    df = pd.read_excel(latest_application_file)

    df['Match Percentage'] = df.apply(lambda row: calculate_overall_match(row, job_requirements, general_requirements), axis=1)
    df['Age'] = datetime.datetime.now().year - pd.to_datetime(df['Birth Date'], dayfirst=True).dt.year

    clusters = df.groupby('Job Title You Are Applying For \'If Not Write in Other\'')

    output_file_path = 'sorted_candidates.xlsx'

    with pd.ExcelWriter(output_file_path) as writer:
        for job_title, group in clusters:
            sorted_group = group.sort_values(by='Match Percentage', ascending=False)
            safe_sheet_name = job_title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('[', '_').replace(']', '_')
            sorted_group = sorted_group.drop(columns=['Birth Date'])
            sorted_group.to_excel(writer, sheet_name=safe_sheet_name, index=False)

    print(f"Sorted candidates file created and saved as '{output_file_path}'.")

    return output_file_path

def save_processed_data(df, output_path):
    # Save the DataFrame to an Excel file
    df.to_excel(output_path, index=False, engine='openpyxl')

def update_model(new_hires_folder_path):
    try:
        latest_hiring_file = get_latest_file_with_keyword(new_hires_folder_path, "hired")
    except FileNotFoundError as e:
        print(e)
        return

    hiring_df = pd.read_excel(latest_hiring_file)
    new_hires = hiring_df[hiring_df['Hired'].str.lower() == 'yes']

    if new_hires.empty:
        print("No new hires to update the model.")
        return

    vectorizer = joblib.load('vectorizer.pkl')
    pipeline = joblib.load('regressor_model.pkl')

    new_hires['Age'] = datetime.now().year - pd.to_datetime(new_hires['Birth Date']).dt.year
    new_hires['Match Percentage'] = new_hires.apply(lambda row: calculate_overall_match(row, job_requirements, general_requirements), axis=1)
    new_X = pd.concat([pd.DataFrame(vectorizer.transform(new_hires['Skillset']).toarray(), columns=vectorizer.get_feature_names_out()), new_hires[['Age']]], axis=1)
    new_y = new_hires['Match Percentage']

    original_X = pd.read_pickle('X_train.pkl')
    original_y = pd.read_pickle('y_train.pkl')

    combined_X = pd.concat([original_X, new_X], ignore_index=True)
    combined_y = pd.concat([original_y, new_y], ignore_index=True)

    pipeline.fit(combined_X, combined_y)

    joblib.dump(pipeline, 'updated_regressor_model.pkl')
    joblib.dump(combined_X, 'X_train.pkl')
    joblib.dump(combined_y, 'y_train.pkl')

    print("Model updated with new hiring data and saved.")

def hiring_data_page():
    return render_template('hiring_data_page.html')

def homepage():
    return render_template('homepage.html')

# Webpage interaction functions
def on_homepage_button_click():
    folder_path = r"C:\Users\Lenovo\OneDrive\Documents\Project_folder"
    process_applications(folder_path)
    return redirect(url_for('homepage'))
    # Trigger download of 'sorted_candidates_by_job_title.xlsx'

def on_hiring_data_page_button_click():
    new_hires_folder_path = r"C:\Users\Lenovo\OneDrive\Documents\Hired data"
    update_model(new_hires_folder_path)
    return redirect(url_for('hiring_data_page'))
    # Confirm model update and save

# Example usage:
# Assuming these functions are connected to the respective button clicks in the webpage UI
# on_homepage_button_click()
# on_hiring_data_page_button_click()
