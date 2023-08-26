from flask import Flask, render_template, request, jsonify
import requests
import pickle

app = Flask(__name__)

model_filename = 'classification_gnb.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

prediction_statements = {
    0: "Sorry, You cannot get a placement.",
    1: "Congartulations! You can get a placement."
}

gender_mapping = {'F': 0, 'M': 1}
ssc_b_mapping = {'Central': 0, 'Others': 1}
hsc_b_mapping = {'Central': 0, 'Others': 1}
hsc_s_mapping = {'Arts': 0, 'Commerce': 1, 'Science': 2}
degree_t_mapping = {'Comm&Mgmt': 0, 'Others': 1, 'Sci&Tech': 2}
workex_mapping = {'No': 0, 'Yes': 1}
specialisation_mapping = {'Mkt&Fin': 0, 'Mkt&HR': 1}

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    ssc_p = float(request.form['ssc_p'])
    hsc_p = float(request.form['hsc_p'])
    degree_p = float(request.form['degree_p'])
    etest_p = float(request.form['etest_p'])
    mba_p = float(request.form['mba_p'])
    gender = request.form['gender']
    ssc_b = request.form['ssc_b']
    hsc_b = request.form['hsc_b']
    hsc_s = request.form['hsc_s']
    degree_t = request.form['degree_t']
    workex = request.form['workex']
    specialisation = request.form['specialisation']

    # Convert categorical data to encoded values using mapping dictionaries
    gender_encoded = gender_mapping[gender]
    ssc_b_encoded = ssc_b_mapping[ssc_b]
    hsc_b_encoded = hsc_b_mapping[hsc_b]
    hsc_s_encoded = hsc_s_mapping[hsc_s]
    degree_t_encoded = degree_t_mapping[degree_t]
    workex_encoded = workex_mapping[workex]
    specialisation_encoded = specialisation_mapping[specialisation]

    features = [ssc_p, hsc_p, degree_p, etest_p, mba_p, gender_encoded, ssc_b_encoded, hsc_b_encoded,
                hsc_s_encoded, degree_t_encoded, workex_encoded, specialisation_encoded]

    import requests

    # NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
    API_KEY = "k8UUMJswsw5607F7TaevmyW74_AoCOetZHX0TUzUjX1O"
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    payload_scoring = {"input_data": [{"fields": ["ssc_p","hsc_p","degree_p","etest_p","mba_p","genderLE","ssc_bLE","hsc_bLE","hsc_sLE","degree_tLE","workexLE","specialisationLE"],
                                       "values": [[ssc_p, hsc_p, degree_p, etest_p, mba_p, gender_encoded, ssc_b_encoded, hsc_b_encoded,hsc_s_encoded, degree_t_encoded, workex_encoded, specialisation_encoded]]}]}

    try:
        response = requests.post('https://eu-gb.ml.cloud.ibm.com/ml/v4/deployments/5676a03e-bff1-42bf-82af-06853637c25b/predictions?version=2021-05-01', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
        print("Scoring response")
        print(response.json())
        if response.status_code == 200:
            api_data = response.json()
            if 'predictions' in api_data and len(api_data['predictions']) > 0:
                prediction_data = api_data['predictions'][0]
                if 'fields' in prediction_data and 'values' in prediction_data:
                    prediction_values = prediction_data['values'][0]
                    prediction = prediction_values[0]
                    prediction_statement = prediction_statements.get(prediction, "Unknown")
                    return render_template('input.html', prediction=prediction_statement)
                else:
                    return "Error: Missing 'fields' or 'values' in prediction data."
            else:
                return "Error: No predictions found in API response."
        else:
            return "Error: Unable to get prediction from IBM Cloud."
    except requests.exceptions.RequestException as e:
        return "Error: " + str(e)

if __name__ == '__main__':
    app.run(debug=True)