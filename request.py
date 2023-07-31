import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'insured_sex': 0, 'insured_occupation': 3,
                  'insured_hobbies': 2, 'capital_gains': 38400, 'capital_loss': 0,
                             'incident_type': 2, 'collision_type': 1, 'incident_severity': 2,
                             'authorities_contacted': 4, 'incident_hour_of_the_day': 22, 'number_of_vehicles_involved': 1,
                             'witnesses': 2, 'total_claim_amount': 87010, 'age_group': 5, 'months_as_customer_groups': 3,
                             'policy_annual_premium_groups': 2})

url = 'http://localhost:5000/results'
r = requests.post(url, json={'ApplicantIncome': 2000, 'CoaplicantIncome': 1000, 'LoanAmount': 400,
                             'Loan_Amount_Term': 240, 'Credit_History': 1, 'Self_Employed': 1, 'Dependents': 1,
                             'Property_Area': 1, 'Married': 1, 'Education': 1, 'Gender': 1})
print(r.json())
