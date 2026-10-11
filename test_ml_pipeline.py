import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_available(self):
        self.assertTrue(os.path.exists("Employee_Attrition_Data.csv"))
        data = pd.read_csv("Employee_Attrition_Data.csv")
        self.assertIn("Attrition", data.columns)
        self.assertGreater(len(data), 0)

    def test_model_created(self):
        self.assertTrue(os.path.exists("attrition_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("attrition_model.pkl")

        sample = pd.DataFrame([{
            "Age": 35,
            "MonthlyIncome": 7000,
            "YearsAtCompany": 5,
            "YearsInCurrentRole": 2,
            "JobSatisfaction": 3,
            "WorkLifeBalance": 3,
            "JobInvolvement": 3,
            "PerformanceRating": 3,
            "Overtime": "No",
            "BusinessTravel": "Rarely",
            "Department": "Sales",
            "MaritalStatus": "Married",
            "EducationLevel": "Bachelor",
            "JobLevel": "Mid"
        }])

        prediction = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        self.assertIn(int(prediction), [0, 1])
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)

    def test_high_risk_employee(self):
        model = joblib.load("attrition_model.pkl")

        sample = pd.DataFrame([{
            "Age": 24,
            "MonthlyIncome": 2500,
            "YearsAtCompany": 1,
            "YearsInCurrentRole": 0,
            "JobSatisfaction": 1,
            "WorkLifeBalance": 1,
            "JobInvolvement": 1,
            "PerformanceRating": 3,
            "Overtime": "Yes",
            "BusinessTravel": "Frequently",
            "Department": "Sales",
            "MaritalStatus": "Single",
            "EducationLevel": "High School",
            "JobLevel": "Entry"
        }])

        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 1)

    def test_low_risk_employee(self):
        model = joblib.load("attrition_model.pkl")

        sample = pd.DataFrame([{
            "Age": 50,
            "MonthlyIncome": 15000,
            "YearsAtCompany": 15,
            "YearsInCurrentRole": 8,
            "JobSatisfaction": 4,
            "WorkLifeBalance": 4,
            "JobInvolvement": 4,
            "PerformanceRating": 4,
            "Overtime": "No",
            "BusinessTravel": "Non-Travel",
            "Department": "Technology",
            "MaritalStatus": "Married",
            "EducationLevel": "Master",
            "JobLevel": "Senior"
        }])

        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 0)


if __name__ == "__main__":
    unittest.main()
