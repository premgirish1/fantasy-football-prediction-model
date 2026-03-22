I built this to help me make better start/sit decisions for my own fantasy team. It pulls real NFL data, engineers features, trains two ML models, and gives you weekly projections through an interactive dashboard.
What it does
You enter your roster (RBs and WRs), and the model predicts how many PPR points each player is projected to score that week then ranks them as START or SIT.
How it works

Pulls NFL weekly data from 2018–2024 using nfl_data_py
Engineers features like rolling averages, lag stats, defensive matchup strength, and injury-adjusted target share
Trains a Ridge regression model and a Random Forest model
Evaluates on 2023–2024 seasons (data the model never saw)
Serves predictions through a Streamlit dashboard

Results

MAE: ~5.0 fantasy points
Start/Sit Accuracy: ~75%
Key finding: model architecture mattered less than feature quality — switching from Ridge to Random Forest barely moved the needle, but adding better features did

How to run it

bashpip install -r requirements.txt

python nfl_model.py 

streamlit run dashboard.py  