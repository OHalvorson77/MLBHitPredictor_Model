# MLBHitPredictor_Model

Author: Owen Halvorson

🎯 Goal
Design a machine learning model that accurately predicts the number of hits in MLB games, with the ultimate goal of identifying profitable opportunities on sportsbooks offering over/under lines that appear to be poorly optimized.

🧠 Strategy
As a dedicated baseball fan, I’ve noticed that the sport is driven by hot and cold streaks — players often perform in cycles. Sportsbooks tend to generalize over/under lines based on a player’s overall reputation or long-term stats, rather than short-term performance trends.

To capitalize on this, I:

Assign heavier weight to recent performance — a struggling star or a surging underdog may offer value if the lines aren’t updated fast enough.

Use multiple rolling averages to capture form trends across entire rosters.

Pull advanced metrics from sources like Statcast (e.g., exit velocity, launch angle, barrel rate) in addition to standard data from the MLB API.

📊 Model Performance Goals
Target MAE (Mean Absolute Error): < 2 hits
This means the model should, on average, predict team hit totals within 2 of the actual result.

🔮 Deployment Plan
Once a reliable model is built:

Compare model predictions with sportsbook over/under lines.

Assume differences between predictions and lines follow a uniform distribution.

Identify bets where the implied probability of the sportsbook line being correct is low.

Surface these opportunities on a front-end dashboard for easy betting analysis.


✅ Results
To validate the model’s practical utility, I tested its predictions against real sportsbook lines over a 10-day simulation:

Each day, I identified the three largest prediction differences (compared to the highest and lowest hit lines offered by various sportsbooks).

I placed simulated $100 bets each day based on the model's strongest deviations from the lines.

As a control, I also placed random bets on over/unders with no model guidance.

Outcome over 10 days:

Model-guided bets: +45% return on investment

Random bets: Near break-even or slightly negative

These results strongly suggest that the model is capable of identifying inefficient lines and turning predictions into profitable strategies.




HOW TO RUN AND TRY ON YOUR OWN

1. Clone this github to a repo on your local machine
2. Run the training_model.py script, alter the start_date and end_date to the period you want to train based on (Can take very long to feature engineer depending on how wide the start and end date are)
3. Find the excel that the training data was stored in
4. Alter completing_model.py script to link to the excel your training data is in
5. Run completing_model.py and it should find the best decision tree classifier parameters and save the model to your local directory
6. Run new_predict.py with your model and the date you are looking to predict for, and it should output recent_load_detailed excel file (games after being feature engineered), and it should output recent_load excel file which is the games with their score and run predictions, with columns showing different betting sites odds too.

















