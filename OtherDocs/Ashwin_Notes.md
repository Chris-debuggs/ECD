# **Workflow and commands used and notes**

# Commit Types

| Type     | Title                    | Description                                                                                           | Emoji |
| -------- | ------------------------ | ----------------------------------------------------------------------------------------------------- | ----- |
| feat     | Features                 | A new feature                                                                                         |  ✨   |
| fix      | Bug Fixes                | A bug fix                                                                                             |  🐛   |
| docs     | Documentation            | Documentation-only changes                                                                            |  📚   |
| style    | Styles                   | Changes that do not affect the meaning of the code (whitespace, formatting, missing semicolons, etc.) |  💎   |
| refactor | Code Refactoring         | A code change that neither fixes a bug nor adds a feature                                             |  📦   |
| perf     | Performance Improvements | A code change that improves performance                                                               |  🚀   |
| test     | Tests                    | Adding missing tests or correcting existing tests                                                     |  🚨   |
| build    | Builds                   | Changes that affect the build system or external dependencies                                         |  🛠   |
| ci       | Continuous Integration   | Changes to CI configuration files and scripts                                                         |  ⚙️    |
| chore    | Chores                   | Other changes that don't modify src or test files                                                     |  ♻️    |
| revert   | Reverts                  | Reverts a previous commit                                                                             |  🗑️   |







# DAY 1: Created a Fork of the main repo and simulated a merge conflict

# Workflow: 

Forked the main repo into my own github acc. Cloned it locally. Named the local repo as Ashwin_Fork. Edited README.md to simulate a merge conflit. 

# Commands:

*cd Desktop/Ashwin_Fork* - To access the repo.

*python -m venv venv* - To create a virtual environment.

*source venv/Scripts/active* - To activate the python environment.

*vim README.md* - To edit the README.md file.

*git add README.md* - Saving the changes of README.md file, so that it can be commited in next commit.

*git commit -m "Changed README.md"* - Commiting the changes.

*git push* - Pushing to repo

# DAY 2: Excecution & State Verification

# Workflow: 

Executed seed_demo_db.py and launched the Steamlit interface via steamlit run app.py and to verify the features of the app.

# Commands:

*cd Desktop/Ashwin_Fork* - To access the repo.

*py -0p* - To list all installed py versions.

*py -3.10 -m venv venv* - Created a virtual environment with py version 3.10.

*source venv/Scripts/active* - To activate the python environment.

*pip install -r requirements.txt* - Successfully installed all project dependencies inside the virtual environment.

*python seed_demo_db.py* - Seeded the database with demo data required by the application.

*streamlit run app.py* - Successfully launched the app.

# Day 3: Domain Isolation & Documentation 

# Workflow:

Analyzed risk_engine.py and trajectory_engine.py. Wrote a standalone script that isolates the XGBoost model (model/risk_model.joblib), bypassing the Streamlit UI entirely, and passes a synthetic NumPy array to trigger a raw inference output.

# Commands:

*cd Desktop/Ashwin_Fork* - To access the repo.

*py -3.10 -m venv venv* - Created a virtual environment with py version 3.10.

*source venv/Scripts/active* - To activate the python environment.

Created a Raw_ouput_testing.py file in repo.

*vim Raw_output_testing.py* - Wrote script to import the model and get a raw prediction output.

*git add Raw_output_testing.py* - To save changes in next commit.

*git commit -m "feat : Added Raw_output_testing.py"* - Commiting the changes.

*git push* - Pushes the changes into repo.




























