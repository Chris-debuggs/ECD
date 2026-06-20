# ECD Internship Learning Journal

# Intern Details
Name:Aadhikesh M S
Project: Early Childhood Development (ECD) System
Duration Covered: Day 1 – Day 3

---

# Day 1 – Git Collaboration & Conflict Resolution

# Objective

Learn Git workflow, branching, commits, pushes, pull requests, and merge conflict handling.

---

# Repository Setup

# Commands Used

```bash
git clone <repository-url>
cd ECD
```

## Why?

To download the project repository and start working locally.

---

# Check Repository Status

# Command

```bash
git status
```

## Why?

Shows modified, staged, and untracked files.

---

# Create New Branch

# Command

```bash
git checkout -b feature-branch
```

## Why?

Creates an isolated workspace for development without affecting the main branch.

---

# Stage Changes

# Command

```bash
git add <file-name>
```

## Why?

Moves changes into Git's staging area.

---

# Commit Changes

# Command

```bash
git commit -m "feat: add new functionality"
```

## Why?

Creates a snapshot of the work completed.

---

# Push Changes

# Command

```bash
git push -u origin feature-branch
```

## Why?

Uploads local commits to GitHub.

---

# Pull Request (PR)

# Purpose

Request review and merge of code changes into the main project.

### Learning

* Branch workflow
* Pull Requests
* Code review process
* Merge conflicts
* Team collaboration

---

# Day 2 – Application Execution & Database Verification

# Objective

Run the application and verify that the database and system components function correctly.

---

# Install Dependencies

# Command

```bash
pip install -r requirements.txt
```

## Why?

Installs all project dependencies required by the application.

---

# Seed Demo Database

# Command

```bash
python seed_demo_db.py
```

## Why?

Populates demo.db with sample data.

---

# Launch Streamlit Application

# Command

```bash
streamlit run app.py
```

## Why?

Starts the ECD application interface for testing.

---

# Verify Application

# Tasks Performed

* Opened Streamlit UI
* Verified dashboard loading
* Verified database connection
* Checked generated outputs
* Confirmed records were processed successfully

---

## Learning Outcomes

* Application deployment workflow
* Dependency management
* Database seeding
* Streamlit application testing
* Data verification

---

# Day 3 – Domain Isolation & Decision Trace Analysis

# Objective

Analyze schema.py and decision_trace.py and extract transition states for ECD-0002 using SQLite.

---

# Analyze schema.py

# Purpose

Understand:

* Database configuration
* Database connection management
* Tier label definitions

# Learning

schema.py acts as the central database utility module.

---

# Analyze decision_trace.py

# Purpose

Understand governance and audit logging.

# Important Functions

* log_event()
* log_screening()
* log_referral()
* log_consent()
* get_trace_log()

# Learning

The system stores screening decisions, referrals, and consent actions inside the decision_trace table.

---

# Verify Database

# Command

```bash
ls *.db
```

# Output

```text
demo.db
```

## Why?

Confirms database availability.

---

# Create Database Explorer

# File Created

```text
db_explorer.py
```

# Purpose

Inspect available database tables.

# Result

```text
children
assessments
risk_results
developmental_trajectories
consents
decision_trace
referrals
```

---

# Create Transition Extraction Script

# File Created

```text
extract_trace.py
```

# Purpose

Extract transition history for:

```text
ECD-0002
```

---

# SQL Query Used

```sql
SELECT *
FROM decision_trace
WHERE child_id = 'ECD-0002';
```

## Why?

Filters governance records belonging to ECD-0002.

---

# Execute Script

# Command

```bash
python extract_trace.py
```

# Result

Successfully displayed:

* Timestamp
* Action
* Risk Tier
* Top Risk Factor
* Notes

for ECD-0002.

---

# Git Workflow Used During Day 3

# Create Branch

```bash
git checkout -b day3-transition-trace
```

## Why?

Creates an isolated branch for the task.

---

# Stage File

```bash
git add extract_trace.py
```

## Why?

Adds the file to the staging area.

---

# Commit

```bash
git commit -m "feat: add transition trace extraction script"
```

## Why?

Creates a tracked version of the completed work.

---

# Verify Remote

```bash
git remote -v
```

## Why?

Checks configured GitHub repository URLs.

---

# Push Branch

```bash
git push -u origin day3-transition-trace
```

## Why?

Uploads the branch to GitHub.

---

# Create Pull Request

# PR Title

```text
Add transition trace extraction script for ECD-0002
```

# Purpose

Submit changes for review and approval.

---

# Git Commit Types Reference

# Feature

```text
feat:
```

Example:

```text
feat: add transition extraction script
```

Used when adding new functionality.

---

# Fix

```text
fix:
```

Example:

```text
fix: correct database query
```

Used for bug fixes.

---

## Documentation

```text
docs:
```

Example:

```text
docs: update learning journal
```

Used for documentation changes.

---

## Refactor

```text
refactor:
```

Example:

```text
refactor: simplify database connection logic
```

Used when improving code structure.

---

## Test

```text
test:
```

Example:

```text
test: add extraction script tests
```

Used when adding or modifying tests.

---

# Overall Learning Summary

During these three days I learned:

* Git and GitHub collaboration workflow
* Branch creation and management
* Pull Requests and code review process
* Merge conflict concepts
* Python project setup
* Dependency installation
* Streamlit application execution
* SQLite database exploration
* Schema and audit trail analysis
* SQL querying
* Governance trace extraction
* Documentation practices
* Commit message standards used in professional software development
