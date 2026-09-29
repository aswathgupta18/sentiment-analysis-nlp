# Uploading this project to GitHub

## One-time setup
1. Install Git: https://git-scm.com/downloads
2. Tell Git who you are (use the same email as your GitHub account):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
3. Make a GitHub account at https://github.com if you don't have one.

## Create the empty repo on GitHub
1. Click **+** (top right) then **New repository**.
2. Name it `sentiment-analysis-nlp`.
3. Set it to **Public** (recruiters need to see it).
4. Leave "Add a README", ".gitignore" and "license" **unchecked**, since the project already has them.
5. Click **Create repository**.

## Push the project
Open a terminal inside the project folder, then run:

```bash
git init
git add .
git commit -m "Initial commit: sentiment analysis with TF-IDF and Streamlit"
git branch -M main
git remote add origin https://github.com/<your-username>/sentiment-analysis-nlp.git
git push -u origin main
```

When it asks for a password, GitHub does not accept your account password. Use a **Personal Access Token**:
GitHub > Settings > Developer settings > Personal access tokens > Tokens (classic) > Generate new token, tick the `repo` scope, copy it, and paste it as the password.
(Or install GitHub Desktop, or the `gh` CLI and run `gh auth login`, and skip tokens.)

## Making it look good
- Repo page > the gear next to **About**: add a description ("Real-time review sentiment classifier: NLP + TF-IDF + scikit-learn + Streamlit") and topics: `nlp`, `sentiment-analysis`, `machine-learning`, `scikit-learn`, `streamlit`, `tf-idf`.
- Add the repo link to your resume.

## Optional: free live demo
1. Go to https://share.streamlit.io and sign in with GitHub.
2. New app > pick this repo, branch `main`, main file `app.py` > Deploy.
3. Paste the live link at the top of your README.

## Updating later
```bash
git add .
git commit -m "Describe what you changed"
git push
```

## Before you push: swap in real numbers
Retrain on a real dataset, then update the results table in `README.md` with your own figures. Only put numbers on your resume that you measured yourself.
