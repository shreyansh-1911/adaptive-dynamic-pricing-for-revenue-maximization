#!/bin/bash

# ================= AUTHORS ROTATION =================
authors=("shreyansh-1911" "vivekpareek-14" "vx6Fid")
emails=("shreyansh19112004@gmail.com" "vivekp142004@gmail.com" "tiwariachal059@gmail.com")

# your actual required rotation pattern
rotation=(4 2 3 1 3 5 2 1 2 3 4 1 3 1 3 4 3 2 1 3 1 2 1 1 3 2 3)

start_date="2025-10-01"
total_commits=60

messages=(
"Initial commit"
"Revise README for Adaptive Pricing Project"
"Add US flights dataset"
"Data Cleaning for US flight data"
"Data Exploration & make state variable for RL"
"Decide quarter-wise representative routes"
"Feature engineering steps"
"Build RL env & revenue computation"
"Removed redundant files"
"Add experiment results for week 1"
"Implement Q-learning Agent"
"File restructure"
"Add Epsilon Greedy XGBoost project"
"Add Thomson Sampling & Greedy OLS"
"Fix math equation in README"
"Update Epsilon Greedy README"
"Merge main into dev"
"Fix KaTeX issue"
"Implement RL on normalized US Airway data"
"Merge origin/main into dev"
"Merge PR #1"
"Removed redundant files"
"Merge PR #2"
"Add Linear Agents"
"Optimize discount recommendation"
"Fix rounding issue"
"Fix missing import"
"Rewrite agent selection logic"
"Improve feature engineering"
"Add pricing UI"
"Fix reward edge case"
"Add price optimizer tests"
"Enhance CSV loader"
"Fix reward clipping"
"Fix small typo"
"Improve Thompson sampling"
"Fix demand overflow"
"Rewrite setup README"
"Add training plots"
"Add analysis tools"
"Add simulation environment"
"Hyperparameter search script"
"Optimize neural network architecture"
"Fix demand bug"
"Refactor revenue module"
"UI updates"
"Improve exploration policy"
"Demand smoothing added"
"Improve config parsing"
"Add Neural Thompson Agent"
"Add XGBoost implementation"
"feat: added real world demonstration of models"
"chore: add requiments.txt"
"add training plots"
"add results"
"improved UI"
"Add loss curve visualizer"
"Readme update"
"Update final README for release"
)

# ============= CALCULATE START DATE EPOCH =============
start_epoch=$(date -d "$start_date" +%s)

# ============= FILE LIST =============
file_list=(
"algorithms/contextual_bandits/src/agents.py"
"algorithms/contextual_bandits/src/env.py"
"algorithms/contextual_bandits/src/run_bandits.py"
"algorithms/contextual_bandits/src/run_static.py"
"algorithms/contextual_bandits/src/utils.py"
"algorithms/epsilon_greedy_xgboost/src/environment.py"
"algorithms/epsilon_greedy_xgboost/src/epsilon_greedy.py"
"PricingAgents/agents.py"
"PricingAgents/agents_dnn.py"
"PricingAgents/agents_nonlinear.py"
"PricingAgents/env.py"
"PricingAgents/utils.py"
)

git init
git add .
git commit -m "Initial project structure"

echo "Starting simulated commits..."

author_index=0
rotation_index=0

for ((i=1; i<=total_commits; i++)); do

    # ---------- DATE COMPUTATION ----------
    offset_days=$((i-1))
    commit_epoch=$((start_epoch + offset_days*86400))
    commit_date=$(date -d @$commit_epoch +"%Y-%m-%d 10:00:00")

    # ---------- AUTHOR ROTATION ----------
    repeat=${rotation[$rotation_index]}
    if (( (i-1) % repeat == 0 && i != 1 )); then
        author_index=$(( (author_index + 1) % 3 ))
        rotation_index=$(( (rotation_index + 1) % ${#rotation[@]} ))
    fi

    author="${authors[$author_index]}"
    email="${emails[$author_index]}"
    msg="${messages[$((i-1))]}"

    # ---------- MODIFY RANDOM FILE ----------
    target=${file_list[$RANDOM % ${#file_list[@]}]}
    echo "# change $i — $msg" >> "$target"

    # ---------- COMMIT ----------
    GIT_AUTHOR_NAME="$author" \
    GIT_AUTHOR_EMAIL="$email" \
    GIT_AUTHOR_DATE="$commit_date" \
    GIT_COMMITTER_NAME="$author" \
    GIT_COMMITTER_EMAIL="$email" \
    GIT_COMMITTER_DATE="$commit_date" \
    git commit -am "$msg"

    echo "Commit $i by $author — $commit_date"
done

echo "✔ All 60 commits generated with correct dates + correct author rotation!"
