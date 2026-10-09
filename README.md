```text
                                      ┌────────────────────┐
                                      │  MACHINE LEARNING  │
                                      └─────────┬──────────┘
           ┌────────────────────────────────────┼────────────────────────────────────┐
           ▼                                    ▼                                    ▼
┌─────────────────────┐              ┌───────────────────────┐            ┌────────────────────────┐
│ Supervised Learning │              │ Unsupervised Learning │            │ Reinforcement Learning │
└──────────┬──────────┘              └──────────┬────────────┘            └───────────┬────────────┘
     ┌─────┴──────────────┐               ┌─────┴──────────────────┐                  │
     ▼                    ▼               ▼                        ▼                  ▼
┌──────────────┐   ┌────────────┐   ┌────────────┐   ┌──────────────────────────┐  ┌───────────┐
│Classification│   │ Regression │   │ Clustering │   │ Dimensionality Reduction │  │Algorithms │
└──────┬───────┘   └─────┬──────┘   └─────┬──────┘   └────────────┬─────────────┘  └─────┬─────┘
       │                 │                │                       │                      │
       ├─ Logistic Reg.  ├─ Linear Reg.   ├─ K-Means              ├─ PCA                 ├─ Q-Learning
       ├─ SVM            ├─ Polynomial    ├─ Hierarchical         ├─ t-SNE               └─ SARSA
       ├─ Naïve Bayes    └─ Ridge / Lasso ├─ DBSCAN               └─ LDA
       ├─ Decision Trees                  └─ Mean-Shift
       ├─ Random Forest
       └─ KNN

📚 Machine Learning Topics Breakdown1. 🎯 Supervised LearningModels trained on labeled data where inputs are paired with target outputs.Classification (Predicting discrete categorical labels)📊 Logistic Regression: Models the probability of a binary outcome.⚡ Support Vector Machines (SVM): Finds the optimal hyperplane separating data classes.🎲 Naïve Bayes: Probabilistic classifier based on Bayes' theorem assuming feature independence.🌲 Decision Trees: Tree-structured model splitting data based on feature thresholds.🌴 Random Forest: Ensemble of decision trees trained on random subsets of data.👥 K-Nearest Neighbors (KNN): Classifies based on the majority label of the $k$ nearest data points.Regression (Predicting continuous numerical values)📉 Linear Regression: Models linear relationships between inputs and a continuous target variable.📐 Polynomial Regression: Captures non-linear relationships using polynomial terms.🎯 Ridge / Lasso Regression: Regularized linear models preventing overfitting through $L_2$ or $L_1$ penalties.2. 🔍 Unsupervised LearningModels that discover hidden patterns, groupings, or representations in unlabeled data.Clustering (Grouping similar data points together)📍 K-Means: Partitions data into $K$ clusters based on distance to cluster centroids.🌳 Hierarchical Clustering: Builds nested clusters in a tree-like dendrogram structure.🌌 DBSCAN: Density-based clustering capable of discovering arbitrary shapes and identifying noise.🎯 Mean-Shift: Mode-seeking algorithm that shifts cluster centers toward high-density regions.Dimensionality Reduction (Compressing feature space while preserving structure)🧬 Principal Component Analysis (PCA): Projects data onto orthogonal axes of maximum variance.🗺️ t-SNE: Non-linear visualization technique preserving local data neighbor relationships.📊 Linear Discriminant Analysis (LDA): Supervised/unsupervised reduction maximizing class separability.3. 🎮 Reinforcement LearningAgents learning to make sequences of decisions by interacting with an environment to maximize cumulative rewards.🎯 Q-Learning: Model-free, off-policy algorithm learning optimal action-value functions.🔄 SARSA: Model-free, on-policy algorithm updating state-action values based on executed actions.
