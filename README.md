graph TD
    ML[🤖 Machine Learning] --> SL[🎯 Supervised Learning]
    ML --> UL[🔍 Unsupervised Learning]
    ML --> RL[🎮 Reinforcement Learning]

    %% Supervised Learning Branch
    SL --> C[🏷️ Classification]
    SL --> R[📈 Regression]

    C --> C1[📊 Logistic Regression]
    C --> C2[⚡ Support Vector Machines - SVM]
    C --> C3[🎲 Naïve Bayes]
    C --> C4[🌲 Decision Trees]
    C --> C5[🌴 Random Forest]
    C --> C6[👥 K-Nearest Neighbors - KNN]

    R --> R1[📉 Linear Regression]
    R --> R2[📐 Polynomial Regression]
    R --> R3[🎯 Ridge / Lasso]

    %% Unsupervised Learning Branch
    UL --> CL[🧩 Clustering]
    UL --> DR[📉 Dimensionality Reduction]

    CL --> CL1[📍 K-Means]
    CL --> CL2[🌳 Hierarchical Clustering]
    CL --> CL3[🌌 DBSCAN]
    CL --> CL4[🎯 Mean-Shift]

    DR --> DR1[🧬 PCA]
    DR --> DR2[🗺️ t-SNE]
    DR --> DR3[📊 LDA]

    %% Reinforcement Learning Branch
    RL --> RL1[🎯 Q-Learning]
    RL --> RL2[🔄 SARSA]

