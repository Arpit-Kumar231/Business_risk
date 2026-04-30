# AI Agent-Based Classifier - Novel Approach Documentation

## Overview

The **AI Agent-Based Classifier** is a novel predictive model that combines **intelligent reasoning** with **multi-strategy decision making** to predict business risk. Unlike traditional ML models that learn correlations, this agent uses **sub-agents** that reason about features independently and combine their insights.

## What Makes It Novel?

### Traditional ML Models (XGBoost, Random Forest, etc.)
- Learn patterns from data through optimization
- Work as "black boxes" - hard to understand why they make predictions
- Single unified learning mechanism

### AI Agent Classifier ✨ (Novel)
- **5 Independent Sub-Agents** each using different reasoning logic
- **Interpretable**: You can see which agents influence the decision
- **Hybrid Approach**: Combines statistical reasoning with feature analysis
- **Adaptive**: Automatically adjusts decision threshold based on training data
- **Explainable**: Can show you the reasoning for each prediction

## The 5 Sub-Agents

### 1️⃣ Anomaly Detection Agent
**How it works:** Calculates Z-scores for features to detect statistical outliers
**Logic:** Outlier companies are riskier than normal ones
**Output:** Anomaly score (0-1)

### 2️⃣ Pattern Recognition Agent
**How it works:** Weights features by their importance and looks for risky combinations
**Logic:** Important features with extreme values = high risk
**Output:** Pattern score (0-1)

### 3️⃣ Weighted Scoring Agent
**How it works:** Uses percentile-based scoring weighted by feature importance
**Logic:** High-importance features with high percentile values indicate risk
**Output:** Weighted score (0-1)

### 4️⃣ Outlier Detection Agent
**How it works:** Uses IQR (Interquartile Range) method to identify outliers
**Logic:** Companies outside expected ranges are outliers and potentially risky
**Output:** Outlier score (0-1)

### 5️⃣ Distribution Analysis Agent
**How it works:** Analyzes how extreme values are in the feature distribution
**Logic:** Extreme values in important features = high risk
**Output:** Distribution score (0-1)

## How It Works

### Training Phase
```
1. Learn feature statistics (mean, std)
2. Calculate feature importance (correlation with target)
3. Find optimal decision threshold using F1-Score
```

### Prediction Phase
```
1. Each sub-agent analyzes the sample independently
2. Generate scores: [0 to 1] for each agent
3. Combine scores based on reasoning depth:
   - Shallow: Average of top 3 agents
   - Medium: Equal weighted average of all 5
   - Deep: Weighted average with priority weighting
4. Normalize to probability [0, 1]
5. Compare against adaptive threshold
```

## Reasoning Depths

### Shallow Reasoning
- Uses top 3 agents (anomaly, pattern, weighted)
- Fast execution
- Good for quick decisions
- Average combination: `(anomaly + pattern + weighted) / 3`

### Medium Reasoning (Default)
- Uses all 5 agents equally
- Balanced speed and accuracy
- Best for most scenarios
- Average combination: `(all 5 agents) / 5`

### Deep Reasoning
- Uses weighted combination of all 5 agents
- More sophisticated analysis
- Better for complex decisions
- Weighted combination:
  - Anomaly: 25%
  - Pattern: 25%
  - Weighted: 20%
  - Outlier: 20%
  - Distribution: 10%

## Usage Examples

### Basic Usage
```python
from src.agent_model import AgentBasedClassifier

# Create and train agent
agent = AgentBasedClassifier(reasoning_depth='medium', adaptive=True)
agent.fit(X_train, y_train)

# Make predictions
predictions = agent.predict(X_test)
probabilities = agent.predict_proba(X_test)
```

### Get Agent Reasoning
```python
# Understand why the agent made a prediction
reasoning = agent.get_agent_reasoning(X_test, sample_idx=0)

print(f"Anomaly Score: {reasoning['anomaly_score']:.4f}")
print(f"Pattern Score: {reasoning['pattern_score']:.4f}")
print(f"Weighted Score: {reasoning['weighted_score']:.4f}")
print(f"Outlier Score: {reasoning['outlier_score']:.4f}")
print(f"Distribution Score: {reasoning['distribution_score']:.4f}")
print(f"Final Prediction: {reasoning['final_prediction']:.4f}")
```

### Integrated in Pipeline
```python
from src.models import train_evaluate_models

# Agent automatically trains alongside other models
results_df, models_dict = train_evaluate_models(
    X_train, y_train, X_test, y_test, return_models=True
)

# AI Agent is one of the trained models
agent_model = models_dict["AI Agent"]
print(results_df)  # Shows AI Agent alongside XGBoost, Random Forest, etc.
```

## Output Format

When you run `python main.py`, the agent generates:

```
=================================================================
AI AGENT-BASED CLASSIFIER - NOVEL REASONING APPROACH
=================================================================

NOVELTY: This model uses intelligent reasoning combining 5 sub-agents:
  1. Anomaly Detection Agent - Identifies statistical outliers
  2. Pattern Recognition Agent - Recognizes risky feature combinations
  3. Weighted Scoring Agent - Uses feature importance for scoring
  4. Outlier Detection Agent - IQR-based outlier analysis
  5. Distribution Analysis Agent - Analyzes feature extremeness

Example: AI Agent Reasoning for Sample 0:
  Anomaly Detection Score: 0.1234
  Pattern Recognition Score: 0.4567
  Weighted Scoring Score: 0.3456
  Outlier Detection Score: 0.2345
  Distribution Analysis Score: 0.5678
  ➜ Final Risk Prediction: 0.3654
  Reasoning Depth: deep
  Adaptive Threshold: 0.4823
```

## Interpreting Results

### High Score (0.7-1.0)
- Multiple agents detecting risk patterns
- Company exhibits multiple risk indicators
- Recommendation: Investigate further

### Medium Score (0.4-0.7)
- Mixed signals from different agents
- Some risk indicators present
- Recommendation: Monitor closely

### Low Score (0.0-0.3)
- Minimal risk patterns detected
- Company behaves like typical low-risk profile
- Recommendation: Standard procedures

## Comparison with Traditional Models

| Aspect | XGBoost/Random Forest | AI Agent |
|--------|----------------------|----------|
| **How It Works** | Learns from data patterns | Reasons about features |
| **Interpretability** | Black box | Transparent (5 agents) |
| **Speed** | Very fast | Fast |
| **Explainability** | SHAP/Feature importance | Sub-agent breakdown |
| **Adaptability** | Fixed after training | Adaptive threshold |
| **Novelty** | Established | Novel approach |

## Advantages of AI Agent

✅ **Interpretable**: See which agents influence decisions
✅ **Explainable**: Understand the reasoning for each prediction
✅ **Hybrid**: Combines multiple analytical perspectives
✅ **Adaptive**: Learns optimal threshold from training data
✅ **Flexible**: Can adjust reasoning depth based on needs
✅ **Novel**: Different from traditional ML approaches
✅ **Fast**: Efficient compared to deep learning

## Disadvantages to Consider

❌ May not capture complex non-linear relationships
❌ Relies on hand-crafted reasoning strategies
❌ Less data-driven than modern ML models
❌ Performance may trail top performers on complex tasks

## When to Use AI Agent

### ✓ Use when:
- You need explainability and interpretability
- You want to understand WHY a company is risky
- You have domain knowledge about risk indicators
- You need transparent decision-making for compliance
- You want a novel approach for research/novelty

### ✗ Don't use when:
- Maximum accuracy is the only goal
- You have very large datasets with complex patterns
- You need state-of-the-art performance
- Your data has many non-linear relationships

## Integration in Your Pipeline

The AI Agent is fully integrated into your business risk assessment pipeline:

```
main.py
├── Load Data (data_loader.py)
├── Preprocess (preprocessing.py)
├── Train Models (models.py)
│   ├── Random Forest
│   ├── Gradient Boosting
│   ├── XGBoost
│   ├── MLP (Deep Learning)
│   └── AI Agent ⭐ (Novel)
├── Evaluate All (metrics)
├── Visualize Comparison
├── SHAP Explainability (XGBoost)
└── AI Agent Explainability ⭐ (Reasoning breakdown)
```

## Experimental Variations

You can experiment with different configurations:

```python
# Shallow reasoning - quick and simple
agent_shallow = AgentBasedClassifier(reasoning_depth='shallow')

# Medium reasoning - balanced (default)
agent_medium = AgentBasedClassifier(reasoning_depth='medium')

# Deep reasoning - sophisticated analysis
agent_deep = AgentBasedClassifier(reasoning_depth='deep')

# Without adaptive thresholding
agent_fixed = AgentBasedClassifier(adaptive=False)  # Uses 0.5 threshold

# With more sub-agents (if modified)
agent_more = AgentBasedClassifier(n_sub_agents=7)
```

## Feature Importance in Agent

The agent learns which features are most important during training:

```python
agent.fit(X_train, y_train)
print(agent.feature_weights)  # Importance scores for each feature
print(agent.feature_means)    # Mean values during training
print(agent.feature_stds)     # Standard deviations during training
print(agent.risk_threshold)   # Adaptive threshold learned
```

## Next Steps

1. **Run the pipeline**: `python main.py`
2. **Check results**: Look for "AI Agent" in model comparison
3. **Review reasoning**: Examine sub-agent scores for test samples
4. **Compare**: How does it compare to XGBoost and others?
5. **Iterate**: Adjust reasoning_depth or adaptive settings

## Files Related to AI Agent

- 📄 **`src/agent_model.py`** - Main implementation
- 📄 **`src/models.py`** - Integration in training pipeline
- 📄 **`main.py`** - Displays agent reasoning and comparison

---

**Novelty Factor**: ⭐⭐⭐⭐⭐ (5/5)
- Combines multiple reasoning agents
- Interpretable decision-making
- Adaptive learning
- Hybrid statistical + analytical approach

**Created**: May 2026
