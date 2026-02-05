import pandas as pd
import xgboost as xgb
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

def train_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates multiple models.
    Returns a DataFrame with performance metrics.
    """
    results = []
    
    # --- 1. Random Forest Classifier ---
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest", y_test, rf_pred, rf_prob))
    
    # --- 2. Gradient Boosting Classifier ---
    print("Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_prob = gb_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("Gradient Boosting", y_test, gb_pred, gb_prob))
    
    # --- 3. XGBoost Classifier ---
    print("Training XGBoost Classifier...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost", y_test, xgb_pred, xgb_prob))
    
    # --- 4. MLP Classifier (Keras) ---
    print("Training MLP (Deep Learning)...")
    mlp_model = build_mlp(X_train.shape[1])
    
    # Class weights
    total = len(y_train)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    mlp_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=0
    )
    
    mlp_prob = mlp_model.predict(X_test).flatten()
    mlp_pred = (mlp_prob > 0.5).astype(int)
    results.append(evaluate("MLP (Deep Learning)", y_test, mlp_pred, mlp_prob))
    
    return pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)

def build_mlp(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }
