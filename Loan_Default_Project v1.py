import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report,accuracy_score, roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from imblearn.over_sampling import SMOTE
from sklearn.impute import IterativeImputer
import joblib






data=pd.read_csv(r'D:\ML PROJECTS\Loan_Default.csv')

columns_to_remove = [
    'ID',
    'year',
    'loan_limit',
    'approv_in_adv',
    'business_or_commercial',
    'construction_type',
    'Secured_by',
    'total_units',
    'submission_of_application',
    'Security_Type',
    'co-applicant_credit_type',
    'Gender',
    'age',
    'Region',
    'dtir1',
    'LTV',
    'rate_of_interest',
    'Interest_rate_spread',
    'Upfront_charges',
]
data=data.drop(columns_to_remove,axis=1)


cat=data.select_dtypes(include='object')
print('Catogorical : ',cat.columns)

num=data.select_dtypes(include=['int64','float64'])
print('Numirical : ',num.columns)

print('Checking correlation of Status with numeric data columns:')
print(data[num.columns].corr()['Status'].sort_values(ascending=False))

num = num.drop('Status', axis=1)

X=data.drop('Status',axis=1)
y=data['Status']



#cat=cat.fillna('Missing')

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,stratify=y)

from sklearn.pipeline import Pipeline as SkPipeline

# Numeric pipeline: impute then scale
numeric_pipeline = SkPipeline([
    ('imputer', IterativeImputer(random_state=42)),
    ('scaler', StandardScaler())
])

preprossing=ColumnTransformer(
    [
        ('num',numeric_pipeline,num.columns),
        ('cat',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),cat.columns),
    ]
)

pipe=Pipeline(
    [
        ('preprocessor',preprossing),
        ('SMOTE',SMOTE()),
        ('model',RandomForestClassifier())
    ]
)

print('Catogorical nulls :\n',cat.isnull().sum())

#scores=cross_val_score(pipe,X,y,cv=5)
# ─────────────────────────────────────────────
# GRIDSEARCH
# ─────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators':     [100, 200],
    'model__max_depth':        [10, 20, None],
    'model__min_samples_split':[2, 5],
    'model__max_features':     ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    pipe,
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV ROC-AUC:", grid_search.best_score_)

best_model = grid_search.best_estimator_
pred = best_model.predict(X_test)
prob = best_model.predict_proba(X_test)[:,1]

print("Test Accuracy:", accuracy_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
print(classification_report(y_test, pred))


joblib.dump(best_model, r'D:\ML PROJECTS\credit_risk_model.pkl')
joblib.dump(X_train.columns.tolist(), r'D:\ML PROJECTS\model_columns.pkl')

print("Saved!")
print("Columns:", X_train.columns.tolist())

'''
for col in cat.columns:
    print(f"\n{col}:")
    print(data.groupby(col)['Status'].mean().sort_values(ascending=False))

'''

'''
# Test with realistic values
test_cases = pd.DataFrame([
    {
        'loan_amount': 500000,
        'term': 360, 'property_value': 120000,
        'income': 1500, 'Credit_Score': 520,
        'loan_type': 'type1', 'loan_purpose': 'p1',
        'Credit_Worthiness': 'l2', 'open_credit': 'opc',
        'Neg_ammortization': 'neg_amm', 'interest_only': 'int_only',
        'lump_sum_payment': 'lpsm', 'occupancy_type': 'ir',
        'credit_type': 'CRIF', 'co-applicant_credit_type': 'EXP',
        'age': '>74', 'Region': 'North-East', 'Security_Type': 'Indriect',
        # dropped columns below — add dummy values
        'loan_limit': 'cf', 'approv_in_adv': 'nopre',
        'business_or_commercial': 'nob/c', 'construction_type': 'sb',
        'Secured_by': 'home', 'total_units': '1U',
        'submission_of_application': 'to_inst',
        'Gender': 'Male', 'dtir1': 55, 'LTV': 95,
    },
    {
        'loan_amount': 80000,
        'term': 180, 'property_value': 600000,
        'income': 15000, 'Credit_Score': 820,
        'loan_type': 'type1', 'loan_purpose': 'p1',
        'Credit_Worthiness': 'l1', 'open_credit': 'nopc',
        'Neg_ammortization': 'not_neg', 'interest_only': 'not_int',
        'lump_sum_payment': 'not_lpsm', 'occupancy_type': 'pr',
        'credit_type': 'CIB', 'co-applicant_credit_type': 'CIB',
        'age': '35-44', 'Region': 'North', 'Security_Type': 'direct',
        'loan_limit': 'cf', 'approv_in_adv': 'pre',
        'business_or_commercial': 'nob/c', 'construction_type': 'sb',
        'Secured_by': 'home', 'total_units': '1U',
        'submission_of_application': 'not_inst',
        'Gender': 'Male', 'dtir1': 22, 'LTV': 55,
    }
])

print('Data Group \n :',data.groupby('credit_type')['Status'].mean().sort_values(ascending=False))

# Drop same columns you dropped during training
test_cases = test_cases.drop(columns_to_remove, axis=1, errors='ignore')

pred = pipe.predict(test_cases)
prob = pipe.predict_proba(test_cases)

print("\nRow 1 (High Risk):")
print(f"  Prediction: {'DEFAULT' if pred[0]==1 else 'SAFE'}")
print(f"  Probability: {prob[0][1]*100:.1f}% default chance")

print("\nRow 2 (Low Risk):")
print(f"  Prediction: {'DEFAULT' if pred[1]==1 else 'SAFE'}")
print(f"  Probability: {prob[1][1]*100:.1f}% default chance")

print("Columns in pipe:", X_train.columns.tolist())
print("\nColumns in test_cases:", test_cases.columns.tolist())
print("\nTest case 1 values:")
print(test_cases.iloc[0])
print("\nTest case 2 values:")
print(test_cases.iloc[1])

feature_names = (
    pipe.named_steps['preprocessor']
    .named_transformers_['cat']
    .get_feature_names_out(cat.columns.tolist())
)

all_names = num.columns.tolist() + list(feature_names)
imp_df = pd.DataFrame({
    'feature': all_names,
    'importance': pipe.named_steps['model'].feature_importances_
}).sort_values('importance', ascending=False)

print(imp_df.head(10).to_string(index=False))

'''

'''
pred=pipe.predict(X_test)
print('report : ',classification_report(y_test,pred))
print('Feature Importance : ',pipe.named_steps['model'].feature_importances_)

'''

'''
print("Fold Accuracies:", scores)
print("Mean Accuracy:", scores.mean())
'''

'''
# Remove credit_type and retrain
X_no_equi = X.drop('credit_type', axis=1)

pipe_test = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', numeric_pipeline, num.columns.tolist()),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
         [c for c in cat.columns if c != 'credit_type'])
    ])),
    ('model', RandomForestClassifier(
        max_depth=20, max_features='sqrt',
        min_samples_split=5, n_estimators=200
    ))
])

pipe_test.fit(X_train.drop('credit_type', axis=1), y_train)
pred_test = pipe_test.predict(X_test.drop('credit_type', axis=1))
print(classification_report(y_test, pred_test))
'''