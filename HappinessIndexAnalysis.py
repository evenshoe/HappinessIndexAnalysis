import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
import textwrap

#1.
#correlation analysis to determinate which variables tend to show most correlation with happines inde

data = pd.read_csv('happinesIndexData.csv')

#correlation matrix
selected_columns = ['Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']
       
selected_data = data[selected_columns]
corr_matrix = selected_data.corr()

wrapped_labels = [textwrap.fill(label, 11) for label in selected_columns]

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=wrapped_labels, yticklabels=wrapped_labels)
plt.xticks(rotation=0, ha='center')
plt.yticks(rotation=0, ha='right')
plt.title('Correlation heatmap')
plt.subplots_adjust(bottom=0.2)
plt.show()


#finding columns with the highest correlation with happines score
corr = selected_data.corr()['Score']
print('\nVariables with the highest correlation with happines score:')
print(corr.nlargest())


#2.
#testing different regression models to determinate which one best predicts the value of happines index

X = data[['GDP per capita']] #'GDP per capita' was no 1 in correlation analysis
y = data['Score']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)


regression_models = {'Linear Regression' : LinearRegression(),
                     'Ridge Regression' : Ridge()}

for name, model in regression_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    rmse = math.sqrt(mean_squared_error(y_test,y_pred))
    print(f'\n{name}:')
    print(f'RMSE: {rmse:.2f}')
    print(f'R-squared: {r2:.2f}')

    
y_pred_rr = regression_models['Ridge Regression'].predict(X_test)
plt.scatter(X_test,y_test, label = 'real value')
plt.plot(X_test,y_pred_rr,color = 'red', label = 'prediction line')
plt.title('Ridge Regression')
plt.xlabel('GDP per capita')
plt.ylabel('Happiness Score')
plt.legend()
plt.show()

#3.
#performing a sensitivity analysis to determine which factors have the greatest impact on the value of the happiness index

X1 = data[selected_columns].drop(['Score'], axis=1)

rr_model = Ridge()
rr_model.fit(X1, y)
importance = rr_model.coef_

for i, v in enumerate(importance):
    print('\nVariable: %s, Importance: %.5f' % (X1.columns[i], v))

wrapped_labels = [textwrap.fill(label, 11) for label in X1.columns]
plt.bar(wrapped_labels, importance)
plt.subplots_adjust(bottom=0.2)
plt.title('Variable importance in ridge regression model')
plt.xlabel('Variables')
plt.ylabel('Importance')
plt.show()