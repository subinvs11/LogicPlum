import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


@app.route('/')
def hello_world():
    data = pd.read_csv(r"/projects/Python3 Projects/LogicPlum/LogicPlum/resources/Demo_Diabetes_20180906.csv")
    X = data.drop('class', axis=1)
    y = data['class']
    #feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
    #X = data[feature_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)

    #------------------------------------

    # We will look at SHAP values for a single row of the dataset (we arbitrarily chose row 5). 
    # For context, we'll look at the raw predictions before looking at the SHAP values
    row_to_show = [51]
    data_for_prediction = X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    #clf.predict_proba(data_for_prediction_array)

    #------------------------------------

    import shap  # package used to calculate Shap values
    # had to use from command line conda install libpython m2w64-toolchain -c msys2

    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(clf)

    # Calculate Shap values
    shap_values = explainer.shap_values(data_for_prediction)
    print (shap_values)#zzz
    #------------------------------------
    shap.initjs()
    html = shap.force_plot(explainer.expected_value[1], shap_values[1], 
                    data_for_prediction, plot_cmap=["#f44141","#415bf4"], show=False)
    #graph_html = html.data
    
    #------------------------------------
    #shap_values = shap.TreeExplainer(clf).shap_values(X)
    sV =explainer.shap_values(X)
    #------------------------------------
    # Impact on model output
    shap.summary_plot(sV[1], X,plot_type="dot", color_bar="Pkyg", show=False)
    plt.savefig('static/graphs/graph1.png')
    plt.clf()
    #------------------------------------
    # Feature depends on what aspect of X[features]
    shap.dependence_plot("BloodPressure", sV[1], X, show=False)
    plt.savefig('static/graphs/graph2.png')
    plt.clf()
    #------------------------------------
    shap.summary_plot(sV[1], X, plot_type="bar", show=False)
    plt.savefig('static/graphs/graph3.png')
    
    return render_template('sample.html', graph_html=html)
    #return 'hi!!'

if __name__ == '__main__':
   app.run(debug = True)