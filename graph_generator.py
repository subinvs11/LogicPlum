import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class GraphGenerator:

  def generate_graph(self, file_path):
    error = ''
    graph_html = None
    try:
      data = pd.read_csv(file_path)
      x = data.drop('class', axis=1)
      y = data['class']
      x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
      clf = RandomForestClassifier(random_state=0).fit(x_train, y_train)

      row_to_show = [51]
      data_for_prediction = x.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
      data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
      
      explainer = shap.TreeExplainer(clf)
      shap_values = explainer.shap_values(data_for_prediction)
      #shap.initjs()
      graph_html = shap.force_plot(explainer.expected_value[1], shap_values[1], 
                      data_for_prediction, plot_cmap=["#f44141","#415bf4"], show=False)
      
      sv = explainer.shap_values(x)
      plt.clf()
      shap.summary_plot(sv[1], x, plot_type="dot", color_bar="Pkyg", show=False)
      plt.savefig('static/graphs/graph1.png', bbox_inches="tight")
      plt.clf()
      
      shap.dependence_plot("BloodPressure", sv[1], x, show=False)
      plt.savefig('static/graphs/graph2.png', bbox_inches="tight")
      plt.clf()
      
      shap.summary_plot(sv[1], x, plot_type="bar", show=False)
      plt.savefig('static/graphs/graph3.png', bbox_inches="tight")
    except Exception as e:
      error = str(e)
    finally:
      return graph_html, error
