from flask import Flask, render_template, url_for, request
from werkzeug import secure_filename

from graph_generator import GraphGenerator

app = Flask(__name__)
UPLOAD_FOLDER = '/uploads' 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/', methods = ['GET', 'POST'])
def upload():
    error = ''
    method = 'get'
    graph_html = None
    if request.method == 'POST':
        method = 'post'
        f = request.files['file']
        file_name = secure_filename(f.filename)
        try:
            extension = file_name.split('.')[1]
            if extension.lower() != 'csv':
                error = 'Invalid file. Please upload a csv file.'
        except:
            error = 'Invalid file. Please upload a csv file.'
        if not error:
            file_path = 'uploads/' + file_name
            f.save(file_path)
            graph_html, error = GraphGenerator().generate_graph(file_path)

    return render_template('upload.html', method=method, 
                            graph_html=graph_html, error=error)
    

if __name__ == '__main__':
   app.run(debug = True)