from flask import Flask, render_template, request
from src.pipelines.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(
            smoothness_mean= float(request.form.get('smoothness_mean')),
            symmetry_mean= float(request.form.get('symmetry_mean')),
            fractal_dimension_mean=float(request.form.get('fractal_dimension_mean')),
            texture_se= float(request.form.get('texture_se')),
            smoothness_se=float(request.form.get('smoothness_se')),
            compactness_se=float(request.form.get('compactness_se')),
            concavity_se=float(request.form.get('concavity_se')),
            concave_points_se= float(request.form.get('concave_points_se')),
            symmetry_se = float(request.form.get('symmetry_se')),
            fractal_dimension_se=float(request.form.get('fractal_dimension_se')),
            smoothness_worst=float(request.form.get('smoothness_worst')),
            symmetry_worst=float(request.form.get('symmetry_worst')),
            fractal_dimension_worst=float(request.form.get('fractal_dimension_worst'))
        )
        
        print("Creating Custom Data as DataFrame")
        data_df = data.get_data_as_dataframe()
        print("Generating predictions...")
        predict_pipeline = PredictPipeline()
        preds = predict_pipeline.predict(data_df)
        classes_dict = {0 : 'Benign', 1 : 'Malignant'}
        return render_template('home.html', results=classes_dict[int(preds[0])])
        
        

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)