import sys, os
# print os.path.dirname(sys.executable)
#sys.path.append('/anaconda/lib/python2.7/site-packages')

# Flask
from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory

# Bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import CustomJS, ColumnDataSource, AjaxDataSource
from bokeh.resources import CDN
from bokeh.embed import components
from bokeh.models.widgets import Select, RadioGroup
from bokeh.models.widgets.inputs import AutocompleteInput
from bokeh.layouts import column, layout

print "Imported flask"

from sklearn.externals import joblib

print "Imported sklearn joblib"

import json
import numpy as np
import re

print "Loading flask"

# from flask import send_static_file
application = Flask(__name__,
                    static_url_path='')

print "Loading sklearn classifiers"

clf_journal, journal_labels = joblib.load('svm_journal.pkl')
clf_category, category_labeller = joblib.load('svm_category.pkl')

# clf_journal, journal_labels = joblib.load('svm_journal_small.pkl')
# clf_category, category_labeller = joblib.load('svm_category_small.pkl')
# journals = ['PRL', 'Nature']
journal_coef = clf_journal.named_steps['clf'].coef_
terms =  clf_journal.named_steps['vect'].get_feature_names()

print "Loading anomaly data"
with open("anomaly/anomaly.json", "r") as f:
    anomaly_data = json.load(f)

@application.route("/")
def route():
    return render_template("abstract_predictor.html")
    # return send_static_file("test.html")

@application.route("/js/<path:path>")
def send_js(path):
    return send_from_directory('js', path)

@application.route('/svm', methods=['POST'])
def svm_category():
    abstract = request.form['abstract'].encode("ascii", "ignore")
    
    journal_predict = clf_journal.predict([abstract])[0]
    journal_object = [{'name': y, 'value': int(x==journal_predict)}
                      for x, y in enumerate(journal_labels)]
    
    
    category_classes = category_labeller.classes
    category_predict = category_labeller.inverse_transform(
                                clf_category.predict([abstract]))[0]
    category_object = [{'name': x, 'value': int(x in category_predict)}
                       for x in category_classes]
        
    print (journal_predict, category_predict)
    
    # Return a copy of the abstract where text is colored by word weights.
    def backgroundcolor(x):
        R = "%0.2X" % int(max(0, min(255, 255*(1-x))))
        G = "%0.2X" % int(max(0, min(255, 255*(1-np.abs(x)))))
        B = "%0.2X" % int(max(0, min(255, 255*(1+x))))
        return "#" + R+G+B

    def color(x):
        R = "%0.2X" % int(max(0, min(255, 255*(-x))))
        G = "00"
        B = "%0.2X" % int(max(0, min(255, 255*(+x))))
        return "#" + R+G+B
    
    tfidf = clf_journal.named_steps['tfidf'].transform(
                        clf_journal.named_steps['vect'].transform([abstract]))
    
    svm_weights = np.squeeze(journal_coef*tfidf.toarray())
    # svm_max = np.max(np.abs(svm_weights))
    
    most_important_words = sorted([(terms[x], y)
                                   for x, y in enumerate(svm_weights)
                                   if svm_weights[x] != 0],
                                  key=lambda x: -np.abs(x[1]))

    abstract_colored = abstract.lower()
    abstract_colored = abstract_colored.replace('@', '')
    for item in most_important_words:
        newString ="<@@%s\">%s<@>" \
                      % (backgroundcolor(-item[1]), item[0].upper())
        abstract_colored = re.sub(r"\b%s\b" % item[0],
                                  newString,
                                 abstract_colored)
    #     break;
    abstract_colored = abstract_colored.replace("@@",
                                        "span style=\"background-color:")
    abstract_colored = abstract_colored.replace("@", "/span")
    abstract_colored = abstract_colored.lower()
    
    
    return json.dumps({'status': 'OK',
                       'abstract': abstract.lower(),
                       'journal': journal_object,
                       'category': category_object,
                       'abstract_colored': abstract_colored})

@application.route("/anomaly")
def anomaly():  
    # prepare some data
    x = [1, 2, 3, 4, 5]
    y = [6, 7, 2, 4, 5]
    source = ColumnDataSource(data=dict(x=x,
                                        y=y,
                                        xstepLeft=x,
                                        xstepRight=x,
                                        ystepLeft=y,
                                        ystepRight=y))
    # Plot
    # TOOLS = "box_zoom,box_select,crosshair,resize,reset"
    TOOLS = ""
    plot = figure(title="test title", tools=TOOLS)
    plot.line('x', 'y', source=source, line_width=2)
    plot.segment('xstepLeft', 'ystepLeft',
                 'xstepRight', 'ystepRight',
                 source=source,
                 line_width=4)
    plot.xaxis.axis_label = "Article number"
    plot.yaxis.axis_label = "TF-IDF (moving average)"
    
    # List of categories
        
    select_category = Select(title="Category:",
                             value=anomaly_data.keys()[0],
                             options=anomaly_data.keys())
                             
    word_list = anomaly_data['atom-ph']['words_sorted_anomaly']
    textbox = AutocompleteInput(title="Word",
                             value="",
                             completions=word_list)
    
    callback_category = CustomJS(args=dict(textbox=textbox),
                                 code="""
        var category = cb_obj.get('value');
        
        $.ajax({
            type: 'POST',
            url: '/anomaly_word_list',
            data: {'category': category},
            success: function(x) {
                word_list = JSON.parse(x);
                textbox.attributes.completions = word_list;
                textbox.attributes.value = '';
                textbox.trigger('change');
                },
        });
        """)    
    
    callback = CustomJS(args=dict(source=source,
                                  select_category=select_category,
                                  plot=plot), code="""
            var data = source.get('data');
            var word = cb_obj.get('value');
            var category = select_category.get('value');
            console.log(plot)
            // var f = 0;
            
            
            // Okay, calling ajax from this script is not ideal, but oh well.
            
            $.ajax({
                type: 'POST',
                url: '/anomaly_data',
                data: {'category': category, 'word': word},
                success: function(x) {
                    data2 = JSON.parse(x);
                    //edata = data2;
                    data['x'] = data2['x'];
                    data['y'] = data2['y'];
                    data['xstepLeft'] = data2['xstepLeft'];
                    data['xstepRight'] = data2['xstepRight'];
                    data['ystepLeft'] = data2['ystepLeft'];
                    data['ystepRight'] = data2['ystepRight'];
                    // x = data2['x'];
                    // y = data2['y'];
                    source.trigger('change');
                    
                    // Change axis labels and title
                    plot.attributes.left[0].attributes.axis_label = "TF-IDF of " + word + " (moving average)";
                    plot.attributes.above[0].attributes.text = word + " (tfidf rank # " + data2['tfidf_rank'] + ", anomaly rank # " + data2['anomaly_rank'] + ")"
                    plot.trigger('change');
                    },
            });
            """)

    # assign callbacks
    select_category.callback = callback_category
    textbox.callback = callback
    
    script, div = components(layout([[select_category], [textbox], [plot]]))
    return render_template("graph.html", script=script, div=div)
    # return render_template("graph.html")
    # return send_static_file("test.html")
    
@application.route("/anomaly_data", methods=['POST'])
def anomalydata():
    # print (request.data, request.args, request.form, request.values)
    category = request.form['category']
    word = request.form['word']
    result = anomaly_data[category]['data'][word]
    
    # word = request.form['word'].encode("ascii", "ignore")
    # print 'Here!'
    return json.dumps({'x': anomaly_data[category]['index_moving_average'],
                       'y': result['moving_average'],
                       'tfidf_rank': result['tfidf_rank'],
                       'anomaly_rank': result['anomaly_rank'],
                       'xstepLeft': result['anomaly_index'][:-1],
                       'ystepLeft': [x for x,y in zip(result['anomaly_mean'], result['anomaly_std'])],
                       'xstepRight': result['anomaly_index'][1:],
                       'ystepRight': [x for x,y in zip(result['anomaly_mean'], result['anomaly_std'])]})
                       
@application.route("/anomaly_word_list", methods=['POST'])
def anomaly_word_list():
    # print (request.data, request.args, request.form, request.values)
    category = request.form['category']
    return json.dumps(anomaly_data[category]['words_sorted_anomaly'])

print "Done loading classifiers"

if __name__ == "__main__":
    application.debug = True
    application.run()
