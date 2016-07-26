import sys, os
print os.path.dirname(sys.executable)
#sys.path.append('/anaconda/lib/python2.7/site-packages')

from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory

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

@application.route("/")
def hello():
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

print "Done loading classifiers"

if __name__ == "__main__":
    # application.debug = True
    application.run()
