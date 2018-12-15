from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import tensorflow as tf
import os
model_file = ''
app = Flask(__name__)


def predict_classes(given_issue_values):
    result = {
        '1': {},
        '7': {},
        '14': {},
        '30': {},
        '90': {},
    }
    result1 = model1.predict_classes([[given_issue_values]])[0]
    result7 = model7.predict_classes([[given_issue_values]])[0]
    result14 = model14.predict_classes([[given_issue_values]])[0]
    result30 = model30.predict_classes([[given_issue_values]])[0]
    result90 = model90.predict_classes([[given_issue_values]])[0]
    result['1']['before'] = '✓' if result1 == 0 else ''
    result['1']['after'] = '✓' if result1 == 1 else ''
    result['7']['before'] = '✓' if result7 == 0 else ''
    result['7']['after'] = '✓' if result7 == 1 else ''
    result['14']['before'] = '✓' if result14 == 0 else ''
    result['14']['after'] = '✓' if result14 == 1 else ''
    result['30']['before'] = '✓' if result30 == 0 else ''
    result['30']['after'] = '✓' if result30 == 1 else ''
    result['90']['before'] = '✓' if result90 == 0 else ''
    result['90']['after'] = '✓' if result90 == 1 else ''
    return result


@app.route('/', methods=['GET'])
def predict_issue_lifetime():
    context = {
        'issueCleanedBodyLen': request.args.get('issueCleanedBodyLen', 0),
        'nCommitsByCreator': request.args.get('nCommitsByCreator', 0),
        'nCommitsInProject': request.args.get('nCommitsInProject', 0),
        'nIssuesByCreator': request.args.get('nIssuesByCreator', 0),
        'nIssuesByCreatorClosed': request.args.get('nIssuesByCreatorClosed', 0),
        'nIssuesCreatedInProject': request.args.get('nIssuesCreatedInProject', 0),
        'nIssuesCreatedInProjectClosed': request.args.get('nIssuesCreatedInProjectClosed', 0)
    }
    given_issue_values = np.array([
        context['issueCleanedBodyLen'],
        context['nCommitsByCreator'],
        context['nCommitsInProject'],
        context['nIssuesByCreator'],
        context['nIssuesByCreatorClosed'],
        context['nIssuesCreatedInProject'],
        context['nIssuesCreatedInProjectClosed'],
    ])
    with graph.as_default():
        context['results'] = predict_classes(given_issue_values)
        return render_template('form.html', **context)


if __name__ == '__main__':
    model1 = load_model(f'{os.getcwd()}/out/models/combinedRepos/beforeClass/menziesbefore1.h5')
    model7 = load_model(f'{os.getcwd()}/out/models/combinedRepos/beforeClass/menziesbefore7.h5')
    model14 = load_model(f'{os.getcwd()}/out/models/combinedRepos/beforeClass/menziesbefore14.h5')
    model30 = load_model(f'{os.getcwd()}/out/models/combinedRepos/beforeClass/menziesbefore30.h5')
    model90 = load_model(f'{os.getcwd()}/out/models/combinedRepos/beforeClass/menziesbefore90.h5')
    global graph
    graph = tf.get_default_graph()
    app.run(port=8000, debug=True)
