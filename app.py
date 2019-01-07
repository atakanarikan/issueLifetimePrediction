from flask import Flask, render_template, request, redirect
from keras.models import load_model
import numpy as np
import tensorflow as tf
import os
model_file = ''
models_by_dataset = {
    'riivo': {},
    'random': {},
    'top': {},
    'menzies': {}
}
app = Flask(__name__)


def load_models():
    for dataset in ['menzies', 'random', 'top']:
        models_by_dataset[dataset] = {
            '1': load_model(f'{os.getcwd()}/out/models/combinedRepos/beforeClass/{dataset}before1.h5'),
            '7': load_model(f'{os.getcwd()}/out/models/combinedRepos/beforeClass/{dataset}before7.h5'),
            '14': load_model(f'{os.getcwd()}/out/models/combinedRepos/beforeClass/{dataset}before14.h5'),
            '30': load_model(f'{os.getcwd()}/out/models/combinedRepos/beforeClass/{dataset}before30.h5'),
            '90': load_model(f'{os.getcwd()}/out/models/combinedRepos/beforeClass/{dataset}before90.h5')
        }
    # models_by_dataset['riivo'] = {
    #     '1': load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore1.h5'),
    #     '7': load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore7.h5'),
    #     '14': load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore14.h5'),
    #     '30': load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore30.h5'),
    #     '90': load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore90.h5')
    # }


def predict_classes(dataset, given_issue_values):
    result = {
        '1': {},
        '7': {},
        '14': {},
        '30': {},
        '90': {},
    }
    for timeclass, model in models_by_dataset[dataset].items():
        prediction = model.predict_classes([[given_issue_values]])[0]
        result[timeclass]['before'] = '✓' if prediction == 0 else ''
        result[timeclass]['after'] = '✓' if prediction == 1 else ''
    return result


@app.route('/', methods=['GET'])
def homepage():
    dataset = request.args.get('dataset')
    if dataset:
        return redirect(f'/{dataset}')
    return render_template('index.html')


@app.route('/<dataset>', methods=['GET'])
def predict_issue_lifetime(dataset):
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
        context['results'] = predict_classes(dataset, given_issue_values)
        return render_template('form.html', **context)


if __name__ == '__main__':
    print('Loading models...')
    load_models()
    print('Done!')
    global graph
    graph = tf.get_default_graph()
    app.run(port=8000, debug=True)
