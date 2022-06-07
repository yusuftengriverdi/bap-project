import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time
# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_validate
from sklearn import model_selection
import sklearn.metrics
from sklearn.metrics import make_scorer


d_keywords = [2, 5, 10, 20, 200]
c_keywords = ['Fe', 'Al', 'Pb']


def stack_images(number_of_samples=None,
                 ckeywords=c_keywords,
                 dkeywords=d_keywords,
                 directory="D:\\data\\Final Data", im_size=128,
                 should_save=True, should_return=True, should_normalize=True):
    im_dict = {}
    for ckeyword in ckeywords:
        for dkeyword in dkeywords:
            try:
                im_list = []
                # image counter for each class
                counter = 0
                # folder name
                fname = f"{ckeyword}{dkeyword}"
                # create path
                path = os.path.join(directory, ckeyword, fname)
                # take image list
                im_names = os.listdir(path)
                for name in tqdm(im_names):
                    # stop if we receive limit
                    if number_of_samples is not None:
                        if counter == number_of_samples:
                            break
                    # add image if its readable by cv2
                    try:
                        im = cv2.imread(os.path.join(path, name))
                        im = cv2.resize(im, (im_size, im_size), cv2.INTER_CUBIC)
                        im_list.append(np.asarray(im))
                        counter += 1
                    except Exception as e:
                        print(str(e))
                        pass
                    im_dict[fname] = im_list
            except Exception as e:
                print(f"{ckeyword} for {dkeyword} Cannot found!")

    for key in im_dict.keys():
        print(key, ":", len(im_dict[key]))
    print(im_dict.keys())
    return im_dict


def extract_data(base_data, selected_keywords=[['Fe'], ['2']], type='e',
                 should_save=True, should_return=True, should_normalize=True):
    """    names_dict = {}
        if type =='e':
            for ckeyword in selected_keywords[0]:
                names = []
                for dkeyword in selected_keywords[1]:
                    name = f"{ckeyword}{dkeyword}"
                    names.append(name)
                names_dict[ckeyword] = names

            x_dict = {}
            for i in range(len(selected_keywords[0])):
                x_dict[i] = []
            x_dict[(len(x_dict.keys())+1)] = []

            ct = 0
            for key in base_data.keys():
                if key in names_dict.values():
                    x_dict[ct] = np.concatenate([x_dict[ct], base_data[key]])
                else:
                    x2 = np.concatenate([x2, base_data[key]])


            one_hot_vector = np.eye(len(selected_keywords[0]))
            for vector in one_hot_vector:

        x1 = np.concatenate([base_data['Fe5'][0], base_data['Fe10'][0], base_data['Fe20'][0]])
        x2 = np.concatenate([base_data['Al10'][0], base_data['Al20'][0], base_data['Pb2'][0], base_data['Pb5'][0],
                             base_data['Pb10'][0], base_data['Pb20'][0]])

        len(x1), len(x2)

        imlist = []
        one_hot_vector = np.eye(2)
        for x in x1:
            imlist.append([x, vector[0]])
        for x in x2:
            imlist.append([x, vector[1]])

        len(imlist), len(x1) + len(x2)"""


def plot_metrics(model_name, history, metric_name):
    plt.figure()
    e = range(1, len(history.history[metric_name]) + 1)
    plt.plot(e, history.history[metric_name], 'bo', label=metric_name)
    plt.plot(e, history.history[f'val_{metric_name}'], 'b', label=f'val_{metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric_name}')
    plt.legend(loc='lower right')
    plt.title(f'Comparing training and validation loss of {model_name}')
    plt.savefig(f'./{model_name}_{metric_name}.png')
    plt.show()


def save_logs(history, model_name):
    pd.DataFrame.from_dict(history.history).to_csv(f'logs_{model_name}.csv', index=False)
    df = pd.DataFrame.from_dict(history.history)
    return df.style.set_caption(f"{model_name}")


def run_classifier(clfr, cv, X, y):
    start_time = time.time()
    scores = cross_validate(clfr, X, y, scoring={'accuracy':'accuracy',
                                                'f1': 'f1',
                                                'roc_auc': 'roc_auc',
                                                'precision': 'precision',
                                                'recall': 'recall',
                                                'neg_log_loss': 'neg_log_loss',
                                                'matt': make_scorer(sklearn.metrics.matthews_corrcoef), 
                                                'cohen': make_scorer(sklearn.metrics.cohen_kappa_score)
                                                }, cv=cv, n_jobs=-1)
    #y_pred = clfr.predict(X)
    #scores['matthews_corr'] = sklearn.metrics.matthews_corrcoef(y, y_pred)
    #scores['cohen_kappa'] = sklearn.metrics.cohen_kappa_score(y, y_pred)
    return scores


def convert_binary(label_list):
    new_label_list = []
    for label in tqdm(label_list):
        new_label_list.append(np.argmax(label))
    return np.asarray(new_label_list)


def extract_features(nb_features, feature_extractor, image_stack, label_stack, is_unet=False):
    """Extract bottleneck features"""
    nb_features = nb_features
    features = np.empty((len(image_stack), nb_features))
    labels = []
    index = 0
    for x in tqdm(image_stack):
        feature = feature_extractor.predict(np.expand_dims(x, axis=0))
        if is_unet:
            feature = feature.flatten()
        features[index, :] = np.squeeze(feature)
        labels.append(label_stack[index])
        index = index + 1
    return features, labels
