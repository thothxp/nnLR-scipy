import json
import pickle
import numpy as np
import seaborn as sns
from nnLRScipy import nnLRScipy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split


def plot_gallery(title, images, n_col, n_row, image_shape, cmap=plt.cm.gray, save=False):
    fig = plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    fig.canvas.set_window_title(title)

    for i, comp in enumerate(images[:n_row * n_col]):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=cmap,
                   interpolation='nearest',
                   # vmin=-vmax, vmax=vmax
                   )
        plt.xticks(())
        plt.yticks(())

    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

    if save:
        plt.savefig('../logs/components.eps', format='eps',
                    # bbox_extra_artists=(lgd,),
                    dpi=100, bbox_inches='tight')


def main():

    # Constraints only for COBYLA, SLSQP and trust-constr.
    params = [
        # ('TNC', {'reltol': 1e-3, 'maxit': 2000, 'opt_method': 'TNC', 'positive': True, 'penalty': 'l2', 'verbose': True})
        ('SLSQP', {'reltol': 1e-9, 'maxit': 2000, 'opt_method': 'SLSQP', 'positive': True, 'penalty': None, 'verbose': True}),
        # ('COBYLA', {'reltol': 1e-6, 'maxit': 2000, 'opt_method': 'COBYLA', 'positive': True, 'penalty': None, 'verbose': True}),
        # ('trust-constr', {'reltol': 1e-6, 'maxit': 2000, 'opt_method': 'trust-constr', 'positive': True, 'penalty': None, 'verbose': True})
        # ('L-BFGS-B', {reltol': 1e-6, 'maxit': 2000, 'opt_method': 'L-BFGS-B', 'positive': False, 'penalty': None, 'verbose': True})

    ]

    n_row, n_col = 1, 10
    for idx, params_ in enumerate(params):
        method_name, par_dict = params_

        estimator = nnLRScipy(**par_dict).fit(X_tr, labels_tr)

        pred_labels_tst = estimator.predict(X_tst)
        acc = np.mean(pred_labels_tst == labels_tst)
        print('Method {} Acc. {}'.format(method_name, acc))

        coef_ = estimator.coef_

        print(np.linalg.norm(coef_, axis=0))

        plot_gallery(title='Original Images', images=X_tr, n_col=n_col, n_row=n_row,
                     image_shape=image_shape)

        plot_gallery(title="coef_", images=coef_.T, n_col=n_col, n_row=n_row,
                     image_shape=image_shape, save=True)

        cm = confusion_matrix(labels_tst, pred_labels_tst)

        plt.figure(figsize=(12, 12))
        sns.heatmap(cm, annot=True,
                    linewidths=.5, square=True, cmap='Blues_r', fmt='0.4g');

        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

        plt.savefig('../logs/cm_{0:.5f}.eps'.format(acc), format='eps',
                    dpi=100, bbox_inches='tight')

        with open('../logs/components.pkl', 'wb') as f:
            pickle.dump(coef_, f)

        with open('../logs/par_dict.json', 'w') as fp:
            json.dump(par_dict, fp, sort_keys=True, indent=4)

    plt.show()


if __name__ == "__main__":

    image_shape = (28, 28)
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))

    # # Change train_samples according to you memory.
    train_samples = 1000

    X_tr, X_tst, labels_tr, labels_tst = train_test_split(
        X, y, train_size=train_samples, test_size=10000)

    labels_tr = np.asarray(labels_tr).astype(np.int)
    labels_tst = np.asarray(labels_tst).astype(np.int)

    X_tr /= 255.
    X_tst /= 255.

    main()
