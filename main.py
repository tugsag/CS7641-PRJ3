import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix, plot_roc_curve, roc_auc_score, homogeneity_completeness_v_measure, homogeneity_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

SEED = 903454028
np.random.seed()


def kmeans(X, y, id=None, redux='', k=None):
    if k != None:
        k_vals = np.arange(2, k)
    else:
        k_vals = np.arange(2, 21)
    sils = []
    homs = []
    sses = []
    best_sil = 0
    best_k = 0
    best_model = 0
    for i, k in enumerate(k_vals):
        print(i/k_vals.shape[0])
        kmeans = KMeans(n_clusters=k, random_state=SEED)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)

        # append scores
        homs.append(homogeneity_score(y, labels))
        sils.append(sil)
        sses.append(kmeans.inertia_)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_model = kmeans

    labels = best_model.labels_
    H, C, V = homogeneity_completeness_v_measure(y, labels)

    # write metrics to txt
    output = open('figures/kmeans_{}{}.txt'.format(id, redux), 'w')
    output.write('Highest silhouette score of {} at {}\n'.format(best_sil, best_k))
    output.write('Best model homogeneity: {}\n'.format(H))
    output.write('Best model completeness: {}\n'.format(C))
    output.write('Best model V score: {}\n'.format(V))
    output.close()

    plt.plot(k_vals, sils, label='Silhouette')
    plt.plot(k_vals, homs, label='Homogeneity')
    plt.xlabel('Clusters')
    plt.ylabel('Score')
    plt.title('KMeans Homogeneity-silhouette')
    plt.legend(loc='best')
    plt.savefig('figures/kmeans_hom_sil_{}{}.png'.format(id, redux))
    plt.clf()

    #SSE
    plt.plot(k_vals, sses, label='SSE')
    plt.xlabel('Clusters')
    plt.ylabel('SSE')
    plt.title('SSE of KMeans')
    plt.savefig('figures/kmeans_SSE_{}{}.png'.format(id, redux))
    plt.clf()

    print('ANN running -------------- \n')
    KM_X = np.hstack([X, np.atleast_2d(labels).T])
    ann(KM_X, y, id=id, redux='_kmeans')

    return best_model


def em(X, y, id=None, redux='', k=None):
    if k != None:
        k_vals = np.arange(2, k, 2)
    else:
        k_vals = np.arange(2, 20, 2)
    cov_types = ['full', 'tied', 'spherical']
    bics = []
    sils = []
    homs = []
    best_bic = np.inf
    best_k = 0
    best_model = 0
    best_cov = 'full'
    best_cov_ind = 0
    for i, k in enumerate(k_vals):
        print(i/k_vals.shape[0])
        # for j, c in enumerate(cov_types):
        gauss = GaussianMixture(n_components=k, covariance_type='full', n_init=5, init_params='random', random_state=SEED)
        gauss.fit(X)
        bic = gauss.bic(X)
        labels = gauss.predict(X)
        homs.append(homogeneity_score(y, labels))
        sils.append(silhouette_score(X, labels))

        # append scores
        bics.append(bic)
        if bic < best_bic:
            best_bic = bic
            best_model = gauss
            best_k = k
            # best_cov = c
            # best_cov_ind = j

    labels = best_model.predict(X)
    H, C, V = homogeneity_completeness_v_measure(y, labels)
    silh = silhouette_score(X, labels)

    # write metrics
    output = open('figures/em_{}{}.txt'.format(id, redux), 'w')
    output.write('Lowest BIC of {} at k={}, cov={}\n'.format(best_bic, best_k, best_cov))
    output.write('Silhouette of {} for best model\n'.format(silh))
    output.write('Best model homogeneity: {}\n'.format(H))
    output.write('Best model completeness: {}\n'.format(C))
    output.write('Best model V score: {}\n'.format(V))
    output.close()

    # Covariance
    # plt.plot(k_vals, bics[0::3], label='Full')
    # plt.plot(k_vals, bics[1::3], label='Tied')
    # plt.plot(k_vals, bics[2::3], label='Spherical')
    # plt.xlabel('Clusters')
    # plt.ylabel('BIC')
    # plt.title('EM Covariance Comparison')
    # plt.legend(loc='best')
    # plt.savefig('figures/em_cov_comparison_{}{}.png'.format(id, redux))
    # plt.clf()

    # best model
    # plt.plot(k_vals, bics[best_cov_ind::3], label='BIC')
    plt.plot(k_vals, bics, label='BIC')
    plt.xlabel('Clusters')
    plt.ylabel('BIC')
    plt.title('EM {} BIC'.format('full'))
    plt.savefig('figures/em_best_cov_{}{}.png'.format(id, redux))
    plt.clf()

    plt.plot(k_vals, sils, label='Silhouette')
    plt.plot(k_vals, homs, label='Homogeneity')
    plt.xlabel('Clusters')
    plt.ylabel('Score')
    plt.title('EM Homogeneity-silhouette')
    plt.legend(loc='best')
    plt.savefig('figures/em_hom_sil_{}{}.png'.format(id, redux))
    plt.clf()

    print('ANN running -------------- \n')
    EM_X = np.hstack([X, best_model.predict_proba(X)])
    ann(EM_X, y, id=id, redux='_em')

    return best_model


# ------------- DimRedux --------------- #

def dimredux_scatter(trans, y, path, id=None):
    for cl in [0, 1]:
        mask = cl==y
        plt.scatter(x=trans[mask][:, 0], y=trans[mask][:, 1],  alpha=.3, label=cl)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title('Projection of {} reduced data'.format(id))
    plt.legend(loc='best')
    plt.savefig(path)
    plt.clf()

def pca(X, y, id=None, cluster=False):
    pca = PCA(n_components=.95, whiten=True, svd_solver='auto', random_state=SEED)
    trans = pca.fit_transform(X)
    vars_ratio = pca.explained_variance_ratio_
    vars = pca.explained_variance_
    print(len(vars))
    print(vars_ratio)
    plt.plot(vars)
    plt.ylabel('Eigenvalues')
    plt.xlabel('Component (total={})'.format(len(vars)))
    plt.title('Components explaining 95% of variance')
    plt.savefig('figures/pca_95%_variance_{}.png'.format(id))
    plt.clf()
    dimredux_scatter(trans, y, 'figures/pca_scatter_{}.png'.format(id), id='PCA')

    # cluster
    print('cluster running\n')
    print('kmeans -------- ')
    kmeans(trans, y, id=id, redux='_PCA', k=len(vars))
    print('EM ----------------')
    em(trans, y, id=id, redux='_PCA', k=len(vars))

    print('ANN running -------------- \n')
    ann(trans, y, id=id, redux='_PCA')

    return trans


def ica(X, y, id=None, cluster=False):
    kurts = []
    recons = []
    highest_kurt = 0
    highest_trans = 0
    upper = int(X.shape[1]/10 * 8)
    for i in range(2, upper):
        print(i/upper)
        ica = FastICA(n_components=i, whiten=True, random_state=SEED, max_iter=10000)
        trans = ica.fit_transform(X)
        proj = ica.inverse_transform(trans)

        recon_error = np.sum(np.square(X - proj).mean())
        recons.append(recon_error)
        kurt = np.mean(kurtosis(trans))
        kurts.append(kurt)
        if kurt > highest_kurt:
            highest_kurt = kurt
            highest_trans = trans

    output = open('figures/ica_{}.txt'.format(id), 'w')
    output.write('Highest kurtosis of {} at {} components\n'.format(max(kurts), np.argmax(kurts) + 3))
    output.close()

    plt.plot(range(2, upper), kurts)
    plt.title('Avg Kurtosis vs Components')
    plt.xlabel('ICA Components')
    plt.ylabel('Avg Kurtosis')
    plt.savefig('figures/ica_avg_kurtosis_{}.png'.format(id))
    plt.clf()

    plt.plot(range(2, upper), recons)
    plt.title('Recon Error')
    plt.ylabel('Error')
    plt.xlabel('Components')
    plt.savefig('figures/ica_recon_error_{}.png'.format(id))
    plt.clf()

    dimredux_scatter(highest_trans, y, 'figures/ica_scatter_{}.png'.format(id), id='ICA')

    ica = FastICA(n_components=134, whiten=True, random_state=SEED, max_iter=10000)
    highest_trans = ica.fit_transform(X)

    #cluster
    print('cluster running\n')
    print('kmeans -------- ')
    kmeans(highest_trans, y, id=id, redux='_ICA', k=np.argmax(kurts))
    print('EM ----------------')
    em(highest_trans, y, id=id, redux='_ICA', k=np.argmax(kurts))

    print('ANN running -------------- \n')
    ann(highest_trans, y, id=id, redux='_ICA')

    return highest_trans

def rp(X, y, id=None, cluster=False):
    recons = []
    kurts = []
    for i in range(2, X.shape[1]):
        print(i/X.shape[1])
        kurts_temp = []
        rp = SparseRandomProjection(i, dense_output=True)
        trans = rp.fit_transform(X)
        inv = np.linalg.pinv(rp.components_.toarray())
        for j in range(9):
            rp = SparseRandomProjection(i, dense_output=True)
            t = rp.fit_transform(X)
            trans += t
            inv += np.linalg.pinv(rp.components_.toarray())
            kurts_temp.append(np.mean(kurtosis(t)))
        recons.append(((X - np.dot(trans/10, inv.T/10))**2).mean())
        kurts.append(np.mean(kurts_temp))

    lowest_err_k = np.argmin(recons) - 1
    best_rp = SparseRandomProjection(lowest_err_k, dense_output=True)
    trans = best_rp.fit_transform(X)

    output = open('figures/rp_{}.txt'.format(id), 'w')
    output.write('Lowest recon error of {} at k={}\n'.format(np.min(recons), lowest_err_k + 3))
    output.close()

    plt.plot(np.arange(2, X.shape[1]), recons, label='Recon error')
    plt.title('Reconstruction Errors for RP')
    plt.xlabel('Components')
    plt.ylabel('MSE')
    plt.savefig('figures/rp_recon_error_{}.png'.format(id))
    plt.clf()

    plt.plot(np.arange(2, X.shape[1]), kurts, label='Kurtosis')
    plt.title('Avg Kurtosis for RP')
    plt.xlabel('Components')
    plt.ylabel('Avg Kurtosis')
    plt.savefig('figures/rp_kurtosis_{}.png'.format(id))
    plt.clf()

    dimredux_scatter(trans, y, 'figures/rp_scatter_{}.png'.format(id), id='RP')

    # cluster
    print('cluster running\n')
    print('EM ----------------')
    em(trans, y, id=id, redux='_RP', k=lowest_err_k)
    print('kmeans -------- ')
    kmeans(trans, y, id=id, redux='_RP', k=lowest_err_k)

    print('ANN running -------------- \n')
    ann(trans, y, id=id, redux='_RP')

    return trans


def svd(X, y, id=None, cluster=False):
    svd = TruncatedSVD(X.shape[1] - 1, random_state=SEED)
    svd.fit(X)
    exp_ratio = svd.explained_variance_ratio_
    summed_ratio = np.cumsum(exp_ratio)
    top = (summed_ratio <= .95).sum()
    print(top)
    # write to file
    output = open('figures/svd_{}.txt'.format(id), 'w')
    output.write('Number features that explain 95%: {}\n'.format(top))
    output.write('Explained ratios: {}\n'.format(exp_ratio))
    output.close()

    trans = svd.transform(X)
    # plot
    plt.plot(range(top), exp_ratio[:top])
    plt.title('Components explaining 95% of variance')
    plt.xlabel('Components')
    plt.ylabel('Ratio of explained variance')
    plt.savefig('figures/svd_95%_variance_{}.png'.format(id))
    plt.clf()
    dimredux_scatter(trans, y, 'figures/svd_scatter_{}.png'.format(id), id='SVD')

    # cluster
    print('cluster running\n')
    print('kmeans -------- ')
    kmeans(trans, y, id=id, redux='_SVD', k=top)
    print('EM ----------------')
    em(trans, y, id=id, redux='_SVD', k=top)

    # ANN
    print('ANN running -------------- \n')
    ann(trans, y, id=id, redux='_SVD')

    return trans

@ignore_warnings(category=ConvergenceWarning)
def ann(X, y, id=None, redux=''):
    split = StratifiedShuffleSplit(n_splits=1, test_size=.33, random_state=SEED)
    for i, j in split.split(X, y):
        y_train, y_test = y[i], y[j]
        X_train, X_test = X[i], X[j]
    cv = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)

    # best prj1 params
    if id == 'e':
        scoring = 'f1'
        nn = MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu', alpha=.01, max_iter=1000, random_state=SEED)
    else:
        scoring = 'roc_auc'
        nn = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic', alpha=.001, max_iter=1000, random_state=SEED)
    nn.fit(X_train, y_train)
    y_probs = nn.predict_proba(X_test)
    y_preds = nn.predict(X_test)

    # plots
    plot_lc(nn, X_train, y_train, cv, 'Old paramater LC', 'figures/nn_oldparams_lc_{}{}.png'.format(id, redux))
    plot_roc_curve(nn, X_test, y_test)
    plt.savefig('figures/nn_oldparams_roc_{}{}.png'.format(id, redux))
    plt.clf()
    plot_confusion_matrix(nn, X_test, y_test, normalize='pred')
    plt.savefig('figures/nn_oldparams_confusion_{}{}.png'.format(id, redux))
    plt.clf()
    clf = classification_report(y_test, y_preds)
    output = open('figures/nn_oldparams_classification_{}{}.txt'.format(id, redux), 'w')
    output.write('Classification report:\n{}'.format(clf))
    output.close()

    # new params
    print('starting gridsearch')
    hidden_layer_sizes = [(15, 15), (10, 10), (15, 10, 10), (25, 25), (20, 20, 20)]
    activation = ['logistic', 'relu']
    alpha = [.001, .01, .1]
    grid = dict(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha)
    model = MLPClassifier(max_iter=1000, random_state=SEED)
    out = GridSearchCV(estimator=model, param_grid=grid, n_jobs=12, cv=cv, scoring=scoring, error_score=0)
    result = out.fit(X_train, y_train)
    best_model = result.best_estimator_
    best_params = result.best_params_
    y_probs = best_model.predict_proba(X_test)
    y_preds = best_model.predict(X_test)

    # plots
    plot_lc(best_model, X_train, y_train, cv, 'New paramater LC', 'figures/nn_newparams_lc_{}{}.png'.format(id, redux))
    plot_roc_curve(best_model, X_test, y_test)
    plt.savefig('figures/nn_newparams_roc_{}{}.png'.format(id, redux))
    plt.clf()
    plot_confusion_matrix(best_model, X_test, y_test, normalize='pred')
    plt.savefig('figures/nn_newparams_confusion_{}{}.png'.format(id, redux))
    plt.clf()
    clf = classification_report(y_test, y_preds)
    output = open('figures/nn_newparams_classification_{}{}.txt'.format(id, redux), 'w')
    output.write('Best params: {}\n'.format(best_params))
    output.write('Classification report:\n{}'.format(clf))
    output.close()


def plot_lc(est, X_train, y_train, cv, title, path):
    train_sizes, train_scores, valid_scores = learning_curve(est, X_train, y_train, n_jobs=12, train_sizes=np.linspace(.1, 1, 10))
    train_scores_mean, valid_scores_mean = train_scores.mean(axis=1), valid_scores.mean(axis=1)
    plt.plot(train_sizes, train_scores_mean, color='red', label='Training Score')
    plt.plot(train_sizes, valid_scores_mean, color='green', label='Validation Score')
    plt.title(title)
    plt.xlabel('Training Samples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(path)
    plt.clf()

# def plot_roc(y_test, y_probs, title, path):
#     fpr, tpr, _ = roc_curve(y_test, y_probs)
#     score = roc_auc_score(y_test, y_probs)
#     plt.plot(fpr, tpr, label='ROC Curve, area={}'.format(score))
#     plt.title(title)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend(loc='lower right')
#     plt.tight_layout()
#     plt.savefig(path)
#     plt.clf()


if __name__ == '__main__':
    d1 = pd.read_csv('epilepsy/epilepsy.csv')
    d2 = pd.read_csv('gamma/gamma.csv')

    # preprocess epilepsy
    temp = np.array(d1['y'].values.tolist())
    d1['y'] = np.where(temp > 1, 0, temp).tolist()
    d1 = d1.drop('Unnamed', axis=1)
    d1x = d1.drop('y', axis=1)
    d1x_ss = StandardScaler().fit_transform(d1x)
    d1y = d1['y']

    #prepross GAMMA
    d2 = d2.rename(columns={'class': 'y'})
    d2 = d2.drop(d2.columns[0], axis=1)
    m = {'g': 1, 'h': 0}
    d2['y'] = d2['y'].map(m)
    d2x = d2.drop('y', axis=1)
    d2x_ss = StandardScaler().fit_transform(d2x)
    d2y = d2['y']

    # make figures dir
    if not os.path.isdir('figures/'):
        os.mkdir('figures')

    x = input('''Choose algorithm:
                    K-Means + ANN: k,
                    Expectation Maximization + ANN: e,
                    PCA + clustering + ANN: p,
                    ICA + clustering + ANN: i,
                    RP + clustering + ANN: r,
                    SVD + clustering + ANN: s: ''')
    y = input('''Choose dataset:
                    Epilepsy: e,
                    GAMMA: g: ''')

    if x == 'k':
        if y == 'e':
            kmeans(d1x_ss, d1y, id=y)
        else:
            kmeans(d2x_ss, d2y, id=y)
    elif x == 'e':
        if y == 'e':
            em(d1x_ss, d1y, id=y)
        else:
            em(d2x_ss, d2y, id=y)
    elif x == 'p':
        if y == 'e':
            pca(d1x_ss, d1y, id=y)
        else:
            pca(d2x_ss, d2y, id=y)
    elif x == 'i':
        if y == 'e':
            ica(d1x_ss, d1y, id=y)
        else:
            ica(d2x_ss, d2y, id=y)
    elif x == 'r':
        if y == 'e':
            rp(d1x_ss, d1y, id=y)
        else:
            rp(d2x_ss, d2y, id=y)
    elif x == 's':
        if y == 'e':
            svd(d1x_ss, d1y, id=y)
        else:
            svd(d2x_ss, d2y, id=y)

    print('done')
    # elif x == 'nn':
    #     if y == 'e':
    #         ann(d1x_ss, d1y, id=y)
    #     else:
    #         ann(d2x_ss, d2y, id=y)
