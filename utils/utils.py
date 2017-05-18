import os
import stat
import shutil
import itertools
import numpy as np
import pandas as pd
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt
import cv2
from IPython.display import SVG
from sklearn.metrics import confusion_matrix
from keras.utils.vis_utils import model_to_dot
from keras_diagram import ascii

def get_non_hidden_dir_contents(dir_path):
    """
    Accepts dir_path -> relative path as string
    Returns a non-recursive list of all file contents
    """
    return [os.path.join(dir_path,f) for f in os.listdir(dir_path) if not f.startswith('.')]

def clear_all(path_):
    """
    Robustly Removes all directories and files under path
    """
    # http://stackoverflow.com/questions/185936/delete-folder-contents-in-python/24844618#24844618
    # http://stackoverflow.com/questions/1889597/deleting-directory-in-python
    def _remove_readonly(fn, path_, excinfo):
        # Handle read-only files and directories
        if fn is os.rmdir:
            os.chmod(path_, stat.S_IWRITE)
            os.rmdir(path_)
        elif fn is os.remove:
            os.lchmod(path_, stat.S_IWRITE)
            os.remove(path_)

    def force_remove_file_or_symlink(path_):
        try:
            os.remove(path_)
        except OSError:
            os.lchmod(path_, stat.S_IWRITE)
            os.remove(path_)

    # Code from shutil.rmtree()
    def is_regular_dir(path_):
        try:
            mode = os.lstat(path_).st_mode
        except os.error:
            mode = 0
        return stat.S_ISDIR(mode)
   
    if is_regular_dir(path_):
        # Given path is a directory, clear its content
        for name in os.listdir(path_):
            fullpath = os.path.join(path_, name)
            if is_regular_dir(fullpath):
                shutil.rmtree(fullpath, onerror=_remove_readonly)
            else:
                force_remove_file_or_symlink(fullpath)
    else:
        # Given path is a file or a symlink.
        # Raise an exception here to avoid accidentally clearing the content
        # of a symbolic linked directory.
        raise OSError("Cannot call clear_dir() on a symbolic link")
        
def copy_dir_structure(from_dir, to_dir):
    """
    Accepts from_dir -> relative path as string; to_dir -> relative path as string
    Copy directory structure from from_dir and recreates it in to_dir
    Non-recursive - Only works at first level
    """
    dirs_to_create = [dir_name.split('/')[-1] for dir_name in get_non_hidden_dir_contents(from_dir)]
    for sub_dir in dirs_to_create:
        newpath = os.path.join(to_dir, sub_dir)
        if not os.path.exists(newpath):
            os.makedirs(newpath) 
            
def sample_files(from_dir, to_dir, sample_proportion = 0.1, copy_only=True):
    """
    Accepts from_dir -> relative path as string; to_dir -> relative path as string;
    sample_proportion -> float; copy_only as Boolean True - copy files; False - move files
    Moves or Copies a sample of files from each sub dir under from_dir to each sub_dir in to_dir
    all dirs must exist  Only works at one level down - non-recursive
    """
    # Sub Directiories are classes for classification problem
    from_sub_dirs = get_non_hidden_dir_contents(from_dir)
    # Form sub directories for copy_to path
    to_sub_dirs = [os.path.join(to_dir, f.split('/')[-1]) for f in from_sub_dirs]
    # Get a list of list of files in each from class directory
    for from_d, to_d in zip(from_sub_dirs, to_sub_dirs):
        d_files = np.array(get_non_hidden_dir_contents(from_d))
        # Random shuffle the files and select a proportion
        np.random.shuffle(d_files)
        sample_files = d_files[0:int(len(d_files)*sample_proportion)]
        for f in sample_files:
            to_file_path = os.path.join(to_d, f.split('/')[-1])
            if copy_only==True:
                shutil.copyfile(f, to_file_path) 
            else:
                shutil.move(f, to_file_path)

def get_random_sample_images(rel_dir_path, num_images):
    """
    Accepts relative directory path as string e.g 'data/train/cat/' - non recursive
    and num_images as integer
    Returns a list of numpy arrays
    Uses cv2.imread to read the arrays    
    """
    #class_name = rel_dir_path.split('/')[0]
    random_sample_images = []
    image_paths = get_non_hidden_dir_contents(rel_dir_path)
    #image_paths = get_non_hidden_dir_contents(rel_dir_path)
    np.random.shuffle(image_paths)
    sample_paths = image_paths[0:num_images]
    class_names = [n.split('/')[-2] for n in sample_paths] # Get class name from directory name
    random_sample_images = [cv2.imread(img_path) for img_path in sample_paths]
    return random_sample_images, class_names

def plot_image_grid(image_list, true_class_list=None, pred_class_list=None, pred_prob_list=None, cols = 4):
    """
    Accepts a list of numpy arrays for images,
    a list of classes to identify the images, and and a column width - integer
    plots a 4 wide  by ? rows grid of images
    Subplots are organized in a Rows x Cols Grid
    Total number of images and num columns drives layout
    """
    tot = len(image_list) # number of subplots
    # Temporarily for testing
    pred_list = []
    
    # Compute Rows required
    rows = tot // cols 
    rows += tot % cols
    
    # Create a Position index
    position = range(1,tot + 1)

    fig_width = 12
    fig_height = (tot // cols) * 6 # set by trial and error 4 to 40 photos in grid
    fig = plt.figure(1, figsize=(fig_width,fig_height))
    for indx,img in enumerate(image_list):
        img_text = ''
        if true_class_list:
            img_text += 'T: ' + str(true_class_list[indx])
        if pred_class_list:
            img_text += ' P: ' + str(pred_class_list[indx])
        if pred_prob_list:
            img_text += ' Pp: ' + str(round(pred_prob_list[indx],2))
        ax = fig.add_subplot(rows,cols,position[indx])
        ax.set_title(img_text)
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #plt.imshow(img)
    fig.tight_layout()
    plt.show();
    
def plot_confusion_matrix(y_true, y_pred, array_labels, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    uses sklearn to get the confusion matrix
    prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    cm = confusion_matrix(y_true, y_pred, labels=array_labels)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(array_labels))
    plt.xticks(tick_marks, array_labels, rotation=45)
    plt.yticks(tick_marks, array_labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_learning(h, steps=None):
    if steps:
        steps = h.params['steps']
    else:
        steps='n/a'
    epochs = h.epoch
    training_metric = h.params['metrics'][1]
    training_metric_values = h.history[training_metric]
    plt.plot(epochs, training_metric_values, label = training_metric)
    if h.params['do_validation']:
        val_metric = h.params['metrics'][3]
        val_metric_values = h.history[val_metric]
        plt.plot(epochs, val_metric_values, marker='o', linestyle='--', color='r', label= val_metric)
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title(h.model.name + ' learning (steps: ' + str(steps) + ')')
    plt.legend(loc='center left')
    plt.show;

def show_model(model):
    return SVG(model_to_dot(model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))

def show_ascii_model(model):
    print(ascii(model))

# Diagnostics
# Histogram of predicted probabilities
def plot_response_distribution_known_class(type1_preds, type2_preds, type3_preds):
    """
    Accepts numpy arrays of predictions
    Plots distributions of probability predictions for each class
    i.e Plots probability distribution of probabilities
    """
    # Lucas Javier Bernardi | Diagnosing Machine Learning Models - https://www.youtube.com/watch?v=ZD8LA3n6YvI
    all_preds = np.concatenate((type1_preds, type2_preds, type3_preds), axis=0)
    fig = plt.figure(figsize=(12,5))
    ax1 = plt.subplot(221)
    ax1.hist(type1_preds,bins = 10,normed=True, color = 'r')
    ax1.set_xlim(0,1)
    ax1.set_xticklabels([])
    ax1.set_title('Type 1')
    
    ax2 = plt.subplot(222)
    ax2.hist(type2_preds,bins = 10,normed=True, color='g')
    ax2.set_xlim(0,1)
    ax2.set_xticklabels([])
    ax2.set_title('Type 2')

    ax3 = plt.subplot(223)
    ax3.hist(type3_preds,bins = 10,normed=True,color='b')
    ax3.set_xlim(0,1)
    ax3.set_title('Type 3')
    
    ax4 = plt.subplot(224)
    for dist,col in zip([type1_preds, type2_preds, type3_preds], ['r','g','b']):
        ax4.hist(dist, color=col, bins = 10, normed=True  )
    ax4.set_xlim(0,1)
    ax4.set_title('All Types')
    plt.show();

def form_submission_df(preds, filenames):
    """
    Accepts a numpy array of predictions, and a array of filenames
    Forms a dataframe in the same format as a submission.csv file
    """
    df = pd.DataFrame(preds, columns = ['Type_1','Type_2', 'Type_3'])
    df['image_name'] = [fn.split('/')[1] for fn in filenames]
    df = df[['image_name','Type_1','Type_2', 'Type_3'] ]
    df['temp_sort_col'] = df['image_name'].apply(lambda x: int(x.split('.')[0]))
    df = df.sort_values(by=['temp_sort_col'], ascending=True)
    df = df.drop('temp_sort_col', axis=1)
    return df

def save_submission_csv(submission_dir, sub_df):
    """
    Accepts the directory for submission csv files, and a dataframe fromated as a submission
    Increments the file number by 1 and saves in the submissions directory
    """
    try:
        submission_path = os.path.abspath(submission_dir)
        subs = get_non_hidden_dir_contents(submission_dir)
        last_sub_num = max([int(fn.split('.')[0][-3:]) for fn in subs])
        new_sub_name = 'sub_' + ('0'* (3 - len(str(last_sub_num+1)))) + str(last_sub_num + 1) + '.csv'
        new_path = os.path.join(submission_dir, new_sub_name)
        #print(new_path)
        sub_df.to_csv(new_path,index=False)
        print('Saved to ' , new_path)
    except:
        print('Problem Saving File')
        
# Calculate class weights
# https://groups.google.com/forum/#!topic/keras-users/MUO6v3kRHUw
def get_class_weight_dict(num_per_class_list):
    max_class_num = max(num_per_class_list)
    class_weight_dict ={}
    for n, num in zip(range(0, len(num_per_class_list)), num_per_class_list):
        class_weight_dict[n] = max_class_num / float(num)
    return class_weight_dict
