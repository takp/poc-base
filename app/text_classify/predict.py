import tensorflow as tf
from tensorflow.python.lib.io import file_io
import os, sys, time, datetime, csv, math, random
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
import argparse

model_dir_path = os.path.join(os.path.dirname(__file__), "./saved_models/")
model_name = "test_model_180814_113741"
model_path_def = model_dir_path + model_name

tweet_data_path =  os.path.join(os.path.dirname(__file__), "./data/tweets_clean_2k.csv")
w2v_data_path =  os.path.join(os.path.dirname(__file__), "./data/GoogleNews-vectors-negative300_50k.bin")

parser = argparse.ArgumentParser()
parser.add_argument('--model',   dest="modelpath", help='set model path', default=model_path_def)
parser.add_argument('--dataset', dest="datasetpath", help='set dataset path', default=tweet_data_path)
parser.add_argument('--w2v',     dest="w2vpath", help='set w2v model path', default=w2v_data_path)
args = parser.parse_args()
model_path   = args.modelpath
dataset_path = args.datasetpath
w2v_path     = args.w2vpath

class_count = 3
locked_class_count = 3

# w2v_limit = 100000
w2v_limit = None

time_start = time.time()
w2v_model = []
with file_io.FileIO(w2v_path, 'rb') as f:
    w2v_model = KeyedVectors.load_word2vec_format(f, limit=w2v_limit, binary=True, unicode_errors='ignore')

w2v_dim = w2v_model.wv.vector_size
w2v_words = set(w2v_model.wv.index2word)
time_finish = time.time()
time_elapsed = np.round((time_finish - time_start)*1000)/1000
w2v_word_count = len(w2v_model.wv.index2word)
print("Word2Vec model loaded:", w2v_word_count, "words in", time_elapsed, "seconds")

stopwords = set()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph(model_path + '.meta')
saver.restore(sess, model_path)
ip = tf.get_default_graph().get_tensor_by_name("ip:0")
ip_len = tf.get_default_graph().get_tensor_by_name("ip_len:0")
op = tf.get_default_graph().get_tensor_by_name("op:0")
# pred = tf.get_default_graph().get_tensor_by_name("pred:0")
# pred_score = tf.get_default_graph().get_tensor_by_name("pred_value_score:0")
# pred_mag = tf.get_default_graph().get_tensor_by_name("pred_mag:0")
pred_mag = tf.get_default_graph().get_tensor_by_name("pred_score:0")
accuracy = tf.get_default_graph().get_tensor_by_name("acc_score:0")
pred_score_class = tf.get_default_graph().get_tensor_by_name("pred_score_class:0")
pred_score = tf.get_default_graph().get_tensor_by_name("pred_score:0")
# label_score_class = tf.get_default_graph().get_tensor_by_name("label_score_class:0")

dropout_rate      = tf.get_default_graph().get_tensor_by_name("dropout_rate:0")
dropout_op_reset  = tf.get_default_graph().get_operation_by_name("dropout_op_reset")
dropout_op_remove = tf.get_default_graph().get_operation_by_name("dropout_op_remove")
sess.run(dropout_op_remove)

conf_label = tf.placeholder(tf.float32, [None], name="conf_label")
conf_pred  = tf.placeholder(tf.float32, [None], name="conf_pred")
conf_matrix_static = tf.confusion_matrix(conf_label, conf_pred)


def wordFilterCheck(word=""):
    if len(word) < 1: return False
    if word not in w2v_words: return False
    if any([ch.isdigit() for ch in word]): return False
    # if word in stopwords: return False
    # if word in stopwords_en: return False
    # if word in stopwords_jp: return False
    return True

def words2Matrix(words=[], wordLenFixed=-1):
    if type(words) == str: words = words.split()
    if type(words) not in [list, np.ndarray]: return np.array([]), 0
    wordLenMax = 5000
    wordLenMin = 6
    if wordLenFixed >= 1:
        wordLenMax = wordLenFixed
        wordLenMin = wordLenFixed
    
    matrix = np.zeros([wordLenMax, w2v_dim])
    word_index = -1
    for w in words:
        word_index += 1
        if word_index >= wordLenMax and wordLenMax >= 0:
            continue
        if not wordFilterCheck(w):
            continue
        matrix[word_index] = np.array(w2v_model.wv[w])
    
    textWordLen = word_index + 1
    if textWordLen < wordLenMin: textWordLen = wordLenMin
    matrix = matrix[0 : textWordLen]
    # matrix = np.transpose(matrix)
    return matrix, textWordLen

def data_get_mini_batch_info(data_wordcount=[], total_cutoff=800, len_cutoff=50):
    mini_batch_info = []
    index_start = 0
    index_end = 0
    for i in range(len(data_wordcount)):
        index_end = i + 1
        index_len = index_end - index_start
        wc_max = max(data_wordcount[index_start : index_end])
        wc_total_padded = wc_max * index_len
        if wc_total_padded >= total_cutoff or index_len >= len_cutoff or index_end == len(data_wordcount):
            mini_batch_info.append([index_start, index_end, index_len, wc_max, wc_total_padded])
            # print(mini_batch_info[-1])
            index_start = index_end
    return mini_batch_info

def predictSent_batch_token(batch_text=[], pad_unknown_words=False):
    wordLenMin = 6
    wordLenMax = 1000
    scores = np.random.rand(0)
    mags = np.random.rand(0)
    if type(batch_text) == str: batch_text = [batch_text]
    if type(batch_text) not in [list, np.ndarray]: return scores, mags
    # batch_word = [[w for w in t.split() if wordFilterCheck(w)] for t in batch_text]
    batch_word = [
        [
            w
            for w in t.split()
            if (wordFilterCheck(w) or pad_unknown_words)
        ]
        for t in batch_text
        if type(t) == str
    ]
    batch_full = [[
        words,
        i,      # original index
        min(max(len(words), wordLenMin), wordLenMax),
        0.0,    # pred_score placeholder
        0.0     # pred_mag placeholder
    ] for i, words in enumerate(batch_word)]
    batch_full_sorted = sorted(batch_full, key=lambda v: v[2])
    batch_wordcount_sorted = [v[2] for v in batch_full_sorted]
    batch_word_sorted = [v[0] for v in batch_full_sorted]
    
    mini_batch_info = data_get_mini_batch_info(batch_wordcount_sorted)
    totalLen = len(batch_word)
    batch_feed_mini = []
    for mini_info in mini_batch_info:
        batch_word_mini = batch_word_sorted[mini_info[0] : mini_info[1]]
        batch_len_mini = mini_info[2]
        max_wordcount_mini = mini_info[3]
        batch_word_mini_in  = np.zeros([batch_len_mini, max_wordcount_mini, w2v_dim])
        batch_word_mini_in_len = np.zeros([batch_len_mini])
        for i in range(batch_len_mini):
            text_matrix, matrix_word_len = words2Matrix(batch_word_mini[i], max_wordcount_mini)
            batch_word_mini_in[i] = text_matrix
            batch_word_mini_in_len[i] = max([matrix_word_len, wordLenMin])
        batch_feed_mini.append({ip : batch_word_mini_in, ip_len : batch_word_mini_in_len})
    pred_outputs = [sess.run([pred_score, pred_mag], b) for b in batch_feed_mini]
    score_outputs = np.concatenate([v[0] for v in pred_outputs], axis=0)
    mag_outputs   = np.concatenate([v[1] for v in pred_outputs], axis=0)
    for i in range(totalLen):
        batch_full_sorted[i][3] = score_outputs[i]
        batch_full_sorted[i][4] = mag_outputs[i]
    batch_full_rearanged = sorted(batch_full_sorted, key=lambda v: v[1])
    scores = np.round([v[3] for v in batch_full_rearanged], 1)
    mags   = np.round([v[4] for v in batch_full_rearanged], 1)
    return scores, mags

def calcConfusionMatrix(labeled_scores=[], predicted_scores=[], rawCount=False, ifPrint=True):
    batch_conf = {conf_label: labeled_scores, conf_pred: predicted_scores}
    conf_matrix_output = sess.run(conf_matrix_static, batch_conf)
    ratio_avg = np.array([0] * 5)
    ratio_avg_class_check = np.array([0] * locked_class_count)
    txt = "[Confusion Matrix - Score]"
    txt += "\n    |"
    for i in range(locked_class_count):
        txt += format(i * .1 - 1., "5.1f") + "|"
        
    for i in range(len(conf_matrix_output)):
        row = conf_matrix_output[i]
        txt += "\n" + format(i * .1 - 1., "4.1f") + "|"
        row_sum = np.sum(row)
        for j in range(len(row)):
            r = 0
            if row_sum > 0:
                r = row[j] / row_sum * 100
                ratio_avg_class_check[i] = 1
                for k in range(len(ratio_avg)):
                    if k >= abs(j-i):
                        ratio_avg[k] += r
            
            if r >= 1.5 or i == j and row_sum > 0:
                r_format = format(r, "3.0f")
                if i == j:
                    txt += "<" + r_format + ">"
                else:
                    txt += " " + r_format + " "
                
            else:
                txt += format("", "5")
            
            txt += "|"
    
    ratio_avg = ratio_avg / np.sum(ratio_avg_class_check)
    if ifPrint: print(txt)
    return conf_matrix_output, ratio_avg, txt

def checkAccuracy_old(data=[], ifPrint=True, rawCount=False):
    data_len = len(data)
    labeled_class   = (np.array([v[1] for v in data]) + 1) * 10
    predicted_class = (np.array([v[3] for v in data]) + 1) * 10
    dif_class = np.abs(labeled_class - predicted_class)
    correct_class = np.array([np.sum(dif_class <= i) / data_len for i in range(4)])
    cfmt, ratio_avg, txt_cfmt = calcConfusionMatrix(labeled_class, predicted_class, rawCount=rawCount, ifPrint=False)
    txt_info    = "score acc : "
    txt_acc     = "     full : "
    txt_acc_avg = "      avg : "
    for i in range(len(correct_class)):
        if i > 0:
            txt_info += " | "
            txt_acc += " | "
            txt_acc_avg += " | "
        txt_info += "  +-0." + format(i, "1") + ""
        txt_acc  += format(correct_class[i] * 100, "6.2f") + "%"
        txt_acc_avg  += format(ratio_avg[i], "6.2f") + "%"
    
    txt = "\n".join([txt_info, txt_acc, txt_acc_avg])
    if ifPrint:
        print(txt)
        print(txt_cfmt)
    return correct_class, cfmt

def score_simple_class(v=0.0):
    if v < -0.1: return 0
    if v >  0.1: return 2
    return 1

def score_simple_value_map(v=0.0):
    simple_class = score_simple_class(v)
    if simple_class not in range(3): simple_class = 1
    return [ -1.0, 0.0, 1.0 ][simple_class]

def checkAccuracy(data=[], rawCount=False):
    data_len = len(data)
    labeled_class   = np.array([score_simple_class(v[1]) for v in data])
    predicted_class = np.array([score_simple_class(v[3]) for v in data])         
    # dif_class = np.abs(labeled_class - predicted_class)
    cf_matrix = np.zeros([locked_class_count, locked_class_count])
    for i in range(data_len):
        cf_matrix[int(labeled_class[i])][int(predicted_class[i])] += 1
    
    correct_ratio     = np.sum([cf_matrix[i][i] for i in range(len(cf_matrix))]) / np.sum(cf_matrix)
    correct_ratio_avg = np.mean([cf_matrix[i][i]/np.sum(cf_matrix[i]) for i in range(len(cf_matrix))])
    txt_info  = "score accuracy: [total] " + format(correct_ratio * 100, "6.2f") + "%"
    txt_info += "  - [avg per class] " + format(correct_ratio_avg * 100, "6.2f") + "%"
    group_labels = [
        ["neg", "neu", "pos"],
        [-1, 0, 1],
    ][0]
    txt = "\n".join([txt_info])
    txt += "\n        " + "".join([format(gl, ">8") for gl in group_labels])
    for i in range(locked_class_count):
        txt += "\n" + format(group_labels[i], ">8")
        for j in range(locked_class_count):
            r = cf_matrix[i][j]
            if not rawCount:
                r = 0
                row_sum = sum(cf_matrix[i])
                if row_sum > 0:
                    r = cf_matrix[i][j] / row_sum * 100
                    txt += " " + format(r, "7.2f")
                    continue
            txt += " " + format(r, "7")
    print(txt)
    return cf_matrix

def processDataset(dataset=[], pad_unknown_words=False):
    dataset_len = len(dataset)
    dataset_texts = [e[0] for e in dataset]
    pred_scores, pred_mags = predictSent_batch_token(dataset_texts, pad_unknown_words=pad_unknown_words)
    label_scores = []
    for i in range(dataset_len):
        label_scores.append(dataset[i][1])
        dataset[i][3] = pred_scores[i]
        dataset[i][4] = pred_mags[i]
    return dataset

def processFile_read(filepath="", limit=None, pattern=[], skip_lines=0):
    # pattern [ 0:text_token, 1:score, 2:mag ]
    if type(limit) not in [int]: limit = -1
    dataset = []
    with open(filepath, encoding="utf-8") as f:
        data = csv.reader(f)
        line_index = -1
        for line in data:
            # if line_index >= 10: break
            if len(line) < 1:
                # print(line)
                continue
            line_merged = " ".join([str(v) for v in line])
            if len(line_merged) < 3: continue
            line_index += 1
            if limit >= 0 and len(dataset) >= limit: break
            if line_index < skip_lines: break
            label_score = int(float(line_merged[0]))
            text_token  = line_merged[2:]
            # print(line_merged)
            dataset.append([text_token, label_score, label_score, 0.0, 0.0])

    return dataset

def processFile(
    filepath          = "./data/tweets_clean_2k.csv",
    output_filepath   = "./data/predict_output.csv",
    limit             = None,
    pattern           = [],
    skip_lines        = 0,
    pad_unknown_words = False
):
    fileFound = False
    if os.path.isfile(filepath):
        fileFound = True
    else:
        alt_filepath = "data/" + filepath
        if os.path.isfile(alt_filepath):
            filepath = alt_filepath
            fileFound = True
    if not fileFound:
        print("[ERROR] File Not Found:", filepath)
        return False
    print("Found File:", filepath)
    if type(pattern) not in [list, np.ndarray] or len(pattern) < 3:
        pattern = [i for i in range(3)]
    dataset = processFile_read(filepath, limit, pattern, skip_lines)
    dataset_len = len(dataset)
    print("Done reading data [" + str(dataset_len) + "]")
    
    print("Processing...")
    dataset = processDataset(dataset, pad_unknown_words=pad_unknown_words)
    
    if type(output_filepath) == str and len(output_filepath) > 0:
        print("writing to file:", output_filepath)
        datawrite = dataset
        with open(output_filepath, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar='\\')
            for line in datawrite:
                wr.writerow(line)
    
    return dataset


# [TESTING]
# dataset = processFile("data/tweets_clean_2k.csv", "")

# python predict.py "data/token-combined-text-sentiment-magnitude.csv"
# processFile("data/token-combined-text-sentiment-magnitude.csv")
# dataset = processFile("data/token-combined-text-sentiment-magnitude.csv", "")
# dataset = processFile("data/scored-messages.csv", "", limit=6000, skip_lines=1, pattern=[5,2,3])
# dataset = processFile("scored-sentences-201711.csv", "")
# dataset = processFile("scored-sentences.csv", "", limit=40000)
# dataset = processFile("scored-sentences-20180703_200k.csv", "", limit=20000, skip_lines=70000)
# dataset = processFile("scored-sentences-20180703_200k.csv", "", limit=20000, skip_lines=70000, pad_unknown_words=True)
# _ = checkAccuracy(dataset)
# _ = checkAccuracy(dataset, rawCount=True)
# _ = checkAccuracy_mag(dataset)
# _ = checkAccuracy_old(dataset)

