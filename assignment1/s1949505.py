"""
Foundations of Natural Language Processing

Assignment 1

Please complete functions, based on their doc_string description
and instructions of the assignment. 

To test your code run:

  [hostname]s1234567 python3 s1234567.py
or
  [hostname]s1234567 python3 -i s1234567.py

The latter is useful for debugging, as it will allow you to access many
 useful global variables from the python command line

*Important*: Before submission be sure your code works _on a DICE machine_
with the --answers flag:

  [hostname]s1234567 python3 s1234567.py --answers

Also use this to generate the answers.py file you need for the interim
checker.

Best of Luck!
"""
from asyncio import MultiLoopChildWatcher
from audioop import mul
from collections import defaultdict, Counter
from pkgutil import get_loader
from typing import Tuple, List, Any, Set, Dict, Callable

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk import MaxentClassifier
from nltk.corpus import brown, ppattach  # import corpora

# Import LgramModel
from nltk_model import *

# Import the Twitter corpus
from twitter.twitter import *

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()


# Some helper functions

import matplotlib.pyplot as plt

def hist(hh: List[float], title: str, align: str = 'mid',
         log: bool = False, block: bool = False):
  """
  Show a histgram with bars showing mean and standard deviations
  :param hh: the data to plot
  :param title: the plot title
  :param align: passed to pyplot.hist, q.v.
  :param log: passed to pyplot.hist, q.v.  If present will be added to title
  """
  hax=plt.subplots()[1] # Thanks to https://stackoverflow.com/a/7769497
  sdax=hax.twiny()
  hax.hist(hh,bins=30,color='lightblue',align=align,log=log)
  hax.set_title(title+(' (log plot)' if log else ''))
  ylim=hax.get_ylim()
  xlim=hax.get_xlim()
  m=np.mean(hh)
  sd=np.std(hh)
  sdd=[(i,m+(i*sd)) for i in range(int(xlim[0]-(m+1)),int(xlim[1]-(m-3)))]
  for s,v in sdd:
       sdax.plot([v,v],[0,ylim[0]+ylim[1]],'r' if v==m else 'pink')
  sdax.set_xlim(xlim)
  sdax.set_ylim(ylim)
  sdax.set_xticks([v for s,v in sdd])
  sdax.set_xticklabels([str(s) for s,v in sdd])
  plt.show(block=block)


def compute_accuracy(classifier, data: List[Tuple[List, str]]) -> float:
    """
    Computes accuracy (range 0 - 1) of a classifier.
    :type classifier: e.g. NltkClassifierWrapper or NaiveBayes
    :param classifier: the classifier whose accuracy we compute.
    :param data: A list with tuples of the form (list with features, label)
    :return accuracy (range 0 - 1).
    """
    correct = 0
    for d, gold in data:
        predicted = classifier.classify(d)
        correct += predicted == gold
    return correct/len(data)


def apply_extractor(extractor_f: Callable[[str, str, str, str, str], List[Any]], data: List[Tuple[Tuple[str], str]])\
        -> List[Tuple[List[Any], str]]:
    """
    Helper function:
    Apply a feature extraction method to a labeled dataset.
    :param extractor_f: the feature extractor, that takes as input V, N1, P, N2 (all strings) and returns a list of features
    :param data: a list with tuples of the form (id, V, N1, P, N2, label)

    :return a list with tuples of the form (list with features, label)
    """
    r = []
    for d in data:
        r.append((extractor_f(*d[1:-1]), d[-1]))
    return r


def get_annotated_tweets():
    """
    :rtype list(tuple(list(str), bool))
    :return: a list of tuples (tweet, a) where tweet is a tweet preprocessed by us,
    and a is True, if the tweet is in English, and False otherwise.
    """
    import ast
    with open("twitter/annotated_dev_tweets.txt") as f:
        return [ast.literal_eval(line) for line in f.readlines()]


class NltkClassifierWrapper:
    """
    This is a little wrapper around the nltk classifiers so that we can interact with them
    in the same way as the Naive Bayes classifier.
    """
    def __init__(self, classifier_class: nltk.classify.api.ClassifierI, train_features: List[Tuple[List[Any], str]], **kwargs):
        """

        :param classifier_class: the kind of classifier we want to create an instance of.
        :param train_features: A list with tuples of the form (list with features, label)
        :param kwargs: additional keyword arguments for the classifier, e.g. number of training iterations.
        :return None
        """
        self.classifier_obj = classifier_class.train(
            [(NltkClassifierWrapper.list_to_freq_dict(d), c) for d, c in train_features], **kwargs)

    @staticmethod
    def list_to_freq_dict(d: List[Any]) -> Dict[Any, int]:
        """
        :param d: list of features

        :return: dictionary with feature counts.
        """
        return Counter(d)

    def classify(self, d: List[Any]) -> str:
        """
        :param d: list of features

        :return: most likely class
        """
        return self.classifier_obj.classify(NltkClassifierWrapper.list_to_freq_dict(d))

    def show_most_informative_features(self, n = 10):
        self.classifier_obj.show_most_informative_features(n)

# End helper functions

# ==============================================
# Section I: Language Identification [60 marks]
# ==============================================

# Question 1.1 [7.5 marks]
def train_LM(corpus: nltk.corpus.CorpusReader) -> LgramModel:
    """
    Build a bigram letter language model using LgramModel
    based on the lower-cased all-alpha subset of the entire corpus

    :param corpus: An NLTK corpus

    :return: A padded letter bigram model based on nltk.model.NgramModel
    """
    # Return the tokens and a smoothed (using the default estimator) padded bigram letter language model

    # subset the corpus to only include all-alpha tokens,
    # converted to lower-case (_after_ the all-alpha check)
    corpus_tokens = []
    for x in corpus.words():
        if x.isalpha() == True:
            corpus_tokens.append(x.lower())

    lm = LgramModel(n=2,train=corpus_tokens, pad_left=True, pad_right=True)
    


    return lm

#lm_brown = train_LM(brown)

# Question 1.2 [7.5 marks]
def tweet_ent(file_name: str, bigram_model: LgramModel) -> List[Tuple[float, List[str]]]:
    """
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens, then converted to lowercase

    :param file_name: twitter file to process

    :return: ordered list of average entropies and tweets"""

    # Clean up the tweet corpus to remove all non-alpha
    # tokens and tweets with less than 5 (remaining) tokens, converted
    # to lowercase

    # Return a list of tuples of the form: (entropy,tweet)
    #  for each tweet in the cleaned corpus, where entropy is the
    #  average per_item bigram entropy of the tokens in the tweet.
    #  The list should be sorted by entropy.
    lm = bigram_model
    
    list_of_tweets = xtwc.sents(file_name)
    cleaned_list_of_tweets = []
    for tweet in list_of_tweets:
        clean_tweet = []
        for word in tweet:
            if word.isalpha() == True:
                clean_tweet.append(word.lower())
        if len(clean_tweet) > 4:
            cleaned_list_of_tweets.append(clean_tweet)

    
    #create a list of entropies for each tweet and append the corresponding entropy to tweet in a final list

    entropy_list = []
    for tweet in cleaned_list_of_tweets:
        ents = []
        for word in tweet:
            ent = lm.entropy(word, perItem=True)
            ents.append(ent)
        tweet_ent = sum(ents) / len(ents)
        entropy_list.append([tweet_ent, tweet])

    #do this to speed up 1.6
    global ent_list 
    ent_list = [word[0] for word in entropy_list]


    entropy_list.sort()

    return entropy_list



# Question 1.3 [3 marks]
def short_answer_1_3() -> str:

    """
    Briefly explain what left and right padding accomplish and why
    they are a good idea. Assuming you have a bigram model trained on
    a large enough sample of English that all the relevant bigrams
    have reliable probability estimates, give an example of a string
    whose average letter entropy you would expect to be (correctly)
    greater with padding than without and explain why.
   
    :return: your answer
    """
    return inspect.cleandoc("""Padding in N-gram models provide a mean to estimate the likelihood of features appearing at the end or beginning of text without depending on a unigram model.\n\
                             Left padding improves estimates for starting features of text, right padding improves estimates for ending text.\n\
                             "Starts" would have higher entropy with padding as "s" is the most common letter at both the start and end of a word.""") 
# Question 1.4 [3 marks]
def short_answer_1_4() -> str:
    """
    Explain the output of lm.entropy('bbq',verbose=True,perItem=True)
    See the Coursework 1 instructions for details.

    :return: your answer
    """
    return inspect.cleandoc("""p(b|('<s>',)) = [2-gram] 0.046511 Bigram probability of b given <s>
                            p(b|('b',)) = [2-gram] 0.007750 Bigram probability of b given b 
                            backing off for ('b', 'q') Operation performs backoff for b and q
                            p(q|()) = [1-gram] 0.000892 Unigram probability of q being in the word
                            p(q|('b',)) = [2-gram] 0.000092 Bigram probability of q given b
                            p(</s>|('q',)) = [2-gram] 0.010636 Bigram probability of <s> given  q
                            7.85102054894183 approximate cross-entropy for the word including padding
                            This set of results shows the probability for each letter in bbq using the trained language model built earlier displaying results for both unigram and bigrams.""")
# Question 1.5 [3 marks]
def short_answer_1_5() -> str:


    """
    Inspect the distribution of tweet entropies and discuss.
    See the Coursework 1 instructions for details.

    :return: your answer
    """
    global ents
    # Uncomment the following lines when you are ready to work on this.
    # Please comment them out again or delete them before submitting.
    # Note that you will have to close the two plot windows to allow this
    # function to return.

    #just_e = [e for (e,tw) in ents]
    #hist(just_e,"Bi-char entropies from cleaned twitter data")
    #hist(just_e,"Bi-char entropies from cleaned twitter data",
    #    log=True,block=True)

    #75 words 

    return inspect.cleandoc(""" These plots show the average entropy across all tweets and plots the number of tweets with the same entropy across 30 bins. 
                            Referring to the log-scaled graph, around 3-5 is where most tweets lie, and it slopes down to 8-10 where the least tweets have such entropies. 
                            After this, more tweets have increasing entropies until ~18. Higher entropy past the minimum may suggest the tweet may not be English or use non-alphabetic characters.""")


# Question 1.6 [10 marks]
def is_English(bigram_model: LgramModel, tweet: List[str]) -> bool:
    """
    Classify if the given tweet is written in English or not.

    :param bigram_model: the bigram letter model trained on the Brown corpus
    :param tweet: the tweet
    :return: True if the tweet is classified as English, False otherwise
    """

    #inital training for tweets using bigram model provided
    #this gives us a mean and standard deviation to fit unseen data to. Anything over the mean + std will be classified as non-english

    #This is why it takes so long - do global variable (1.2) 
    #tweet_ents = tweet_ent(twitter_file_ids, bigram_model)
    #ent_list = [word[0] for word in tweet_ents]


    #ent_list defined as global in 1.2 for this question to speed it up
    threshold = np.mean(ent_list) + np.std(ent_list)

    is_it = True

    #initial test of characters - anything non-ascii likely to not be english
    for x in tweet:
        if x.isascii() == False:
            return False
    
    
    #get entropy for new unseen tweets and average entropy for a tweet
    
    #ent = lm.entropy(tweet, perItem =True)

    ents = []
    for word in tweet:
        ent = bigram_model.entropy(word, perItem=True)
        ents.append(ent)
    tweet = sum(ents) / len(ents)
    
    #print(tweet)
    #print(threshold)

    if tweet > threshold:    
        is_it = False
   
    return is_it
    
###
# Question 1.7 [16 marks]

def essay_question():

    """

    THIS IS AN ESSAY QUESTION WHICH IS INDEPENDENT OF THE PREVIOUS
    QUESTIONS ABOUT TWITTER DATA AND THE BROWN CORPUS!

    See the Coursework 1 instructions for a question about the average
    per word entropy of English.
    1) Name 3 problems that the question glosses over
    2) What kind of experiment would you perform to get a better estimate
       of the per word entropy of English?

    There is a limit of 400 words for this question.
    :return: your answer
    """
    return inspect.cleandoc("""
    Problem 1: English is expansive, what words are included in the domain? Do the results want to reflect abbreviations, slang and words that have fallen out of use or into fashion? 
    It is important to define a vocabulary that is representative of what we are interested in. Considering homonyms is also important as for this we would need 
    contextual clues to realise the intended meaning of the word, that is, if we are to consider these words different.

    Problem 2: Per word entropy changes depending on context, so to do this across a language depends on the 
    order of words and how they are presented. The data we use will greatly impact the result we get. For
    example, do we use a massive set of professionally written text, or do we include writings that are informal. 
    Including fictional novels poses another point of dealing with made up words so the data we use is something we must consider.  
    
    Problem 3: How do we ensure that every word that we have stated should be on the vocabulary will be present in the input data for training and testing.
    Given we must cover every word in the English language, some entropies will be extremes on the scale and will result
    in a skewed result to a larger average entropy so do we want to remove such outliers or are we willing to display results in full.  

    Experiment: 

    Data: web-scraped; comprising of written articles, published papers, informal sources (social media, discussion boards etc.), freely available online books and sources of transcribed speech from interviews, speeches and so forth.  

    A generative model should be used given the amount of data. A 5-gram would be suitable with Katz Back-off smoothing to ensure we achieve a fair count for all data
    and so that all unseen data is not of equal probabilities. I would split the data into training and testing data at a 70:30 ratio respectively as we want to give the model enough to train on while ensuring 
    the testing set is large enough to cover a full set of possible uses of language.  

    Equations:  

    Katz P(w_i|w_i-6...w_i-1)  

    = d*(C(w_i-6...w_i-1w_i)/(C(w_i-6 ... w_i-1)) if C(w_i-6 ... w_i)>k  

    =a*(P(w_i|w_i-7...w_i-1)) otherwise  

    Where a is a weight learned in training  

    With these probabilities I would then use the entropy equation for each word: H(x)=-sum(p(x)longp(x)) 
    Then an average of every entropy to produce the final answer. 
    """)        

#############################################
# SECTION II - RESOLVING PP ATTACHMENT AMBIGUITY
#############################################

# Question 2.1 [15 marks]
class NaiveBayes:
    """
    Naive Bayes model with Lidstone smoothing (parameter alpha).
    """

    def __init__(self, data: List[Tuple[List[Any], str]], alpha: float):
        """
        :param data: A list with tuples of the form (list with features, label)
        :param alpha: \alpha value for Lidstone smoothing
        """
        self.vocab = self.get_vocab(data)
        self.alpha = alpha
        self.prior, self.likelihood = self.train(data, alpha, self.vocab)

    @staticmethod
    def get_vocab(data: List[Tuple[List[Any], str]]) -> Set[Any]:
        """
        Compute the set of all possible features from the (training) data.
        :param data: A list with tuples of the form (list with features, label)

        :return: The set of all features used in the training data for all classes.
        """
        all_features = []
        for x in data:
            for b in x[0]:
                if b not in all_features:
                    all_features.append(b)

        return all_features
        

    @staticmethod
    def train(data: List[Tuple[List[Any], str]], alpha: float, vocab: Set[Any]) -> Tuple[Dict[str, float],
          Dict[str, Dict[
          Any, float]]]:
        """
        Estimates the prior and likelihood from the data with Lidstone smoothing.

        :param data: A list of tuples ([f1, f2, ... ], c) with
                    the first element being a list of features and
                    the second element being its class.
        :param alpha: alpha value for Lidstone smoothing
        :param vocab: The set of all features used in the training data
                      for all classes.

        :return: Two dictionaries: the prior and the likelihood
                 (in that order).
        The returned values should relate as follows to the probabilities:
            prior[c] = P(c)
            likelihood[c][f] = P(f|c)
        """ 
        
        # Compute raw frequency distributions
        # use lidstone with alpha for P(f|c)
        # Compute prior (MLE). Compute likelihood with smoothing

        assert alpha >= 0.0

        likelihood, prior, classes = {}, {}, {}
        v = len(vocab)
        n = len(data)

        #get counts for number of features each class has and the total number of times each class appears

        for x in data:
            if x[1] not in classes:
                classes[x[1]] = len(x[0])
                prior[x[1]] = 1
            else:
                classes[x[1]] += len(x[0])
                prior[x[1]] += 1

        #calculte prior
        for x in prior.keys():
            prior[x] = prior[x]/n

        #give each feature in each class a starting value of alpha
        for x in classes.keys():
            likelihood[x] = {y: alpha for y in vocab}

        #add the number of each word given a class to the initial alpha value
        for x in data:
            for y in x[0]:
                likelihood[x[1]][y] += 1

        #divide by lidstone smoothing denominator for likelihood
        for x in likelihood.keys():
            for y in likelihood[x].keys():
                likelihood[x][y] = (likelihood[x][y]) / (classes[x] + alpha*v)

        return prior, likelihood

    def prob_classify(self, d: List[Any]) -> Dict[str, float]:
        """
        Compute the probability P(c|d) for all classes.
        :param d: A list of features.

        :return: The probability p(c|d) for all classes as a dictionary.
        """
    
        #Do naive bayes

        probabilty = {}
        #p(d)
        denominator = 0

        for x in self.prior.keys():
            #p(c)
            probabilty[x] = self.prior[x]
            for y in d:
                if y in self.vocab:
                   #p(c)*p(d|c)
                    probabilty[x] = probabilty[x] * self.likelihood[x].get(y,1)
            #p(d)        
            denominator += probabilty[x]
            #probabilty[c] = multiplier         

        #bayes calculation (p(c)*p(d|c))/p(d)

        for x in probabilty.keys():
            probabilty[x] /= denominator
        
        return probabilty

    def classify(self, d: List[Any]) -> str:
        """
        Compute the most likely class of the given "document" with ties broken arbitrarily.
        :param d: A list of features.

        :return: The most likely class.
        """

        prob = self.prob_classify(d)
        best_class = max(prob, key = prob.get)

        return best_class 
        

# Question 2.2 [15 marks]
def open_question_2_2() -> str:
    """
    See the Coursework 1 instructions for detail of the following:
    1) The differences in accuracy between the different ways
        to extract features?
    2) The difference between Naive Bayes vs Logistic Regression
    3) An explanation of a binary feature that returns 1
        if V=`imposed' AND N_1 = `ban' AND P=`on' AND N_2 = `uses'.

    Limit: 150 words for all three sub-questions together.
    """

    return inspect.cleandoc("""1. The differences in accuracy show that for feature extraction using a single feature, [P] is the most useful.
                            Given this, using multiple features proves better than using any sole feature. Just using individual features does not 
                            tell us about orders or combinations of features so is less accurate. 
                            2. As accuracies are similar so as Logisitic regression is a discriminative model which preforms worse on larger datasets, 
                            this may not be a contributing factor. However, logistic regression 
                            does not assume feature independence like naïve bayes does which makes NB more bias to repeated words. 
                            3. I would advocate against as it is a very specific case of language that has no guarantee to occur frequently in training and testing data. """)

# Feature extractors used in the table:

def feature_extractor_1(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("v", v)]

def feature_extractor_2(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("n1", n1)]

def feature_extractor_3(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("p", p)]

def feature_extractor_4(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("n2", n2)]

def feature_extractor_5(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("v", v), ("n1", n1), ("p", p), ("n2", n2)]


# Question 2.3, part 1 [10 marks]
def your_feature_extractor(v: str, n1: str, p:str, n2:str) -> List[Any]:
    """
    Takes the head words and produces a list of features. The features may
    be of any type as long as they are hashable.

    :param v: The verb.
    :param n1: Head of the object NP.
    :param p: The preposition.
    :param n2: Head of the NP embedded in the PP.

    :return: A list of features produced by you.
    """

    v, n1, p, n2 = v.lower(), n1.lower(), p.lower(), n2.lower()

    features = [("v", v), ("n1", n1), ("p", p), ("n2", n2), ("pn2", p + n2), ("vp", v + p), ("n1p", n1 + p),("vpn2", v + p + n2)]
    
    return features 
    

    

# Question 2.3, part 2 [10 marks]
def open_question_2_3() -> str:
    """
    Briefly describe your feature templates and your reasoning for them.
    Pick three examples of informative features and discuss why they make sense or why they do not make sense
    and why you think the model relies on them.

    There is a limit of 300 words for this question.
    """
    return inspect.cleandoc("""My feature templates include the first four basic features included previously (n1, v, p and n2) but also combinations of p with n1, v and n2. Given the importance of p alone, the combination vp increases accuracy. These combinations highlight correct grammatical terms like "attributed to". Vpn2 is also important to include as it gives "following of the" attachment to the VP instead of two smaller phrases where "following of" could have been NP. 
                            5.453 ('p', 'of')==1 and label is 'V'  
                            This feature has the highest absolute weight and therefore the model relies on this the most. This makes sense given the number of occurrences of the word “of” and where it is usually found in text with neighbouring words. 
                            -3.045 ('p', 'via')==1 and label is 'N' 
                            This feature is interesting as “via” is not a word often related to specific parts of text besides places or methods. Here we can see the N tag which is fair given the words relation to places but, in many cases, “by” can be replaced with “via” in instructions where this would become an incorrect tag.
                            4.104 ('vpn2', 'seekofinc.')==1 and label is 'V'
                            This is the second most important feature to the model and strikes me as an outlier. This is because the term “seek of inc.” is not a phrase you would think appears regularly and not one that you would depend on to make this classification. To me this does not make sense as the models choice for second most important feature in identifying categories.""")


"""
Format the output of your submission for both development and automarking. 
!!!!! DO NOT MODIFY THIS PART !!!!!
"""

def answers():
    # Global variables for answers that will be used by automarker
    global ents, lm, top10_ents, bottom10_ents
    global answer_open_question_2_2, answer_open_question_2_3
    global answer_short_1_4, answer_short_1_5, answer_short_1_3, answer_essay_question

    global naive_bayes
    global acc_extractor_1, naive_bayes_acc, lr_acc, logistic_regression_model, dev_features
    global dev_tweets_preds

    
    print("*** Part I***\n")

    print("*** Question 1.1 ***")
    print('Building Brown news bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model built')

    print("*** Question 1.2 ***")
    ents = tweet_ent(twitter_file_ids, lm)

    top10_ents = ents[:10]
    print(top10_ents)
    bottom10_ents = ents[-10:]

    answer_short_1_3 = short_answer_1_3()
    print("*** Question 1.3 ***")
    print(answer_short_1_3)

    answer_short_1_4 = short_answer_1_4()
    print("*** Question 1.4 ***")
    print(answer_short_1_4)

    answer_short_1_5 = short_answer_1_5()
    print("*** Question 1.5 ***")
    print(answer_short_1_5)

    print("*** Question 1.6 ***")
    all_dev_ok = True
    dev_tweets_preds = []
    for tweet, gold_answer in get_annotated_tweets():
        prediction = is_English(lm, tweet)
        dev_tweets_preds.append(prediction)
        if prediction != gold_answer:
            all_dev_ok = False
            print("Missclassified", tweet)
    if all_dev_ok:
        print("All development examples correctly classified! "
              "We encourage you to test and tweak your classifier on more tweets.")

    answer_essay_question = essay_question()
    print("*** Question 1.7 (essay question) ***")
    print(answer_essay_question)
    

    print("*** Part II***\n")

    print("*** Question 2.1 ***")
    naive_bayes = NaiveBayes(apply_extractor(feature_extractor_5, ppattach.tuples("training")), 0.1)
    naive_bayes_acc = compute_accuracy(naive_bayes, apply_extractor(feature_extractor_5, ppattach.tuples("devset")))
    print(f"Accuracy on the devset: {naive_bayes_acc * 100}%")

    print("*** Question 2.2 ***")
    answer_open_question_2_2 = open_question_2_2()
    print(answer_open_question_2_2)

    print("*** Question 2.3 ***")
    training_features = apply_extractor(your_feature_extractor, ppattach.tuples("training"))
    dev_features = apply_extractor(your_feature_extractor, ppattach.tuples("devset"))
    logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=10)
    lr_acc = compute_accuracy(logistic_regression_model, dev_features)

    print("30 features with highest absolute weights")
    logistic_regression_model.show_most_informative_features(30)

    print(f"Accuracy on the devset: {lr_acc*100}")

    answer_open_question_2_3 = open_question_2_3()
    print("Answer to open question:")
    print(answer_open_question_2_3)



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        from autodrive_embed import run, carefulBind
        import adrive1

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive1.extract_answers, errlog)
    else:
        answers()
