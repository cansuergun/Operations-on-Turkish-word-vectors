
"""
   - Adaptation of the assignment named 'Operations on word vectors' in Sequence Models 
     course offered by Andrew Ng's deep learning specialization in coursera.
   - Original assignment provides an incomplete code to be solved using English word vectors
   - Solved code is applied to word vectors pre-trained by Turkish word2vec model. See: https://github.com/akoksal/Turkish-Word2Vec
    
   - Equations used in debiasing task are implemented according to the following paper:
        Bolukbasi et al., 2016, Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
        https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf
"""

#------------------------------------------------------------------------------
## General imports

import numpy as np
# Load the pre-trained model as described in 
# https://github.com/akoksal/Turkish-Word2Vec/wiki/5.-Using-Word2Vec-Model-and-Examples
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('trmodel', binary=True)

#------------------------------------------------------------------------------
## Task 1 - Cosine similarity ##

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v.
    """

    
    # Compute the dot product between u and v
    dot = np.dot(u, v)
    # Compute the L2 norm of u
    norm_u = np.sqrt(np.sum(u * u))
    
    # Compute the L2 norm of v
    norm_v = np.sqrt(np.sum(v * v))
    # Compute the cosine similarity
    cosine_similarity = dot / (norm_u * norm_v)
    
    
    return cosine_similarity


# Compare cosine_similarity() with built-in similarity() function in order to check 
# if cosine_similarity() was implemented correctly
    
kadin = word_vectors["kadın"] # woman in Turkish
erkek = word_vectors["erkek"] # man in Turkish

similarity = word_vectors.similarity('kadın', 'erkek')

print("similarity(kadın, erkek) = ",similarity)
print("cosine_similarity(kadın, erkek) = ",cosine_similarity(kadin, erkek))
print("Confirmed ✓✓✓ ")


#------------------------------------------------------------------------------
## Task 2 - Word Analogy ##

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task : a is to b as c is to ____. 
    
    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_vectors -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a + v_c is close to v_best_word , as measured by cosine similarity
    """
    
    # convert words to lower case
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
    # Get the word embeddings v_a, v_b and v_c 
    e_a, e_b, e_c = word_vectors[word_a],word_vectors[word_b],word_vectors[word_c]
    
    words = list(word_vectors.vocab.keys())
    max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
    best_word = None                   # Initialize best_word with None, it will help keep track of the word to output

    # loop over the whole word vector set
    for w in words:        
        # to avoid best_word being one of the input words, pass on them.
        if w in [word_a, word_b, word_c] :
            continue
        
        # Compute cosine similarity between the vector (e_b - e_a + e_c) and the vector (w's vector representation) 
        cosine_sim = cosine_similarity(e_b - e_a + e_c, word_vectors[w])
        
        # If the cosine_sim is more than the max_cosine_sim seen so far,
            # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word 
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        
    return best_word



# Perform the analogy task and find the best word 

triads_to_try = [('italya', 'italyan', 'ispanya'), ('anne', 'baba', 'kız'), ('büyük', 'küçük', 'geniş'), 
                 ('gelmek', 'geldi', 'gitmek'),
                 ('gelmek','geliyor','gitmek'),('akıllı','zeki','neşeli'),('adam','kadın','dede')]
    

for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_vectors)))


# Results in English
    
# italy -> italian :: spain -> spanish ✓✓✓ 
# mother -> father :: girl -> boy ✓✓✓ 
# big -> small :: large -> narrow ✓✓✓ 
# to come -> came :: to go-> went ✓✓✓ 
# to come-> is coming :: to go -> is going ✓✓✓ 
# smart -> intelligent :: cheerful -> clumsy  (a synonym of cheerful was expected here )
# man-> woman :: grandfather -> dervish (grandmother was expected here)
    
    

#------------------------------------------------------------------------------
## Task 3: Debiasing word vectors ##
    
    ## Task 3-a: Neutralize bias for non-gender specific words

def neutralize(word, g, word_vectors):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
    This function ensures that gender neutral words are zero in the gender subspace.
    
    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array, corresponding to the bias axis (such as gender)
        word_vectors -- dictionary mapping words to their corresponding vectors.
    
    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """
    
    
    # Select word vector representation of "word". Use word_vectors.
    e = word_vectors[word]
    
    # Compute e_biascomponent using the formula 
    e_biascomponent = np.multiply(np.dot(e,g)/(np.linalg.norm(g)**2),g)
 
    # Neutralize e by substracting e_biascomponent from it 
    # e_debiased should be equal to its orthogonal projection.
    e_debiased = e-e_biascomponent
    
    
    return e_debiased


# set array g, corresponding to the bias axis as "woman" - "man"
g = word_vectors['kadın'] - word_vectors['adam']
# cosine similarity between "nurse" and g before neutralizing
e = "hemşire" #nurse
print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_vectors["hemşire"], g))
# cosine similarity between "nurse" and g after neutralizing
e_debiased = neutralize("hemşire", g, word_vectors)
print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))



# Next, lets see how debiasing can also be applied to word pairs such as "uncle" and "aunt." 
# Equalization is applied to pairs of words that you might want
# to have differ only through the gender property. As a concrete example, 
# suppose that "aunt" is closer to "nurse" than "uncle." By applying neutralizing
# to "nurse" we can reduce the gender-stereotype associated with nursing. 
# But this still does not guarantee that "uncle" and "aunt" are equidistant
# from "nurse." The equalization algorithm takes care of this.



    ## Task 3-b: Equalization algorithm for gender-specific words

def equalize(pair, bias_axis, word_vectors):
    """
    Debias gender specific words by following the equalize method .
    
    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
    bias_axis -- numpy-array, vector corresponding to the bias axis, e.g. gender
    word_vectors -- dictionary mapping words to their corresponding vectors
    
    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """
    
    # Step 1: Select word vector representation of "word". Use word_vectors. 
    w1, w2 = pair
    e_w1, e_w2 = word_vectors[w1],word_vectors[w2]
    
    # Step 2: Compute the mean of e_w1 and e_w2 
    mu = np.add(e_w1,e_w2)/2.0

    # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis 
    mu_B = np.multiply(np.dot(mu,bias_axis)/(np.linalg.norm(bias_axis)**2),bias_axis)
    mu_orth = mu-mu_B

    # Step 4: Compute e_w1B and e_w2B 
    e_w1B = np.multiply(np.dot(e_w1,bias_axis)/(np.linalg.norm(bias_axis)**2),bias_axis)
    e_w2B = np.multiply(np.dot(e_w2,bias_axis)/(np.linalg.norm(bias_axis)**2),bias_axis)
        
    # Step 5: Adjust the Bias part of e_w1B and e_w2B 
    corrected_e_w1B = np.multiply(np.sqrt(abs(1-np.linalg.norm(mu_orth)**2)),((e_w1B-mu_B)/np.linalg.norm(e_w1-mu_orth-mu_B)))
    corrected_e_w2B = np.multiply(np.sqrt(abs(1-np.linalg.norm(mu_orth)**2)),((e_w2B-mu_B)/np.linalg.norm(e_w2-mu_orth-mu_B)))

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections 
    e1 = corrected_e_w1B+mu_orth
    e2 = corrected_e_w2B+mu_orth
                                                                
    
    return e1, e2



# equalize words "amca" ("uncle") and "teyze" ("aunt"), making them equidistant from debiased axis 

print()
print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"amca\"], gender) = ", cosine_similarity(word_vectors["amca"], g))
print("cosine_similarity(word_to_vec_map[\"teyze\"], gender) = ", cosine_similarity(word_vectors["teyze"], g))
print()
e1, e2 = equalize(("amca", "teyze"), g, word_vectors)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))


