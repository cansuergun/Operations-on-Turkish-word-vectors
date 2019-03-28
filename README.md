# Operations on Turkish word vectors


- Adaptation of the assignment named 'Operations on word vectors' in Sequence Models course offered by Andrew Ng's deep learning specialization in coursera.
- Original assignment provides an incomplete code to be solved using English word vectors.
- Solved code is applied to word vectors pre-trained by Turkish word2vec model. See: https://github.com/akoksal/Turkish-Word2Vec
- Equations used in debiasing task are implemented according to the following paper:
   Bolukbasi et al., 2016, Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
   https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf
   
   ## Task 1 - Cosine similarity
   Computation of  cosine similarity  between two word vectors. It is used in tasks 2 and 3.
   ## Task 2 - Word Analogy
   Performs the Turkish version of word analogy task such as : "woman" is to "man" as "girl" is to ____. Finds the best word given the analogy. For this example,  ____ will be "boy". 
   ## Task 3 - Debiasing word vectors
         ### Task 3-a: Neutralize bias for non-gender specific words
         For the gender bias vector: g=word_vectors['kadın'] - word_vectors['adam'] ("woman" - "man"), this task creates the neutralized word vector representation of the input "hemşire" ("nurse").
         - cosine similarity between "hemşire" and g, before neutralizing:  0.050692555
         - cosine similarity between "hemşire" and g, after neutralizing:  -7.24249e-09
         ### Task 3-b: Equalization algorithm for gender-specific words
         If  "aunt"("teyze" is closer to "nurse" ("hemşire") than "uncle" ("amca"), by applying neutralizing to "nurse" ("hemşire") we can reduce the gender-stereotype associated with nursing. But this still does not guarantee that "uncle" and "aunt" are equidistant from "nurse." The equalization algorithm takes care of this.
         #### cosine similarities before equalizing:
         - cosine_similarity(word_to_vec_map["amca"], gender) =  -0.13712864
         - cosine_similarity(word_to_vec_map["teyze"], gender) =  0.014657278
         #### cosine similarities after equalizing:
         - cosine_similarity(e1, gender) =  -0.21136397
         - cosine_similarity(e2, gender) =  0.21136402
   
   
   
   
   

