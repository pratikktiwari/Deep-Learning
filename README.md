# Tf-idf

**Tf denotes how frequently a term appears in a sentence**

Tf = $\frac{ Frequency-of-a-particular-word-in-a-sentence}{total-number-of-words-in-the-sentence}$

**Idf calculates the weight of rare words.**
**The words that appear rare in the corpus has a high Idf**

Idf = $\frac{\log(Total-number-of-sentences)}{Number-of-sentences-with-a-particular-word-in-it}$

**Syntax:**
``` {python}
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
doc_1 = "This is a sample text to check Tf-idf"
doc_2= "Sample text can be longer or shorter"

response = tfidf.fit_transform( [ doc_1, doc_2 ] )
print( tfidf.vocabulary_ )
print(response)

#another method
feature_names = tfidf.get_feature_names()
for col in response.nonzero()[1]:
    print(feature_names[col], " - ", response[0, col])
```
__Output__
```
{'this': 11,
 'is': 4,
 'sample': 7,
 'text': 9,
 'to': 12,
 'check': 2,
 'tf': 10,
 'idf': 3,
 'can': 1,
 'be': 0,
 'longer': 5,
 'or': 6,
 'shorter': 8}
 ```
 **Meaning:**
``` 'idf': 3
 (0, 3) 0.377...
 In 0th row, the 3rd item ('idf') has a frequency weight of 0.377..
 ```
 ```
(0, 3)	0.37762778074064174
(0, 10)	0.37762778074064174
(0, 2)	0.37762778074064174
(0, 12)	0.37762778074064174
(0, 9)	0.26868527618515564
(0, 7)	0.26868527618515564
(0, 4)	0.37762778074064174
(0, 11)	0.37762778074064174
(1, 8)	0.4078241041497786
(1, 6)	0.4078241041497786
(1, 5)	0.4078241041497786
(1, 0)	0.4078241041497786
(1, 1)	0.4078241041497786
(1, 9)	0.29017020899133733
(1, 7)	0.29017020899133733
  ```

**transform, fit_transform**
--
`fit()` : used for generating learning model parameters from training data

`transform()` : parameters generated from `fit()` method, applied upon model to generate transformed data set.

`fit_transform()` : combination of `fit()` and `transform()` api on same data set

1.**Fit():** Method calculates the parameters μ and σ and saves them as internal objects.

2.**Transform():** Method using these calculated parameters apply the transformation to a particular dataset.

3.**Fit_transform():** joins the fit() and transform() method for transformation of dataset.
