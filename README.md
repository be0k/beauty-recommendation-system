# Beauty Recommendation System

## Necessary Libraries
```bash
$ pip install -r requirements.txt
```

## Validation
```bash
$ python main.py --test cold --mode col --sim bert --k 10 --month 3 --m 0
```
- `test` param is only available to enter one of `zero`, `cold`, and `warm`.
- `mode` param is only available to enter one of `col`, and `content`.     
  `col` means that you wanna apply collaborative filtering. other one means content based filtering.

- `sim` param is only available to enter one of `bert`, `tfidf`, and `bm25`.
- `k` param means that the number of Items recommended. 
- `month` param is used to calculate trend based filtering.                 
If month is zero, the code will calculate whole trend instead of recent-aware trend.
- `m` param means the number of users that Similarity on Input user is high when collaborative filtering.              
If m is zero, the code will select whole users

## Api Activation
```bash
$ python api.py
$ python api_test_request.py #To Check whether Api is normal
```
api_test_request.py contains examples that notice how to use api successfully.

## Comments
experiment.ipynb file contains visualization code.


## Data Reference
```bibTeX
@inproceedings{ni-etal-2019-justifying,
    title = "Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects",
    author = "Ni, Jianmo  and
      Li, Jiacheng  and
      McAuley, Julian",
    editor = "Inui, Kentaro  and
      Jiang, Jing  and
      Ng, Vincent  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1018/",
    doi = "10.18653/v1/D19-1018",
    pages = "188--197",
    abstract = "Several recent works have considered the problem of generating reviews (or `tips') as a form of explanation as to why a recommendation might match a customer{'}s interests. While promising, we demonstrate that existing approaches struggle (in terms of both quality and content) to generate justifications that are relevant to users' decision-making process. We seek to introduce new datasets and methods to address the recommendation justification task. In terms of data, we first propose an `extractive' approach to identify review segments which justify users' intentions; this approach is then used to distantly label massive review corpora and construct large-scale personalized recommendation justification datasets. In terms of generation, we are able to design two personalized generation models with this data: (1) a reference-based Seq2Seq model with aspect-planning which can generate justifications covering different aspects, and (2) an aspect-conditional masked language model which can generate diverse justifications based on templates extracted from justification histories. We conduct experiments on two real-world datasets which show that our model is capable of generating convincing and diverse justifications."
}
```
