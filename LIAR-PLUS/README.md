## LIAR-PLUS
The extended LIAR dataset for fact-checking and fake news detection released in our paper:
[Where is Your Evidence: Improving Fact-Checking by Justification Modeling. Tariq Alhindi, Savvas Petridis and Smaranda Muresan](http://aclweb.org/anthology/W18-5513). In Proceedings of the First Workshop on Fact Extraction and VERification (FEVER) Brussels, Belgium November 1st, 2018.
<br><br>
This dataset has evidence sentences extracted automatically from the full-text verdict report written by journalists in Politifact. Our objective is to provide a benchmark for evidence retrieval and show empirically that including evidence information in any automatic fake news detection method (regardless of features or classifier) always results in superior performance to any method lacking such information.
<br><br>
Below is the description of the TSV file taken as is from the original [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip), which was published in this [paper](https://www.aclweb.org/anthology/P17-2067). We added a new column at the end that has the extracted justification.
<br>
- Column 1: the ID of the statement ([ID].json).
- Column 2: the label.
- Column 3: the statement.
- Column 4: the subject(s).
- Column 5: the speaker.
- Column 6: the speaker's job title.
- Column 7: the state info.
- Column 8: the party affiliation.
- Columns 9-13: the total credit history count, including the current statement.
  - 9: barely true counts.
  - 10: false counts.
  - 11: half true counts.
  - 12: mostly true counts.
  - 13: pants on fire counts.
- Column 14: the context (venue / location of the speech or statement).
- **Column 15: the extracted justification**

Our justification extraction method is done as follows:
- Get all sentences in the 'Our Ruling' section of the report if it exists or get the last five sentences.
- Remove any sentence that have the verdict and any verdict-related words. Verdict-related words are provided in the forbidden words file.

**Please Note:**<br>
The dataset in the current commit is the second version which was updated after publishing the paper. We increased the list of forbidden words in the second version after realizing that we have missed a few in v1. To find the results of our experiments on v2 of the dataset, please refer to the [poster](http://www.cs.columbia.edu/~tariq/slides/2018FEVER_Where_is_your_evidence_Poster.pdf). To find the results on v1 of the dataset, please refer to the [paper](http://aclweb.org/anthology/W18-5513). V1 of the dataset can be found in this [commit](https://github.com/Tariq60/LIAR-PLUS/tree/42d9791cee78f275a9f865387b958f4f29049241).
<br><br>
Note that we do not provide the full-text verdict report in this current version of the dataset,
but you can use the following command to access the full verdict report and links to the source documents:<br>
```
wget http://www.politifact.com//api/v/2/statement/[ID]/?format=json
```
<br>
The original sources retain the copyright of the data.
Note that there are absolutely no guarantees with this data,
and we provide this dataset "as is",
but you are welcome to report the issues of the preliminary version
of this data.
<br>
You are allowed to use this dataset for research purposes only.
<br><br>
Kindly cite our paper if you find this dataset useful.

@inproceedings{alhindi2018your,<br>
  title={Where is your Evidence: Improving Fact-checking by Justification Modeling},<br>
  author={Alhindi, Tariq and Petridis, Savvas and Muresan, Smaranda},<br>
  booktitle={Proceedings of the First Workshop on Fact Extraction and VERification (FEVER)},<br>
  pages={85--90},<br>
  year={2018}<br>
}

v2.0 10/24/2018
