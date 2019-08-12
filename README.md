# Instructions

## Files

1. bert_siamese.py - Code to train the binary/six-way classifier

2. main_attention.py - Keras code for Attention model (Need not be trained)

3. Fake_News_classification.pdf - Explanation about the architectures and techniques used

4. requirements.txt - File to install all the dependencies

5. README.md - Instructions to run the code

## Usage

Install Python3.5

Then install the requirements by running

$ pip3 install -r requirements.txt

To run the training code, first download the dataset into your HOME directory by running

$ git clone https://github.com/Tariq60/LIAR-PLUS.git ~/LIAR-PLUS

Now to run the training code for binary classification, execute

$ python3.5 bert_siamese.py -num_labels 2

Now to run the training code for 6 class classification, execute

$ python3.5 bert_siamese.py -num_labels 6
