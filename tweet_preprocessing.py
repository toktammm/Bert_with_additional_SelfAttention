# Reference: https://github.com/VinAIResearch/BERTweet

from emoji import demojize
from nltk.tokenize import TweetTokenizer
import pandas as pd


tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())

if __name__ == "__main__":
    file = open("input_tweets.csv", "r")
    Lines = file.readlines()
    splits = []
    for line in Lines:
        splits.append(line.split("\t"))

    df = pd.DataFrame(splits, columns=['text', 'label', 'voters'])
    df = df.drop_duplicates()
    df['normal'] = df.text.apply(normalizeTweet)
    df['normal_lower'] = df.normal.apply(lambda x: x.lower())

    df.to_csv("normalized_tweets.csv", index=False)
    
