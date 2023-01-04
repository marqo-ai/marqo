"""
This is a DSL for reranker problems.

We are given a string:

`
Background:
Top result: {once(0, doc.title + " " + doc.url)} # future work
{"\n" + position(doc) + "):" + doc.title + "||" + doc.content}

Question: Why are days hotter in summer?

Answer:`

Parsing algorithm:
For each substring that is between a pair of curly braces, replace it with:
    Preprocess the DSL string into a token list:
    1. Split the substring by "+" signs
    2. Strip whitespace around each element
    For each document in the results:
        For each element in the DSL list:
            1. if it's within quotes, leave it be
            2. else substitute it with relevant information from the doc
Notes:
    If a field is missing (like title doesn't exist), it is substituted
    with an empty string

Raise parsing errors
"""